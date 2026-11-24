from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
from typing import Mapping, Optional, Set

from ..source import copy_ast_location
from .scope_analyzer import AssignmentCollector


@dataclass(frozen=True)
class StateFieldRewritePolicy:
    fields: Mapping[str, str]
    protect_names: Set[str]
    state_arg: str = "s"
    strip_yield_expr: bool = False
    preserve_nested_functions: bool = False


class StateFieldRewriter(ast.NodeTransformer):
    def __init__(self, policy: StateFieldRewritePolicy):
        self.policy = policy
        self.fields = policy.fields
        self.protect_names = set(policy.protect_names) | {policy.state_arg}
        self.state_arg = policy.state_arg

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if hasattr(node, '_definition_symbol') or not self._should_rewrite(node.id):
            return node
        return copy_ast_location(ast.Attribute(
            value=copy_ast_location(ast.Name(id=self.state_arg, ctx=ast.Load()), node),
            attr=self.fields[node.id],
            ctx=node.ctx,
        ), node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.stmt:
        node = copy.deepcopy(node)
        if isinstance(node.target, ast.Name) and self._should_rewrite(node.target.id):
            value = node.value
            if value is None:
                value = copy_ast_location(ast.Constant(value=0), node)
            assign = ast.Assign(targets=[node.target], value=value)
            return self.generic_visit(copy_ast_location(assign, node))
        return self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> Optional[ast.stmt]:
        if self.policy.strip_yield_expr and isinstance(node.value, ast.Yield):
            return None
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Optional[ast.stmt]:
        if not self.policy.preserve_nested_functions:
            return None

        nested = copy.deepcopy(node)
        nested_protect = set(self.protect_names)
        nested_protect.update(arg.arg for arg in nested.args.args)
        nested_protect.update(arg.arg for arg in nested.args.posonlyargs)
        nested_protect.update(arg.arg for arg in nested.args.kwonlyargs)
        if nested.args.vararg:
            nested_protect.add(nested.args.vararg.arg)
        if nested.args.kwarg:
            nested_protect.add(nested.args.kwarg.arg)

        collector = AssignmentCollector()
        for stmt in nested.body:
            collector.visit(stmt)
        nested_protect.update(collector.annotated_assigned)

        nested_rewriter = StateFieldRewriter(StateFieldRewritePolicy(
            fields=self.fields,
            protect_names=nested_protect,
            state_arg=self.state_arg,
            strip_yield_expr=self.policy.strip_yield_expr,
            preserve_nested_functions=True,
        ))
        nested.body = [
            rewritten
            for stmt in nested.body
            for rewritten in _as_stmt_list(nested_rewriter.visit(stmt))
        ]
        return nested

    def _should_rewrite(self, name: str) -> bool:
        return name in self.fields and name not in self.protect_names


def _as_stmt_list(node):
    if node is None:
        return []
    if isinstance(node, list):
        return node
    return [node]
