from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
from typing import Optional, Set

from .scope_analyzer import AssignmentCollector


@dataclass(frozen=True)
class StateFieldRewritePolicy:
    field_names: Set[str]
    protect_names: Set[str]
    state_arg: str = "s"
    strip_yield_expr: bool = False
    preserve_nested_functions: bool = False


class StateFieldRewriter(ast.NodeTransformer):
    def __init__(self, policy: StateFieldRewritePolicy):
        self.policy = policy
        self.field_names = set(policy.field_names)
        self.protect_names = set(policy.protect_names) | {policy.state_arg}
        self.state_arg = policy.state_arg

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.protect_names or node.id not in self.field_names:
            return node
        return ast.Attribute(
            value=ast.Name(id=self.state_arg, ctx=ast.Load()),
            attr=node.id,
            ctx=node.ctx,
        )

    def visit_Assign(self, node: ast.Assign) -> ast.stmt:
        self.generic_visit(node)
        node.targets = [self._rewrite_target(t) for t in node.targets]
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.stmt:
        node = copy.deepcopy(node)
        if node.value:
            node.value = self.visit(node.value)
        if isinstance(node.target, ast.Name) and self._should_rewrite(node.target.id):
            assign = ast.Assign(
                targets=[self._rewrite_target(node.target)],
                value=node.value if node.value else ast.Constant(value=0),
            )
            ast.copy_location(assign, node)
            return assign
        return self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.stmt:
        node.value = self.visit(node.value)
        node.target = self._rewrite_target(node.target)
        return node

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
            field_names=self.field_names,
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

    def _rewrite_target(self, target: ast.AST) -> ast.AST:
        if isinstance(target, ast.Name) and self._should_rewrite(target.id):
            return ast.Attribute(
                value=ast.Name(id=self.state_arg, ctx=ast.Load()),
                attr=target.id,
                ctx=ast.Store(),
            )
        return target

    def _should_rewrite(self, name: str) -> bool:
        return name in self.field_names and name not in self.protect_names


def _as_stmt_list(node):
    if node is None:
        return []
    if isinstance(node, list):
        return node
    return [node]
