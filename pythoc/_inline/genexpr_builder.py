"""
Shared generator-expression expansion helpers.

Generator expressions are compile-time carriers in PythoC.  Both yield
for-loop inlining and instantiate lower them by replaying the expression as a
synthetic yield function.
"""

import ast
import copy
from typing import Optional

from ..logger import logger
from .names import ExpansionNames
from .renaming import ScopedVariableRenamer


def build_genexpr_yield_function_ast(
    genexpr_ast: ast.GeneratorExp,
    func_name: str,
    location_node: Optional[ast.AST] = None,
) -> ast.FunctionDef:
    """Build a synthetic yield function AST from a generator expression."""
    if not genexpr_ast.generators:
        logger.error(
            "Generator expression must contain at least one generator clause",
            node=location_node or genexpr_ast,
            exc_type=TypeError,
        )

    for comp in genexpr_ast.generators:
        if getattr(comp, "is_async", 0):
            logger.error(
                "Async generator expression is not supported",
                node=location_node or genexpr_ast,
                exc_type=TypeError,
            )

    # A comprehension target shadows only the clauses after its iterable.
    # Give it a distinct binding before lowering into ordinary for statements.
    names = ExpansionNames()
    names.reserve_ast(genexpr_ast)
    renames = {}
    generators = []
    for original in genexpr_ast.generators:
        comp = copy.deepcopy(original)
        comp.iter = ScopedVariableRenamer(renames).visit(comp.iter)
        targets = {node.id for node in ast.walk(comp.target)
                   if isinstance(node, ast.Name)}
        renames.update({name: names.fresh(f'{func_name}_{name}')
                        for name in sorted(targets)})
        renamer = ScopedVariableRenamer(renames)
        comp.target = renamer.visit(comp.target)
        comp.ifs = [renamer.visit(value) for value in comp.ifs]
        generators.append(comp)
    value = ScopedVariableRenamer(renames).visit(copy.deepcopy(genexpr_ast.elt))
    current_body = [ast.Expr(value=ast.Yield(value=value))]

    for comp in reversed(generators):
        body = current_body
        for cond in reversed(comp.ifs):
            body = [
                ast.If(
                    test=copy.deepcopy(cond),
                    body=body,
                    orelse=[],
                )
            ]

        loop_stmt = ast.For(
            target=copy.deepcopy(comp.target),
            iter=copy.deepcopy(comp.iter),
            body=body,
            orelse=[],
            type_comment=None,
        )
        current_body = [loop_stmt]

    func_ast = ast.FunctionDef(
        name=func_name,
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
            vararg=None,
            kwarg=None,
        ),
        body=current_body,
        decorator_list=[],
        returns=None,
    )
    if location_node is not None:
        ast.copy_location(func_ast, location_node)
    else:
        func_ast.lineno = getattr(genexpr_ast, "lineno", 1)
        func_ast.col_offset = getattr(genexpr_ast, "col_offset", 0)
    ast.fix_missing_locations(func_ast)
    return func_ast
