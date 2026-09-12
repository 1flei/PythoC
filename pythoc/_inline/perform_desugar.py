"""AST desugar for `yield effect.perform(op, *args)` inside yield producers.

`yield` performs the ambient loop effect; `yield effect.perform(op, args)`
performs a DI-routed named effect whose handler is a compile-level yield
function. Inside a yield producer it is rewritten (CPS) so the remainder
of the current block becomes the continuation of the effect call:

    x = yield effect.perform(op, args)      =>    for x in op(args):
    <rest>                                        <rest>

    yield effect.perform(op, args)          =>    for _pv in op(args):
                                                      yield _pv
                                                      <rest>

The rewrite is purely syntactic (an AST pre-pass at placeholder
creation); `effect.perform` itself is only a marker and never executes.
The `op` expression is left in place inside the generated for-iterator,
so effect resolution (overrides, usage recording) happens at lowering
time exactly as for a hand-written loop.
"""

import ast
import copy

from ..utils import get_next_id


def _is_perform_call(node) -> bool:
    """True if node is a Call to effect.perform(...)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == 'perform'
        and isinstance(func.value, ast.Name)
        and func.value.id == 'effect'
    )


def _op_call(perform_call: ast.Call) -> ast.Call:
    """effect.perform(op, *args) -> op(*args)"""
    return ast.Call(
        func=perform_call.args[0],
        args=perform_call.args[1:],
        keywords=perform_call.keywords,
    )


def _match_perform(stmt):
    """Match a perform statement; returns (target, op_call) or None.

    target is an ast.Name(Store) for the assign form, None for the
    statement form.
    """
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        value = stmt.value
        if (isinstance(value, ast.Yield) and value.value is not None
                and _is_perform_call(value.value)):
            target = stmt.targets[0]
            if not isinstance(target, ast.Name):
                from ..logger import logger
                logger.error(
                    "yield effect.perform only supports a single name target",
                    node=stmt, exc_type=SyntaxError)
            return target, _op_call(value.value)
        return None
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Yield):
        y = stmt.value
        if y.value is not None and _is_perform_call(y.value):
            return None, _op_call(y.value)
    return None


def _desugar_block(stmts):
    out = []
    i = 0
    while i < len(stmts):
        stmt = stmts[i]
        matched = _match_perform(stmt)
        if matched is None:
            out.append(_desugar_nested(stmt))
            i += 1
            continue
        target, op_call = matched
        rest = _desugar_block(stmts[i + 1:])
        if target is None:
            target = ast.Name(
                id=f"_perform_val_{get_next_id()}", ctx=ast.Store())
            body = [ast.Expr(value=ast.Yield(
                value=ast.Name(id=target.id, ctx=ast.Load())))] + rest
        else:
            body = rest if rest else [ast.Pass()]
        for_node = ast.For(
            target=target,
            iter=op_call,
            body=body,
            orelse=[],
            type_comment=None,
        )
        out.append(ast.fix_missing_locations(ast.copy_location(for_node, stmt)))
        return out
    return out


def _desugar_nested(stmt):
    """Recursively desugar performs in nested blocks."""
    for field in ('body', 'orelse', 'finalbody'):
        block = getattr(stmt, field, None)
        if isinstance(block, list) and block:
            setattr(stmt, field, _desugar_block(block))
    handlers = getattr(stmt, 'handlers', None)
    if handlers:
        for h in handlers:
            h.body = _desugar_block(h.body)
    cases = getattr(stmt, 'cases', None)
    if cases:
        for c in cases:
            c.body = _desugar_block(c.body)
    return stmt


def desugar_effect_performs(func_ast):
    """Normalize a yield producer's AST: rewrite `yield effect.perform`.

    Returns a deep copy; the input AST is not mutated. Idempotent (no
    perform markers remain after one pass).
    """
    func_ast = copy.deepcopy(func_ast)
    func_ast.body = _desugar_block(func_ast.body)
    return func_ast
