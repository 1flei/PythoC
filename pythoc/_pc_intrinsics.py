"""
AST helper for compiler-intrinsic references.

This module is intentionally dependency-free (only ``ast``) so that it can be
imported from anywhere — including ``builtin_entities`` — without triggering
circular imports.
"""

import ast


def _intrinsic_name(attr: str) -> ast.Attribute:
    """Create AST node for ``__pc_intrinsics.<attr>``.

    Used by compiler-generated AST to reference intrinsics (move, label,
    goto_begin, goto_end, assume, etc.) via a single namespace object
    rather than bare names in user_globals.
    """
    return ast.Attribute(
        value=ast.Name(id='__pc_intrinsics', ctx=ast.Load()),
        attr=attr,
        ctx=ast.Load()
    )


class _BoundValue:
    """Preserve one evaluated reference when copying quoted syntax."""

    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def __deepcopy__(self, memo):
        return self


def _bound_value_expression(value) -> ast.Call:
    expr = ast.Call(func=_intrinsic_name('bound_value'), args=[], keywords=[])
    expr._pc_bound_value = _BoundValue(value)
    return expr


def _get_bound_value(expr):
    """Read the compiler payload attached to a bound value expression."""
    binding = getattr(expr, '_pc_bound_value', None)
    return binding.value if binding is not None else None
