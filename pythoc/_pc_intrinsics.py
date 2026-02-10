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
