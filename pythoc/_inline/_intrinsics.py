"""
Compiler-intrinsic namespace singleton.

Re-exports ``_intrinsic_name`` from ``pythoc._pc_intrinsics`` for convenience
and defines the lazily-populated ``_PC_INTRINSICS`` namespace.
"""

from .._pc_intrinsics import _intrinsic_name  # noqa: F401 – re-export


class _IntrinsicNamespace:
    """Namespace object that holds compiler intrinsics.

    Resolved via ``__pc_intrinsics.X`` attribute access in generated AST.
    ``PythonType.handle_attribute`` will call ``getattr(ns, attr)`` to resolve.

    Lazily populated on first attribute access to avoid circular imports.
    """
    _populated = False

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        if not _IntrinsicNamespace._populated:
            _populate_intrinsics()
        return object.__getattribute__(self, name)


# Singleton – populated lazily on first attribute access
_PC_INTRINSICS = _IntrinsicNamespace()


def _populate_intrinsics():
    """Populate the intrinsic namespace with all compiler intrinsics."""
    from ..builtin_entities import move, bool as pc_bool, label, goto_begin, goto_end, assume
    _PC_INTRINSICS.move = move
    _PC_INTRINSICS.bool = pc_bool
    _PC_INTRINSICS.label = label
    _PC_INTRINSICS.goto_begin = goto_begin
    _PC_INTRINSICS.goto_end = goto_end
    _PC_INTRINSICS.assume = assume
    _IntrinsicNamespace._populated = True
