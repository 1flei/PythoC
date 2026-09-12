"""
Closure adapter compatibility shim.

The actual logic lives in ``InlineAdapter`` (see ``inline_adapter.py``),
which now handles both ``@inline`` expansion and closure inlining via a
``kind`` parameter. ``ClosureAdapter`` stays as a thin wrapper so call
sites that already reach for it keep working; new code should
instantiate ``InlineAdapter(..., kind='closure')`` directly.
"""

from typing import Any, Dict, TYPE_CHECKING

from .inline_adapter import InlineAdapter

if TYPE_CHECKING:
    from ..ast_visitor.visitor_impl import LLVMIRVisitor


class ClosureAdapter(InlineAdapter):
    """Adapter for closure inlining. Equivalent to ``InlineAdapter`` with
    ``kind='closure'``.
    """

    def __init__(self, parent_visitor: 'LLVMIRVisitor',
                 param_bindings: Dict[str, Any],
                 func_globals: Dict[str, Any] = None):
        super().__init__(
            parent_visitor,
            param_bindings,
            func_globals=func_globals,
            kind="closure",
        )
