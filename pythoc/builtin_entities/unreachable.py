from .base import BuiltinFunction
from ..valueref import wrap_value
from ..logger import logger
import ast


class unreachable(BuiltinFunction):
    """unreachable() -> void

    Mark the current program point as unreachable (undefined behavior if
    actually reached): emits LLVM's ``unreachable`` terminator and ends the
    current basic block.  This is the C ``__builtin_unreachable()`` marker.
    """

    @classmethod
    def get_name(cls) -> str:
        return 'unreachable'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
        from .types import void

        if len(args) != 0:
            logger.error("unreachable() takes no arguments",
                         node=node, exc_type=TypeError)
        visitor.builder.unreachable()
        return wrap_value(None, kind='python', type_hint=void)


class _require_unreachable(BuiltinFunction):
    """Diagnose a reachable program point in the completed PythoC CFG."""

    @classmethod
    def get_name(cls) -> str:
        return '_require_unreachable'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node):
        from .types import void

        if (len(args) != 1 or not args[0].is_python_value()
                or not isinstance(args[0].get_python_value(), str)):
            logger.error("Expected a constant diagnostic string", node=node, exc_type=TypeError)
        visitor.builder.require_unreachable(args[0].get_python_value(), node)
        return wrap_value(None, kind="python", type_hint=void)
