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
