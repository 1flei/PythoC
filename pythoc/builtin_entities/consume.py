from .base import BuiltinFunction
from ..valueref import wrap_value
from ..logger import logger
from ..literal_protocol import iter_literal_value_refs
import ast


class consume(BuiltinFunction):
    """consume(t: linear) -> void

    Consume a linear token, marking it as destroyed.
    The token variable becomes invalid after consumption.
    """

    @classmethod
    def get_name(cls) -> str:
        return 'consume'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
        """Handle consume(token) call.

        consume() is a no-op at IR level. The actual consumption happens in
        visit_Call when it calls _transfer_linear_ownership on the argument.

        We validate that the argument contains a linear value through the shared
        literal carrier protocol and then return void.
        """
        from .types import void

        if len(args) != 1:
            logger.error("consume() takes exactly 1 argument", node=node, exc_type=TypeError)

        arg_value = args[0]
        if not hasattr(arg_value, 'type_hint') or not arg_value.type_hint:
            logger.error(
                f"consume() argument must have type information (line {node.lineno})",
                node=node,
                exc_type=TypeError,
            )

        contains_linear = visitor._is_linear_type(arg_value.type_hint)
        if not contains_linear and arg_value.is_python_value():
            for nested_value in iter_literal_value_refs(arg_value.get_python_value()):
                if visitor._is_linear_type(nested_value.type_hint):
                    contains_linear = True
                    break

        if not contains_linear:
            logger.error(
                f"consume() requires a linear type argument (line {node.lineno})",
                node=node,
                exc_type=TypeError,
            )

        return wrap_value(None, kind='python', type_hint=void)
