"""
ValueRef semantic dispatch helpers.
"""

from __future__ import annotations

import ast
import operator
from typing import Optional, Sequence

from ..builtin_entities.base import BuiltinType
from ..ir_helpers import safe_load
from ..logger import logger
from ..valueref import ValueRef, ensure_ir, wrap_value


class ValueRefDispatcher:
    """Centralize ValueRef semantic dispatch for visitor operations."""

    def __init__(self, visitor):
        self.visitor = visitor

    def _require_valueref(self, value_ref, *, context: str, node: Optional[ast.AST]) -> ValueRef:
        if not isinstance(value_ref, ValueRef):
            logger.error(
                f"{context} must evaluate to a ValueRef, got {value_ref}",
                node=node,
                exc_type=TypeError,
            )
        return value_ref

    def read_rvalue(self, value_ref: ValueRef, *, name: Optional[str] = None) -> ValueRef:
        """Materialize a binding reference into an rvalue when needed."""
        value_ref = self._require_valueref(
            value_ref,
            context="rvalue expression",
            node=getattr(value_ref, "source_node", None),
        )

        if value_ref.is_python_value() or not value_ref.has_place():
            return value_ref

        place = value_ref.require_place()
        if value_ref.value is not place:
            return value_ref

        from ..type_converter import get_base_type

        base_type = get_base_type(value_ref.type_hint)
        if (
            base_type is not None
            and isinstance(base_type, type)
            and issubclass(base_type, BuiltinType)
            and base_type.is_array()
        ):
            return value_ref

        loaded_val = safe_load(
            self.visitor.builder,
            ensure_ir(place),
            value_ref.type_hint,
            name=name or getattr(value_ref, "var_name", "") or "",
        )
        return wrap_value(
            loaded_val,
            kind="address",
            type_hint=value_ref.type_hint,
            address=place,
            source_node=value_ref.source_node,
            var_name=value_ref.var_name,
            linear_path=value_ref.linear_path,
            vref_id=value_ref.vref_id,
        )

    def prepare_protocol_base(self, value_ref: ValueRef) -> ValueRef:
        """Prepare a ValueRef before protocol dispatch."""
        value_ref = self._require_valueref(
            value_ref,
            context="protocol base",
            node=getattr(value_ref, "source_node", None),
        )
        if value_ref.is_python_value():
            return value_ref

        from ..type_converter import get_base_type

        base_type = get_base_type(value_ref.type_hint)
        if (
            base_type is not None
            and isinstance(base_type, type)
            and issubclass(base_type, BuiltinType)
            and base_type.is_pointer()
        ):
            return self.read_rvalue(value_ref, name=getattr(value_ref, "var_name", None))
        return value_ref

    def _resolve_callable_protocol(
        self,
        func_ref: ValueRef,
        *,
        name: Optional[str] = None,
        node: Optional[ast.AST] = None,
    ):
        func_ref = self._require_valueref(func_ref, context="call target", node=node)
        func_ref = self.read_rvalue(func_ref, name=name)
        callable_protocol = getattr(func_ref, "type_hint", None)
        handle_call = getattr(callable_protocol, "handle_call", None)
        if not callable(handle_call):
            logger.error(
                f"Object does not support calling: {func_ref}",
                node=node,
                exc_type=TypeError,
            )
        return func_ref, callable_protocol

    def handle_call(
        self,
        func_ref: ValueRef,
        args: Sequence[ValueRef],
        node: ast.Call,
        *,
        name: Optional[str] = None,
    ) -> ValueRef:
        """Dispatch a call through the callable protocol."""
        func_ref, callable_protocol = self._resolve_callable_protocol(
            func_ref,
            name=name,
            node=node.func,
        )

        defer_linear = getattr(callable_protocol, "defer_linear_transfer", False)
        if not defer_linear:
            for arg in args:
                self.visitor._transfer_linear_ownership(
                    arg,
                    reason="function argument",
                    node=node,
                )

        return callable_protocol.handle_call(self.visitor, func_ref, args, node)

    def _evaluate_python_binop(self, left: ValueRef, right: ValueRef, node: ast.BinOp) -> ValueRef:
        """Evaluate a binary operation for two Python-backed values."""
        from ..builtin_entities.python_type import PythonType

        def c_style_floordiv(a, b):
            return int(a / b)

        def c_style_mod(a, b):
            return a - int(a / b) * b

        python_binary_ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: c_style_floordiv,
            ast.Mod: c_style_mod,
            ast.Pow: operator.pow,
            ast.LShift: operator.lshift,
            ast.RShift: operator.rshift,
            ast.BitOr: operator.or_,
            ast.BitXor: operator.xor,
            ast.BitAnd: operator.and_,
        }
        python_op = python_binary_ops.get(type(node.op))
        if python_op is None:
            return self.visitor._perform_binary_operation(node.op, left, right, node)

        result = python_op(left.get_python_value(), right.get_python_value())
        python_type_inst = PythonType.wrap(result, is_constant=True)
        return wrap_value(result, kind="python", type_hint=python_type_inst)

    def handle_binop(self, left: ValueRef, right: ValueRef, node: ast.BinOp) -> ValueRef:
        """Dispatch a binary operation through the value protocol."""
        left = self._require_valueref(left, context="binary left operand", node=node.left)
        right = self._require_valueref(right, context="binary right operand", node=node.right)

        if left.is_python_value() and right.is_python_value():
            return self._evaluate_python_binop(left, right, node)

        op_to_handler = {
            ast.Add: ("handle_add", "handle_radd"),
            ast.Sub: ("handle_sub", "handle_rsub"),
            ast.Mult: ("handle_mul", "handle_rmul"),
            ast.Div: ("handle_div", "handle_rdiv"),
            ast.FloorDiv: ("handle_floordiv", "handle_rfloordiv"),
            ast.Mod: ("handle_mod", "handle_rmod"),
            ast.Pow: ("handle_pow", "handle_rpow"),
            ast.LShift: ("handle_lshift", "handle_rlshift"),
            ast.RShift: ("handle_rshift", "handle_rrshift"),
            ast.BitOr: ("handle_bitor", "handle_rbitor"),
            ast.BitXor: ("handle_bitxor", "handle_rbitxor"),
            ast.BitAnd: ("handle_bitand", "handle_rbitand"),
        }
        handler_names = op_to_handler.get(type(node.op))
        if handler_names is None:
            return self.visitor._perform_binary_operation(node.op, left, right, node)

        handler_name, reverse_handler_name = handler_names
        left_handler = getattr(getattr(left, "type_hint", None), handler_name, None)
        if callable(left_handler):
            return left_handler(self.visitor, left, right, node)

        right_handler = getattr(getattr(right, "type_hint", None), reverse_handler_name, None)
        if callable(right_handler):
            return right_handler(self.visitor, left, right, node)

        return self.visitor._perform_binary_operation(node.op, left, right, node)

    def _resolve_attribute_protocol(self, base: ValueRef, attr_name: str, node: ast.Attribute):
        base = self._require_valueref(base, context="attribute base", node=node.value)
        base = self.prepare_protocol_base(base)

        if base.is_python_value():
            python_value = base.get_python_value()
            if isinstance(python_value, type) and getattr(python_value, "_is_enum", False):
                handle_attribute = getattr(python_value, "handle_attribute", None)
                if callable(handle_attribute):
                    return base, python_value
                logger.error(
                    f"Enum '{python_value.__name__}' has no attribute '{attr_name}' or handle_attribute method",
                    node=node,
                    exc_type=AttributeError,
                )

        value_protocol = getattr(base.value, "handle_attribute", None)
        if callable(value_protocol):
            return base, base.value

        type_protocol = getattr(base.type_hint, "handle_attribute", None)
        if callable(type_protocol):
            return base, base.type_hint

        logger.error(
            f"Object does not support attribute access: valueref: {base}",
            node=node,
            exc_type=TypeError,
        )

    def handle_attribute(self, base: ValueRef, attr_name: str, node: ast.Attribute) -> ValueRef:
        """Dispatch an attribute lookup through the attribute protocol."""
        base, attribute_protocol = self._resolve_attribute_protocol(base, attr_name, node)
        return attribute_protocol.handle_attribute(self.visitor, base, attr_name, node)

    def _resolve_subscript_protocol(self, base: ValueRef, node: ast.Subscript):
        base = self._require_valueref(base, context="subscript base", node=node.value)
        base = self.prepare_protocol_base(base)

        type_protocol = getattr(base.type_hint, "handle_subscript", None)
        if callable(type_protocol):
            return base, base.type_hint

        logger.error(
            f"Object does not support subscripting: valueref: {base}",
            node=node,
            exc_type=TypeError,
        )

    def handle_subscript(self, base: ValueRef, index: ValueRef, node: ast.Subscript) -> ValueRef:
        """Dispatch a subscript through the subscript protocol."""
        base, subscript_protocol = self._resolve_subscript_protocol(base, node)
        return subscript_protocol.handle_subscript(self.visitor, base, index, node)
