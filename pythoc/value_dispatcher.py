"""
ValueRef semantic dispatch helpers.
"""

from __future__ import annotations

import ast
import operator
from typing import Optional, Sequence

from .builtin_entities.base import BuiltinType
from .ir_helpers import safe_load
from .logger import logger
from .type_converter import get_base_type
from .valueref import ValueRef, ensure_ir, get_type, get_type_hint, wrap_value


class ValueRefDispatcher:
    """Centralize ValueRef semantic dispatch for visitor operations."""

    def __init__(self, visitor):
        self.visitor = visitor

    def _get_base_pc_type(self, value) -> Optional[type]:
        return get_base_type(getattr(value, "type_hint", None))

    @staticmethod
    def _has_type_flag(pc_type, flag_name: str) -> bool:
        return pc_type is not None and getattr(pc_type, flag_name, False)

    def _is_integer_like_pc_type(self, pc_type) -> bool:
        return self._has_type_flag(pc_type, "_is_integer") or self._has_type_flag(pc_type, "_is_bool")

    def _zero_constant_for(self, value_or_type):
        return self.visitor.type_converter.create_zero_constant(get_type(value_or_type))

    def _bool_constant(self, value: bool):
        return self.visitor.type_converter.create_bool_constant(value)

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

        base_type = get_base_type(value_ref.type_hint)
        if (
            base_type is not None
            and isinstance(base_type, type)
            and issubclass(base_type, BuiltinType)
            and base_type.is_pointer()
        ):
            return self.read_rvalue(value_ref, name=getattr(value_ref, "var_name", None))
        return value_ref

    def prepare_assignment_rvalue(self, value_ref: ValueRef, node: Optional[ast.AST] = None) -> ValueRef:
        """Apply assignment-context adaptation before storing a value."""
        value_ref = self._require_valueref(value_ref, context="assignment rvalue", node=node)
        type_hint = value_ref.get_pc_type()
        handle_assign_decay = getattr(type_hint, "handle_assign_decay", None)
        if callable(handle_assign_decay):
            return handle_assign_decay(self.visitor, value_ref)
        return value_ref

    def handle_type_call(
        self,
        func_ref: ValueRef,
        args: Sequence[ValueRef],
        node: Optional[ast.Call],
    ):
        """Dispatch a pyconst type object call through the value path."""
        func_ref = self._require_valueref(func_ref, context="type call target", node=node)
        if not func_ref.is_python_value():
            logger.error(
                f"Type call target must be a Python constant ValueRef, got {func_ref}",
                node=node,
                exc_type=TypeError,
            )

        type_obj = func_ref.get_python_value()
        handle_type_call = getattr(type_obj, "handle_type_call", None)
        if not callable(handle_type_call):
            type_name = type_obj.get_name() if hasattr(type_obj, "get_name") else str(type_obj)
            logger.error(
                f"Object does not support type calling: {type_name}",
                node=node,
                exc_type=TypeError,
            )

        return handle_type_call(self.visitor, func_ref, list(args), node)

    def explicit_cast(self, target_type, value: ValueRef, node: Optional[ast.AST] = None):
        """Perform an explicit conversion through TypeConverter."""
        value = self._require_valueref(value, context="explicit cast input", node=node)
        try:
            return self.visitor.type_converter.convert(value, target_type, node)
        except TypeError as exc:
            target_name = target_type.get_name() if hasattr(target_type, "get_name") else str(target_type)
            logger.error(f"Cannot convert to {target_name}: {exc}", node=node, exc_type=TypeError)

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

    def _evaluate_python_binop(
        self,
        left: ValueRef,
        right: ValueRef,
        op: ast.operator,
        node: Optional[ast.AST],
    ) -> ValueRef:
        """Evaluate a binary operation for two Python-backed values."""
        from .builtin_entities.python_type import PythonType

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
        python_op = python_binary_ops.get(type(op))
        if python_op is None:
            return self.visitor._perform_binary_operation(op, left, right, node)

        result = python_op(left.get_python_value(), right.get_python_value())
        python_type_inst = PythonType.wrap(result, is_constant=True)
        return wrap_value(result, kind="python", type_hint=python_type_inst)

    def handle_binary_operation(
        self,
        op: ast.operator,
        left: ValueRef,
        right: ValueRef,
        node: Optional[ast.AST] = None,
        *,
        left_node: Optional[ast.AST] = None,
        right_node: Optional[ast.AST] = None,
    ) -> ValueRef:
        """Dispatch a binary operation through the value protocol."""
        left = self._require_valueref(left, context="binary left operand", node=left_node or node)
        right = self._require_valueref(right, context="binary right operand", node=right_node or node)

        if left.is_python_value() and right.is_python_value():
            return self._evaluate_python_binop(left, right, op, node)

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
        handler_names = op_to_handler.get(type(op))
        if handler_names is None:
            return self.visitor._perform_binary_operation(op, left, right, node)

        handler_name, reverse_handler_name = handler_names
        left_handler = getattr(getattr(left, "type_hint", None), handler_name, None)
        if callable(left_handler):
            return left_handler(self.visitor, left, right, node)

        right_handler = getattr(getattr(right, "type_hint", None), reverse_handler_name, None)
        if callable(right_handler):
            return right_handler(self.visitor, left, right, node)

        return self.visitor._perform_binary_operation(op, left, right, node)

    def handle_binop(self, left: ValueRef, right: ValueRef, node: ast.BinOp) -> ValueRef:
        """Dispatch a binary operation through the value protocol."""
        return self.handle_binary_operation(
            node.op,
            left,
            right,
            node,
            left_node=node.left,
            right_node=node.right,
        )

    def _evaluate_python_unary(self, operand: ValueRef, node: ast.UnaryOp) -> ValueRef:
        """Evaluate a unary operation for a Python-backed value."""
        from .builtin_entities.python_type import PythonType

        python_unary_ops = {
            ast.UAdd: operator.pos,
            ast.USub: operator.neg,
            ast.Not: operator.not_,
            ast.Invert: operator.invert,
        }
        python_op = python_unary_ops.get(type(node.op))
        if python_op is None:
            logger.error(
                f"Unary operator {type(node.op).__name__} not supported",
                node=node,
                exc_type=NotImplementedError,
            )

        result = python_op(operand.get_python_value())
        python_type_inst = PythonType.wrap(result, is_constant=True)
        return wrap_value(result, kind="python", type_hint=python_type_inst)

    def handle_unary(self, operand: ValueRef, node: ast.UnaryOp) -> ValueRef:
        """Dispatch a unary operation through the value protocol."""
        operand = self._require_valueref(operand, context="unary operand", node=node.operand)

        if operand.is_python_value():
            return self._evaluate_python_unary(operand, node)

        from .type_converter import forget_refinement

        operand_pc_type = self._get_base_pc_type(operand)
        operand_llvm_type = get_type(operand)

        if isinstance(node.op, ast.UAdd):
            return operand

        if isinstance(node.op, ast.USub):
            zero = self._zero_constant_for(operand)
            if (
                self._has_type_flag(operand_pc_type, "_is_float")
                or self.visitor.type_converter.is_llvm_float_type(operand_llvm_type)
            ):
                result = self.visitor.builder.fsub(zero, ensure_ir(operand))
            elif (
                self._is_integer_like_pc_type(operand_pc_type)
                or self.visitor.type_converter.is_llvm_integer_type(operand_llvm_type)
            ):
                result = self.visitor.builder.sub(zero, ensure_ir(operand))
            else:
                logger.error(
                    f"Unary operator {type(node.op).__name__} not supported for {operand.type_hint}",
                    node=node,
                    exc_type=TypeError,
                )
            result_type = forget_refinement(operand.type_hint)
            return wrap_value(result, kind="value", type_hint=result_type)

        if isinstance(node.op, ast.Not):
            from .builtin_entities import bool as bool_type

            bool_val = self.to_boolean(operand, node=node)
            result = self.visitor.builder.xor(bool_val, self._bool_constant(True))
            return wrap_value(result, kind="value", type_hint=bool_type)

        if isinstance(node.op, ast.Invert):
            if not (
                self._is_integer_like_pc_type(operand_pc_type)
                or self.visitor.type_converter.is_llvm_integer_type(operand_llvm_type)
            ):
                logger.error(
                    f"Unary operator {type(node.op).__name__} not supported for {operand.type_hint}",
                    node=node,
                    exc_type=TypeError,
                )
            result = self.visitor.builder.xor(
                ensure_ir(operand),
                self.visitor.type_converter.create_int_constant(operand_llvm_type, -1),
            )
            result_type = forget_refinement(operand.type_hint)
            return wrap_value(result, kind="value", type_hint=result_type)

        logger.error(
            f"Unary operator {type(node.op).__name__} not supported",
            node=node,
            exc_type=NotImplementedError,
        )

    def to_boolean(self, value, node: Optional[ast.AST] = None):
        """Convert a value to an i1 boolean in value space."""
        if isinstance(value, ValueRef):
            value = self.read_rvalue(value, name=getattr(value, "var_name", None))

        if isinstance(value, ValueRef) and value.is_python_value():
            from .builtin_entities import bool as bool_type

            value = self.visitor.type_converter.convert(value, bool_type, node)

        pc_type = self._get_base_pc_type(value)
        llvm_type = get_type(value)
        value_ir = ensure_ir(value)

        if self._has_type_flag(pc_type, "_is_bool"):
            return value_ir
        if self._is_integer_like_pc_type(pc_type):
            return self.visitor.builder.icmp_signed(
                "!=",
                value_ir,
                self._zero_constant_for(value),
            )
        if self._has_type_flag(pc_type, "_is_float"):
            return self.visitor.builder.fcmp_ordered(
                "!=",
                value_ir,
                self._zero_constant_for(value),
            )
        if self._has_type_flag(pc_type, "_is_pointer"):
            return self.visitor.builder.icmp_unsigned(
                "!=",
                value_ir,
                self._zero_constant_for(value),
            )

        if self.visitor.type_converter.is_llvm_integer_type(llvm_type, width=1):
            return value_ir
        if self.visitor.type_converter.is_llvm_integer_type(llvm_type):
            return self.visitor.builder.icmp_signed(
                "!=",
                value_ir,
                self._zero_constant_for(value),
            )
        if self.visitor.type_converter.is_llvm_float_type(llvm_type):
            return self.visitor.builder.fcmp_ordered(
                "!=",
                value_ir,
                self._zero_constant_for(value),
            )
        if self.visitor.type_converter.is_llvm_pointer_type(llvm_type):
            return self.visitor.builder.icmp_unsigned(
                "!=",
                value_ir,
                self._zero_constant_for(value),
            )

        logger.error(f"Cannot convert {llvm_type} to boolean", node=node, exc_type=TypeError)

    def _combine_boolean_and(
        self,
        left: ValueRef,
        right: ValueRef,
        node: Optional[ast.AST] = None,
    ) -> ValueRef:
        """Combine two boolean-like values with logical and."""
        from .builtin_entities import bool as bool_type
        from .builtin_entities.python_type import PythonType

        if left.is_python_value() and right.is_python_value():
            result = bool(left.get_python_value()) and bool(right.get_python_value())
            python_type_inst = PythonType.wrap(result, is_constant=True)
            return wrap_value(result, kind="python", type_hint=python_type_inst)

        left_ir = self.to_boolean(left, node=node)
        right_ir = self.to_boolean(right, node=node)
        result = self.visitor.builder.and_(left_ir, right_ir)
        return wrap_value(result, kind="value", type_hint=bool_type)

    def handle_compare(
        self,
        op: ast.cmpop,
        left: ValueRef,
        right: ValueRef,
        node: Optional[ast.AST] = None,
    ) -> ValueRef:
        """Dispatch a comparison through value protocols and fallback lowering."""
        left = self._require_valueref(left, context="comparison left operand", node=node)
        right = self._require_valueref(right, context="comparison right operand", node=node)

        if left.is_python_value() and right.is_python_value():
            from .builtin_entities.python_type import PythonType

            left_py = left.get_python_value()
            right_py = right.get_python_value()
            if isinstance(op, ast.Lt):
                result = left_py < right_py
            elif isinstance(op, ast.LtE):
                result = left_py <= right_py
            elif isinstance(op, ast.Gt):
                result = left_py > right_py
            elif isinstance(op, ast.GtE):
                result = left_py >= right_py
            elif isinstance(op, ast.Eq):
                result = left_py == right_py
            elif isinstance(op, ast.NotEq):
                result = left_py != right_py
            elif isinstance(op, ast.Is):
                result = left_py is right_py
            elif isinstance(op, ast.IsNot):
                result = left_py is not right_py
            else:
                logger.error(
                    f"Comparison operator {type(op).__name__} not supported for Python values",
                    node=node,
                    exc_type=NotImplementedError,
                )

            python_type_inst = PythonType.wrap(result, is_constant=True)
            return wrap_value(result, kind="python", type_hint=python_type_inst)

        compare_handler = getattr(getattr(left, "type_hint", None), "handle_compare", None)
        if callable(compare_handler):
            return compare_handler(self.visitor, left, op, right, node)

        reverse_compare_handler = getattr(getattr(right, "type_hint", None), "handle_compare", None)
        if callable(reverse_compare_handler):
            return reverse_compare_handler(self.visitor, left, op, right, node)

        left, right, is_float_cmp = self.visitor.type_converter.unify_binop_types(left, right)

        cmp_dispatch = {
            (ast.Lt, False): "<",
            (ast.Lt, True): "<",
            (ast.LtE, False): "<=",
            (ast.LtE, True): "<=",
            (ast.Gt, False): ">",
            (ast.Gt, True): ">",
            (ast.GtE, False): ">=",
            (ast.GtE, True): ">=",
            (ast.Eq, False): "==",
            (ast.Eq, True): "==",
            (ast.NotEq, False): "!=",
            (ast.NotEq, True): "!=",
            (ast.Is, False): "==",
            (ast.IsNot, False): "!=",
        }

        op_key = (type(op), is_float_cmp)
        if op_key not in cmp_dispatch:
            if isinstance(op, ast.In):
                logger.error("'in' operator not yet supported", node=node, exc_type=NotImplementedError)
            if isinstance(op, ast.NotIn):
                logger.error("'not in' operator not yet supported", node=node, exc_type=NotImplementedError)
            logger.error(
                f"Comparison operator {type(op).__name__} not supported",
                node=node,
                exc_type=NotImplementedError,
            )

        predicate = cmp_dispatch[op_key]
        from .builtin_entities import bool as bool_type

        if is_float_cmp:
            result = self.visitor.builder.fcmp_ordered(
                predicate,
                ensure_ir(left),
                ensure_ir(right),
            )
            return wrap_value(result, kind="value", type_hint=bool_type)

        left_ir = ensure_ir(left)
        right_ir = ensure_ir(right)
        left_hint = self._get_base_pc_type(left) or get_type_hint(left)
        right_hint = self._get_base_pc_type(right) or get_type_hint(right)
        left_is_pointer = (
            self._has_type_flag(left_hint, "_is_pointer")
            or self.visitor.type_converter.is_llvm_pointer_type(left_ir.type)
        )
        right_is_pointer = (
            self._has_type_flag(right_hint, "_is_pointer")
            or self.visitor.type_converter.is_llvm_pointer_type(right_ir.type)
        )
        left_is_integer_like = (
            self._is_integer_like_pc_type(left_hint)
            or self.visitor.type_converter.is_llvm_integer_type(left_ir.type)
        )
        right_is_integer_like = (
            self._is_integer_like_pc_type(right_hint)
            or self.visitor.type_converter.is_llvm_integer_type(right_ir.type)
        )

        if not (left_is_integer_like or left_is_pointer):
            logger.error(
                f"Cannot compare type '{left_hint}' with icmp. "
                f"Only integers and pointers support == comparison. "
                f"For enum types, use match or extract tag with e[0].",
                node=node,
                exc_type=TypeError,
            )
        if not (right_is_integer_like or right_is_pointer):
            logger.error(
                f"Cannot compare type '{right_hint}' with icmp. "
                f"Only integers and pointers support == comparison. "
                f"For enum types, use match or extract tag with e[0].",
                node=node,
                exc_type=TypeError,
            )

        if left_is_pointer and right_is_pointer:
            left_ir, right_ir = self.visitor._align_pointer_comparison(
                left,
                right,
                left_ir,
                right_ir,
                node,
            )

        from .builtin_entities import is_unsigned_int

        use_unsigned = is_unsigned_int(left_hint) or is_unsigned_int(right_hint)
        if left_is_pointer and right_is_pointer and predicate in ("<", "<=", ">", ">="):
            use_unsigned = True
        icmp = self.visitor.builder.icmp_unsigned if use_unsigned else self.visitor.builder.icmp_signed
        result = icmp(predicate, left_ir, right_ir)
        return wrap_value(result, kind="value", type_hint=bool_type)

    def handle_compare_chain(
        self,
        left: ValueRef,
        ops: Sequence[ast.cmpop],
        comparators: Sequence[ast.AST],
        node: ast.Compare,
    ) -> ValueRef:
        """Evaluate a chained comparison through the unified compare path."""
        result = None
        current_left = left

        for op, comparator in zip(ops, comparators):
            right = comparator
            if not isinstance(right, ValueRef):
                right = self.visitor.visit_rvalue_expression(comparator)
            cmp_result = self.handle_compare(op, current_left, right, comparator)
            if result is None:
                result = cmp_result
            else:
                result = self._combine_boolean_and(result, cmp_result, node=node)
            current_left = right

        if result is None:
            logger.error("Comparison chain is empty", node=node, exc_type=ValueError)
        return result

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
