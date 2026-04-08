"""
Expressions mixin for LLVMIRVisitor
"""

import ast
import operator
from typing import Optional
from llvmlite import ir
from ..valueref import ValueRef, ensure_ir, wrap_value, get_type, get_type_hint
from ..builtin_entities import (
    i8, i16, i32, i64,
    u8, u16, u32, u64,
    f32, f64, ptr,
    sizeof, nullptr,
    get_builtin_entity,
    is_builtin_type,
    is_builtin_function,
    TYPE_MAP,
    is_unsigned_int,
    is_signed_int,
)
from ..builtin_entities import bool as pc_bool
from ..registry import get_unified_registry, infer_struct_from_access
from ..logger import logger
from ..type_converter import ImplicitCoercer, get_base_type


class ExpressionsMixin:
    """Mixin containing expressions-related visitor methods"""

    def visit_Name(self, node: ast.Name):
        """Handle variable references, returning ValueRef"""

        var_info = self.lookup_variable(node.id)
        if var_info:
            from ..builtin_entities.python_type import PythonType

            type_hint = var_info.type_hint
            if var_info.value_ref is not None:
                return self._bind_name_reference(var_info.value_ref, node.id)

            if isinstance(type_hint, PythonType):
                return wrap_value(
                    type_hint.get_python_object(),
                    kind="python",
                    type_hint=type_hint,
                )

            if var_info.alloca is None:
                logger.error(f"Variable '{node.id}' has no alloca and no value_ref", node=node, exc_type=RuntimeError)

            if type_hint is None:
                logger.error(f"Variable '{node.id}' has no type hint", node=node, exc_type=TypeError)

            return wrap_value(
                var_info.alloca,
                kind="address",
                type_hint=type_hint,
                address=var_info.alloca,
                var_name=node.id,
                linear_path=(),
            )
        if self.is_constexpr():
            raise NameError(f"Variable '{node.id}' not defined")
        logger.error(f"Variable '{node.id}' not defined", node=node, exc_type=NameError)
    

    def visit_Constant(self, node: ast.Constant):
        """Handle constant values - return as Python values for lazy conversion"""

        from ..builtin_entities.python_type import PythonType
        # Wrap constant as Python value with PythonType instance
        # Conversion to LLVM will happen on-demand via ensure_ir/type_converter
        python_type_inst = PythonType.wrap(node.value, is_constant=True)
        return wrap_value(node.value, kind="python", type_hint=python_type_inst)

    def visit_BinOp(self, node: ast.BinOp):
        """Handle binary operations through the unified value dispatcher."""
        left = self.visit_rvalue_expression(node.left)
        right = self.visit_rvalue_expression(node.right)
        return self.value_dispatcher.handle_binop(left, right, node)
    

    def visit_UnaryOp(self, node: ast.UnaryOp):
        """Handle unary operations through the unified value dispatcher."""
        operand = self.visit_rvalue_expression(node.operand)
        return self.value_dispatcher.handle_unary(operand, node)
    
    def _unary_plus(self, operand: ValueRef) -> ValueRef:
        """Unary plus: no-op"""
        return operand

    def _unary_minus(self, operand: ValueRef) -> ValueRef:
        """Unary minus: negate value

        Result type forgets refinement since -x does not preserve predicates.
        """
        from ..type_converter import forget_refinement

        operand_type = get_type(operand)
        if isinstance(operand_type, (ir.FloatType, ir.DoubleType)):
            result = self.builder.fsub(ir.Constant(operand_type, 0.0), ensure_ir(operand))
        else:
            result = self.builder.sub(ir.Constant(operand_type, 0), ensure_ir(operand))
        # Forget refinement for operation result
        result_type = forget_refinement(operand.type_hint)
        return wrap_value(result, kind="value", type_hint=result_type)

    def _unary_not(self, operand: ValueRef) -> ValueRef:
        """Logical not: boolean negation"""
        from ..builtin_entities import bool as bool_type
        operand_type = get_type(operand)
        if isinstance(operand_type, ir.IntType) and operand_type.width == 1:
            # Already boolean, just XOR with 1
            result = self.builder.xor(ensure_ir(operand), ir.Constant(ir.IntType(1), 1))
        else:
            # Convert to boolean first, then negate
            bool_val = self.value_dispatcher.to_boolean(operand)
            result = self.builder.xor(ensure_ir(bool_val), ir.Constant(ir.IntType(1), 1))
        return wrap_value(result, kind="value", type_hint=bool_type)

    def _unary_invert(self, operand: ValueRef) -> ValueRef:
        """Bitwise not: invert all bits

        Result type forgets refinement since ~x does not preserve predicates.
        """
        from ..type_converter import forget_refinement

        result = self.builder.xor(ensure_ir(operand), ir.Constant(get_type(operand), -1))
        # Forget refinement for operation result
        result_type = forget_refinement(operand.type_hint)
        return wrap_value(result, kind="value", type_hint=result_type)
    

    def visit_Compare(self, node: ast.Compare):
        """Handle chained comparisons through the unified value dispatcher."""
        left = self.visit_rvalue_expression(node.left)
        return self.value_dispatcher.handle_compare_chain(
            left,
            node.ops,
            node.comparators,
            node,
        )

    def visit_BoolOp(self, node: ast.BoolOp):
        """Handle boolean operations (and, or) with table-driven dispatch"""
        # Dispatch table for boolean operations
        bool_op_dispatch = {
            ast.And: ('and', False, 0),  # (label_prefix, short_circuit_on_true, short_circuit_value)
            ast.Or: ('or', True, 1),
        }
        
        op_key = type(node.op)
        if op_key not in bool_op_dispatch:
            logger.error(f"Boolean operator {type(node.op).__name__} not supported", node=node,
                        exc_type=NotImplementedError)
        
        label_prefix, short_circuit_on_true, short_circuit_value = bool_op_dispatch[op_key]
        return self._visit_short_circuit_op(node.values, label_prefix, short_circuit_on_true, short_circuit_value)
    
    def _visit_short_circuit_op(self, values, label_prefix, short_circuit_on_true, short_circuit_value):
        """Unified implementation for short-circuit boolean operations (AND/OR)
        
        Args:
            values: List of operand expressions
            label_prefix: Prefix for basic block labels ('and' or 'or')
            short_circuit_on_true: If True, short-circuit when value is true (OR logic)
                                   If False, short-circuit when value is false (AND logic)
            short_circuit_value: The value to use when short-circuiting (0 or 1)
        """
        if len(values) == 1:
            result = self.value_dispatcher.to_boolean(self.visit_expression(values[0]))
            return wrap_value(result, kind="value")

        end_block = self.builder.create_block(f"{label_prefix}_end")
        phi_incoming = []

        val = self.value_dispatcher.to_boolean(self.visit_expression(values[0]))
        first_block = self.builder.block

        if len(values) == 2:
            continue_block = self.builder.create_block(f"{label_prefix}_continue")

            if short_circuit_on_true:
                self.builder.cbranch(val, end_block, continue_block)
            else:
                self.builder.cbranch(val, continue_block, end_block)

            phi_incoming.append((ir.Constant(ir.IntType(1), short_circuit_value), first_block))

            self.builder.position_at_end(continue_block)
            val2 = self.value_dispatcher.to_boolean(self.visit_expression(values[1]))
            second_block = self.builder.block
            self.builder.branch(end_block)
            phi_incoming.append((ensure_ir(val2), second_block))
        else:
            next_block = self.builder.create_block(f"{label_prefix}_next")

            if short_circuit_on_true:
                self.builder.cbranch(val, end_block, next_block)
            else:
                self.builder.cbranch(val, next_block, end_block)
            phi_incoming.append((ir.Constant(ir.IntType(1), short_circuit_value), first_block))

            for i in range(1, len(values) - 1):
                self.builder.position_at_end(next_block)
                val = self.value_dispatcher.to_boolean(self.visit_expression(values[i]))
                current_block = self.builder.block

                next_block = self.builder.create_block(f"{label_prefix}_next")
                if short_circuit_on_true:
                    self.builder.cbranch(val, end_block, next_block)
                else:
                    self.builder.cbranch(val, next_block, end_block)
                phi_incoming.append((ir.Constant(ir.IntType(1), short_circuit_value), current_block))

            self.builder.position_at_end(next_block)
            val = self.value_dispatcher.to_boolean(self.visit_expression(values[-1]))
            last_block = self.builder.block
            self.builder.branch(end_block)
            phi_incoming.append((ensure_ir(val), last_block))
        
        # Create phi node in end block
        self.builder.position_at_end(end_block)
        phi = self.builder.phi(ir.IntType(1))
        
        for val, block in phi_incoming:
            phi.add_incoming(val, block)
        
        from ..builtin_entities import bool as bool_type
        return wrap_value(phi, kind="value", type_hint=bool_type)
    

    def visit_IfExp(self, node: ast.IfExp):
        """Handle ternary conditional expressions (a if condition else b)"""
        condition = self.value_dispatcher.to_boolean(self.visit_expression(node.test))
        
        # Create basic blocks
        then_block = self.builder.create_block("ternary_then")
        else_block = self.builder.create_block("ternary_else")
        merge_block = self.builder.create_block("ternary_merge")
        
        # Branch based on condition
        self.builder.cbranch(condition, then_block, else_block)
        
        # Generate then block
        self.builder.position_at_end(then_block)
        then_val = self.visit_rvalue_expression(node.body)
        then_block = self.builder.block  # Update in case of nested blocks
        self.builder.branch(merge_block)
        
        # Generate else block
        self.builder.position_at_end(else_block)
        else_val = self.visit_rvalue_expression(node.orelse)
        else_block = self.builder.block  # Update in case of nested blocks
        self.builder.branch(merge_block)
        
        # Merge results
        self.builder.position_at_end(merge_block)
        phi = self.builder.phi(get_type(then_val))
        phi.add_incoming(ensure_ir(then_val), then_block)
        phi.add_incoming(ensure_ir(else_val), else_block)
        
        # Extract type from then_val (prefer then_val's type)
        pc_type = getattr(then_val, 'pc_type', None)
        if pc_type is None:
            pc_type = getattr(else_val, 'pc_type', None)
        
        # If still None, try to infer from LLVM type
        if pc_type is None:
            llvm_type = get_type(phi)
            pc_type = self._infer_pc_type_from_llvm(llvm_type)
        
        return wrap_value(phi, kind="value", type_hint=pc_type)
    
    def _infer_pc_type_from_llvm(self, llvm_type):
        """Infer PC type from LLVM type"""
        from ..builtin_entities import i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool as bool_type
        
        if isinstance(llvm_type, ir.IntType):
            width = llvm_type.width
            if width == 1:
                return bool_type
            elif width == 8:
                return i8
            elif width == 16:
                return i16
            elif width == 32:
                return i32
            elif width == 64:
                return i64
        elif isinstance(llvm_type, (ir.FloatType, ir.DoubleType)):
            if llvm_type == ir.FloatType():
                return f32
            elif llvm_type == ir.DoubleType():
                return f64
        
        return None

    
    def visit_List(self, node: ast.List):
        """Handle list expressions
        
        Returns:
        - In constexpr mode: Python list (for type subscripts like func[[i32, i32], i32])
        - All elements are PC type objects (not pc_list): Python list (for type annotations)
        - Otherwise: pc_list type (for runtime array initialization)
        
        This distinction is important because:
        - Type annotations need Python lists for func parameter types
        - Runtime code needs pc_list to track ValueRefs for array conversion
        """
        from ..builtin_entities.pc_list import pc_list, PCListType
        from ..builtin_entities.base import BuiltinEntity
    
        elements = [self.visit_rvalue_expression(elt) for elt in node.elts]

        # In constexpr mode, return Python list directly (for type subscripts)
        if self.is_constexpr():
            values = [elem.value if isinstance(elem, ValueRef) else elem for elem in elements]
            return wrap_value(values, kind="python", type_hint=list)

        # Check if all elements are PC type objects (for type annotations like func[[i32, i32], i32])
        # Type objects are: BuiltinEntity subclasses (but NOT pc_list which is a value container)
        def is_pc_type_object(elem):
            if not elem.is_python_value():
                return False
            val = elem.get_python_value()
            # Check if it's a type/class that is a BuiltinEntity but NOT pc_list
            # pc_list is a value container, not a type annotation
            if isinstance(val, type):
                if issubclass(val, PCListType):
                    return False  # pc_list is not a type annotation
                if issubclass(val, BuiltinEntity):
                    return True  # Other BuiltinEntity types are type annotations
            return False
        
        if all(is_pc_type_object(elem) for elem in elements):
            # Type list - return Python list for type annotations
            values = [elem.get_python_value() for elem in elements]
            from ..builtin_entities.python_type import PythonType
            python_type_inst = PythonType.wrap(values, is_constant=True)
            return wrap_value(values, kind="python", type_hint=python_type_inst)

        # Value list (Python values, IR values, or nested pc_lists) - create pc_list
        list_type = pc_list.from_elements(elements)
        return wrap_value(list_type, kind="python", type_hint=list_type)

    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        """Handle generator expression as compile-time-only pyconst value.

        Generator expressions are represented as Python values carrying AST metadata.
        They are not lowered to runtime iterators in this stage.
        """
        import copy
        from ..builtin_entities.python_type import PythonType

        class _GeneratorExprPlaceholder:
            pass

        placeholder = _GeneratorExprPlaceholder()
        placeholder._pc_generator_expr = copy.deepcopy(node)
        placeholder._pc_is_generator_expr = True

        result = wrap_value(
            placeholder,
            kind="python",
            type_hint=PythonType.wrap(placeholder, is_constant=True)
        )
        result._pc_generator_expr_info = {
            "ast": copy.deepcopy(node),
            "kind": "generator_expression",
        }
        return result
    

    def visit_Slice(self, node: ast.Slice):
        """Handle slice syntax (x: Type) as refined[struct[...], "slice"].
        
        ast.Slice structure:
        - lower: field name (ast.Name 'x' -> string "x")
        - upper: field type (type expression like i32)
        - step: None (not used)
        
        Equivalence:
            x: i32      <=>  refined[struct[pyconst["x"], pyconst[i32]], "slice"]
            "name": f64 <=>  refined[struct[pyconst["name"], pyconst[f64]], "slice"]
        
        Returns: ValueRef representing refined[struct[...], "slice"]
        """
        from ..builtin_entities.python_type import PythonType, pyconst
        from ..builtin_entities.struct import struct
        from ..builtin_entities.refined import refined
        
        # Extract field name as STRING (not variable lookup!)
        if node.lower is None:
            logger.error("Slice must have lower bound (field name)", node=node, exc_type=TypeError)
        
        if isinstance(node.lower, ast.Name):
            # x: i32 -> "x" (Name.id as string literal)
            field_name = node.lower.id
        elif isinstance(node.lower, ast.Constant) and isinstance(node.lower.value, str):
            # "x": i32 -> "x" (already a string)
            field_name = node.lower.value
        else:
            logger.error(f"Invalid field name in slice, expected name or string: {ast.dump(node.lower)}",
                        node=node.lower, exc_type=TypeError)
        
        # Visit field type (upper bound)
        if node.upper is None:
            logger.error("Slice must have upper bound (field type)", node=node, exc_type=TypeError)
        
        field_type_ref = self.visit_expression(node.upper)
        
        # Extract the actual type from field_type_ref
        # field_type_ref is pyconst[i32], we need to get i32
        if field_type_ref.is_python_value():
            field_type = field_type_ref.value
        else:
            field_type = field_type_ref.type_hint
        
        # Create 2-element struct: struct[pyconst["x"], pyconst[field_type]]
        # Field 0: the name (as pyconst[str])
        # Field 1: the type (as pyconst[type])
        inner_struct_type = struct.handle_type_subscript((
            (None, pyconst[field_name]),
            (None, pyconst[field_type]),
        ))
        
        # Wrap as refined[struct[...], "slice"]
        refined_type = refined[inner_struct_type, "slice"]
        
        # Return as python value (this is a type, not a runtime value)
        return wrap_value(refined_type, kind="python", type_hint=PythonType.wrap(refined_type))

    def visit_Tuple(self, node: ast.Tuple):
        """Handle tuple expressions - returns struct type
        
        Creates an anonymous struct type from the tuple elements.
        For Python values, uses pyconst[value] as the field type.
        This allows tuples to be used as lightweight structs in PC code.
        
        If all fields are zero-sized (pyconst), no IR is generated and a python
        value is returned. This enables using visit_expression for type resolution.
        """
        from ..builtin_entities.struct import struct
        from ..builtin_entities.python_type import PythonType
        
        # Evaluate all elements
        elements = [self.visit_rvalue_expression(elt) for elt in node.elts]
        
        # In constexpr mode, return Python tuple directly
        if self.is_constexpr():
            values = [elem.value if isinstance(elem, ValueRef) else elem for elem in elements]
            return wrap_value(tuple(values), kind="python", type_hint=tuple)
        
        # Build struct type from element types
        # Format: ((None, type1), (None, type2), ...) for unnamed fields
        # For Python values, use pyconst[value] as the type
        field_specs = [(None, elem.get_pc_type()) for elem in elements]
        # Create anonymous struct type: struct[type1, type2, ...]
        struct_type = struct.handle_type_subscript(tuple(field_specs))
        
        # Get LLVM struct type to check if it has any runtime fields
        llvm_struct_type = struct_type.get_llvm_type(self.module.context)
        
        # If struct has no runtime fields (all pyconst), return as python value
        # This enables compile-time evaluation without IR generation
        if len(llvm_struct_type.elements) == 0:
            return wrap_value(struct_type, kind="python", type_hint=PythonType.wrap(struct_type))
        
        # Allocate struct and store elements
        struct_alloca = self.builder.alloca(llvm_struct_type, name="tuple_struct")
        
        # Store each element into the struct
        # Zero-sized fields (like pyconst) exist as {} in LLVM IR
        for i, elem in enumerate(elements):
            field_ptr = self.builder.gep(
                struct_alloca,
                [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), i)],
                name=f"tuple_field_{i}"
            )
            self.builder.store(ensure_ir(elem), field_ptr)
            
            # Transfer ownership of linear elements (they're being moved into the tuple)
            self._transfer_linear_ownership(elem, reason="tuple construction", node=node)
        
        # Load the struct value
        struct_val = self.builder.load(struct_alloca, name="tuple_val")
        
        return wrap_value(struct_val, kind="address", type_hint=struct_type, address=struct_alloca)
    

    def visit_JoinedStr(self, node: ast.JoinedStr):
        """Handle f-string expressions by converting to simple string"""
        logger.error("F-string expressions not implemented", node=node, exc_type=NotImplementedError)
    

    def _promote_to_float(self, value, target_type):
        """Promote integer value to floating point"""
        return self.type_converter.promote_to_float(value, target_type)
    

    
    def _align_pointer_comparison(self, left, right, left_ir, right_ir, node):
        """Align pointer types for comparison (null retargeting, void bridging).

        Returns (left_ir, right_ir) with matching LLVM pointer types.
        """
        IC = ImplicitCoercer

        # Fast path: same LLVM type
        if left_ir.type == right_ir.type:
            return left_ir, right_ir

        left_type = get_base_type(left.type_hint)
        right_type = get_base_type(right.type_hint)

        # Null constant: retarget to other side's type
        if IC.is_null_pointer_constant(left):
            return ir.Constant(right_ir.type, None), right_ir
        if IC.is_null_pointer_constant(right):
            return left_ir, ir.Constant(left_ir.type, None)

        # Void pointer bridging: bitcast both to i8*
        void_ptr = ir.PointerType(ir.IntType(8))
        if IC.is_void_pointer(left_type) or IC.is_void_pointer(right_type):
            l = self.builder.bitcast(left_ir, void_ptr) if left_ir.type != void_ptr else left_ir
            r = self.builder.bitcast(right_ir, void_ptr) if right_ir.type != void_ptr else right_ir
            return l, r

        # Same pointee type but different LLVM repr
        if IC.are_compatible_pointers(left_type, right_type):
            right_ir = self.builder.bitcast(right_ir, left_ir.type)
            return left_ir, right_ir

        # Reject incompatible pointers
        left_name = left_type.get_name() if hasattr(left_type, 'get_name') else str(left_type)
        right_name = right_type.get_name() if hasattr(right_type, 'get_name') else str(right_type)
        logger.error(
            f"Cannot compare incompatible pointer types '{left_name}' and '{right_name}'",
            node=node, exc_type=TypeError)

    def _perform_binary_operation(self, op: ast.operator, left: ValueRef, right: ValueRef,
                                   node: ast.AST = None) -> ValueRef:
        """Unified binary operation handler with table-driven dispatch

        This method handles all binary operations with automatic type promotion
        and unified dispatch logic, eliminating isinstance chains.

        Note: Result types have refinement forgotten - operations do not preserve
        refinement predicates since predicates are not closed under most operations.
        """
        from ..type_converter import forget_refinement

        # Check for pointer arithmetic (before type promotion)
        # Only check if values are not Python values (Python values can't be pointers)
        if not (left.is_python_value() or right.is_python_value()):
            left_type = get_type(left)
            right_type = get_type(right)

            if isinstance(op, ast.Add):
                if isinstance(left_type, ir.PointerType) and isinstance(right_type, ir.IntType):
                    left_ir = ensure_ir(left)
                    right_ir = ensure_ir(right)
                    gep_result = self.builder.gep(left_ir, [right_ir])
                    return wrap_value(gep_result, kind="value", type_hint=left.type_hint)
                elif isinstance(right_type, ir.PointerType) and isinstance(left_type, ir.IntType):
                    left_ir = ensure_ir(left)
                    right_ir = ensure_ir(right)
                    gep_result = self.builder.gep(right_ir, [left_ir])
                    return wrap_value(gep_result, kind="value", type_hint=right.type_hint)
            elif isinstance(op, ast.Sub):
                if isinstance(left_type, ir.PointerType) and isinstance(right_type, ir.IntType):
                    left_ir = ensure_ir(left)
                    right_ir = ensure_ir(right)
                    neg_right = self.builder.sub(ir.Constant(right_type, 0), right_ir)
                    gep_result = self.builder.gep(left_ir, [neg_right])
                    return wrap_value(gep_result, kind="value", type_hint=left.type_hint)

        # Unified type promotion for numeric operations
        # (Python values are auto-promoted by TypeConverter)
        left, right, is_float_op = self.type_converter.unify_binop_types(left, right)

        # Operation dispatch table: maps (op_type, is_float) to builder method
        # This eliminates the long isinstance chain
        op_dispatch = {
            (ast.Add, False): ('add', False),
            (ast.Add, True): ('fadd', False),
            (ast.Sub, False): ('sub', False),
            (ast.Sub, True): ('fsub', False),
            (ast.Mult, False): ('mul', False),
            (ast.Mult, True): ('fmul', False),
            (ast.Div, False): ('sdiv', False),
            (ast.Div, True): ('fdiv', False),
            (ast.FloorDiv, False): ('sdiv', False),
            (ast.FloorDiv, True): ('fdiv', True),  # needs floor intrinsic
            (ast.Mod, False): ('srem', False),
            (ast.Mod, True): ('frem', False),
            (ast.LShift, False): ('shl', False),
            (ast.RShift, False): ('ashr', False),
            (ast.BitOr, False): ('or_', False),
            (ast.BitXor, False): ('xor', False),
            (ast.BitAnd, False): ('and_', False),
        }

        # Special handling for power
        if isinstance(op, ast.Pow):
            result = self.builder.call(
                self._get_pow_intrinsic(get_type(ensure_ir(left))),
                [ensure_ir(left), ensure_ir(right)]
            )
            # Forget refinement for operation result
            result_type = forget_refinement(left.type_hint)
            return wrap_value(result, kind="value", type_hint=result_type)

        # Lookup and execute operation
        op_key = (type(op), is_float_op)
        if op_key not in op_dispatch:
            logger.error(f"Binary operator {type(op).__name__} not supported", node=node,
                        exc_type=NotImplementedError)

        method_name, needs_intrinsic = op_dispatch[op_key]
        builder_method = getattr(self.builder, method_name)
        result = builder_method(ensure_ir(left), ensure_ir(right))

        # Special handling for floor division with floats
        if needs_intrinsic and isinstance(op, ast.FloorDiv):
            result = self.builder.call(self._get_floor_intrinsic(get_type(result)), [result])

        # Forget refinement - result type should not retain refinement
        # since predicates are not closed under most operations
        result_type = forget_refinement(left.type_hint)
        return wrap_value(result, kind="value", type_hint=result_type)