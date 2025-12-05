"""
Expressions mixin for LLVMIRVisitor
"""

import ast
import builtins
import operator
from typing import Optional, Any
from llvmlite import ir
from ..valueref import ValueRef, ensure_ir, wrap_value, get_type, get_type_hint
from ..ir_helpers import safe_load
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


class ExpressionsMixin:
    """Mixin containing expressions-related visitor methods"""

    def _wrap_python_to_valueref(self, name: str, value: Any) -> ValueRef:
        """Wrap a python value into valueref"""
        # If the value is already a ValueRef (e.g., nullptr), return it directly
        if isinstance(value, ValueRef):
            return value

        logger.debug(f"Converting Python constant to IR: {name} = {value}")
        if hasattr(value, 'get_value'):
            return value.get_value()
        
        from ..builtin_entities import PythonType
        python_type = PythonType.wrap(value, is_constant=True)
        return wrap_value(value, kind="python", type_hint=python_type)
    
    def visit_Name(self, node: ast.Name):
        """Handle variable references, returning ValueRef"""
        
        # Unified lookup: variables and functions
        var_info = self.lookup_variable(node.id)
        if var_info:
            # If we have a value_ref, return it directly
            if var_info.value_ref:
                if hasattr(var_info.value_ref.value, 'handle_cast'):
                    return var_info.value_ref.value.handle_cast(self, node)
                
                # Check if value has handle_call (e.g., ExternFunctionWrapper)
                if hasattr(var_info.value_ref.value, 'handle_call'):
                    return var_info.value_ref
            
            # Check if it's a Python type (for backward compatibility)
            if var_info.type_hint and hasattr(var_info.type_hint, 'is_python_type'):
                try:
                    if var_info.type_hint.is_python_type():
                        # Return Python value wrapped in ValueRef
                        python_obj = var_info.type_hint.get_python_object()
                        return wrap_value(
                            python_obj,
                            kind="python",
                            type_hint=var_info.type_hint
                        )
                except:
                    pass
            
            var = var_info.alloca
            
            # If alloca is None, check if we have a value_ref
            if var is None:
                if var_info.value_ref:
                    # Return the value_ref directly (e.g., for nullptr)
                    return var_info.value_ref
                raise RuntimeError(f"Variable '{node.id}' has no alloca and no value_ref")
            type_hint = var_info.type_hint
            # Ensure we have a type_hint
            # Note: This fallback is expected for variables without explicit type annotations
            if type_hint is None:
                raise TypeError(f"Variable '{node.id}' has no type hint")
            
            # Special case: global function (source == "function")
            if var_info.source == "function":
                # Return function as pointer value (for function pointer assignment)
                return wrap_value(var, kind="value", type_hint=type_hint)
            
            # Regular variable handling
            if isinstance(var.type, ir.PointerType):
                pointee_type = var.type.pointee
                
                # Check if type_hint is a pointer type (ptr[T])
                # If so, load it and return as pointer
                # Check for ptr class or subclass (SpecializedPtr)
                from ..builtin_entities.types import ptr as ptr_class
                is_ptr_type = (isinstance(type_hint, type) and issubclass(type_hint, ptr_class)) or \
                              (hasattr(type_hint, 'get_name') and type_hint.get_name() == 'ptr')
                if node.id == 'node':
                    pass
                if is_ptr_type:
                    loaded_val = safe_load(self.builder, ensure_ir(var), type_hint, name=f"{node.id}_val")
                    result_ref = wrap_value(loaded_val, kind="address", type_hint=type_hint, address=var,
                                          var_name=node.id, linear_path=())
                    if node.id == 'node':
                        pass
                    return result_ref
                
                # Do NOT auto-decay arrays here. Keep array variables as their alloca pointer.
                # But DO auto-load union types (which are represented as arrays)
                if isinstance(pointee_type, ir.ArrayType):
                    # Check if this is a union type (unions use ArrayType for storage)
                    from ..builtin_entities.union import union
                    if isinstance(type_hint, type) and issubclass(type_hint, union):
                        # Union type: load the value for passing to functions
                        loaded_val = safe_load(self.builder, ensure_ir(var), type_hint, name=f"{node.id}_val")
                        return wrap_value(loaded_val, kind="address", type_hint=type_hint, address=var,
                                        var_name=node.id, linear_path=())
                    # Regular array: keep as pointer, array is its own address
                    return wrap_value(var, kind="address", type_hint=type_hint, address=var,
                                    var_name=node.id, linear_path=())
                
                # Tuple (struct) variables: load the value
                # Note: For tuple subscript access, visit_subscript will handle it
                # For tuple as whole (e.g., function parameter), load it
                if isinstance(pointee_type, (ir.LiteralStructType, ir.IdentifiedStructType)):
                    loaded_val = safe_load(self.builder, ensure_ir(var), type_hint, name=f"{node.id}_val")
                    return wrap_value(loaded_val, kind="address", type_hint=type_hint, address=var,
                                    var_name=node.id, linear_path=())
                
                loaded_val = safe_load(self.builder, ensure_ir(var), type_hint, name=f"{node.id}_val")
                return wrap_value(loaded_val, kind="address", type_hint=type_hint, address=var,
                                var_name=node.id, linear_path=())
            else:
                return wrap_value(var, kind="value", type_hint=type_hint)
        
        # Check if it's in user's global namespace (constants, type aliases, etc.)
        if node.id in self.ctx.user_globals:
            value = self.ctx.user_globals[node.id]
            # Convert Python constant to LLVM constant
            return self._wrap_python_to_valueref(node.id, value)
        
        # Check if it's in builtins (print, len, etc.)
        if '__builtins__' in self.ctx.user_globals:
            builtins_val = self.ctx.user_globals['__builtins__']
            # __builtins__ can be either a dict or a module
            if isinstance(builtins_val, dict):
                if node.id in builtins_val:
                    value = builtins_val[node.id]
                    return self._wrap_python_to_valueref(node.id, value)
            elif hasattr(builtins_val, node.id):
                value = getattr(builtins_val, node.id)
                return self._wrap_python_to_valueref(node.id, value)
        
        # Otherwise raise error
        logger.debug(f"Variable '{node.id}' not found, available: {list(self.ctx.user_globals.keys())}")
        raise NameError(f"Variable '{node.id}' not defined")
    

    def visit_Constant(self, node: ast.Constant):
        """Handle constant values - return as Python values for lazy conversion"""

        from ..builtin_entities.python_type import PythonType
        # Wrap constant as Python value with PythonType instance
        # Conversion to LLVM will happen on-demand via ensure_ir/type_converter
        python_type_inst = PythonType.wrap(node.value, is_constant=True)
        return wrap_value(node.value, kind="python", type_hint=python_type_inst)

    def visit_BinOp(self, node: ast.BinOp):
        """Handle enhanced binary operations with unified protocol support"""
        left = self.visit_expression(node.left)
        right = self.visit_expression(node.right)
        
        # Check if both operands are PythonType values
        from ..builtin_entities.python_type import PythonType
        if left.is_python_value() and right.is_python_value():
            # Call Python's binary operator
            left_val = left.value
            right_val = right.value
            
            # Map AST binary op to Python operator function
            python_binary_ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.FloorDiv: operator.floordiv,
                ast.Mod: operator.mod,
                ast.Pow: operator.pow,
                ast.LShift: operator.lshift,
                ast.RShift: operator.rshift,
                ast.BitOr: operator.or_,
                ast.BitXor: operator.xor,
                ast.BitAnd: operator.and_,
            }
            
            op_key = type(node.op)

            # Execute Python operation
            result = python_binary_ops[op_key](left_val, right_val)
            
            # Wrap result as Python value
            python_type_inst = PythonType.wrap(result, is_constant=True)
            return wrap_value(result, kind="python", type_hint=python_type_inst)
        
        op_to_handler = {
            ast.Add: ('handle_add', 'handle_radd'),
            ast.Sub: ('handle_sub', 'handle_rsub'),
            ast.Mult: ('handle_mul', 'handle_rmul'),
            ast.Div: ('handle_div', 'handle_rdiv'),
            ast.FloorDiv: ('handle_floordiv', 'handle_rfloordiv'),
            ast.Mod: ('handle_mod', 'handle_rmod'),
            ast.Pow: ('handle_pow', 'handle_rpow'),
            ast.LShift: ('handle_lshift', 'handle_rlshift'),
            ast.RShift: ('handle_rshift', 'handle_rrshift'),
            ast.BitOr: ('handle_bitor', 'handle_rbitor'),
            ast.BitXor: ('handle_bitxor', 'handle_rbitxor'),
            ast.BitAnd: ('handle_bitand', 'handle_rbitand'),
        }
        
        handler_name, handler_rname = op_to_handler.get(type(node.op))
        
        if handler_name and left.type_hint and hasattr(left.type_hint, handler_name):
            handler = getattr(left.type_hint, handler_name)
            return handler(self, left, right, node)
        elif handler_rname and right.type_hint and hasattr(right.type_hint, handler_rname):
            rhandler = getattr(right.type_hint, handler_rname)
            return rhandler(self, left, right, node)
        
        # Fallback to default binary operation
        return self._perform_binary_operation(node.op, left, right)
    

    def visit_UnaryOp(self, node: ast.UnaryOp):
        """Handle unary operations with table-driven dispatch"""
        operand = self.visit_expression(node.operand)

        # Before dispatching, check if operand is a PythonType value
        from ..builtin_entities.python_type import PythonType
        if isinstance(operand.type_hint, PythonType):
            # Call Python's unary operator
            python_val = operand.value
            
            # Map AST unary op to Python operator function
            python_unary_ops = {
                ast.UAdd: operator.pos,      # +x
                ast.USub: operator.neg,      # -x
                ast.Not: operator.not_,      # not x
                ast.Invert: operator.invert, # ~x
            }
            
            op_key = type(node.op)
            
            # Execute Python operation
            result = python_unary_ops[op_key](python_val)
            
            # Wrap result as Python value
            python_type_inst = PythonType.wrap(result, is_constant=True)
            return wrap_value(result, kind="python", type_hint=python_type_inst)
        
        # Unary operation dispatch table
        unary_op_dispatch = {
            ast.UAdd: self._unary_plus,
            ast.USub: self._unary_minus,
            ast.Not: self._unary_not,
            ast.Invert: self._unary_invert,
        }
        
        op_key = type(node.op)
        if op_key not in unary_op_dispatch:
            raise NotImplementedError(f"Unary operator {type(node.op).__name__} not supported")
        
        return unary_op_dispatch[op_key](operand)
    
    def _unary_plus(self, operand: ValueRef) -> ValueRef:
        """Unary plus: no-op"""
        return operand
    
    def _unary_minus(self, operand: ValueRef) -> ValueRef:
        """Unary minus: negate value"""
        operand_type = get_type(operand)
        if isinstance(operand_type, (ir.FloatType, ir.DoubleType)):
            result = self.builder.fsub(ir.Constant(operand_type, 0.0), ensure_ir(operand))
        else:
            result = self.builder.sub(ir.Constant(operand_type, 0), ensure_ir(operand))
        return wrap_value(result, kind="value", type_hint=operand.type_hint)
    
    def _unary_not(self, operand: ValueRef) -> ValueRef:
        """Logical not: boolean negation"""
        from ..builtin_entities import bool as bool_type
        operand_type = get_type(operand)
        if isinstance(operand_type, ir.IntType) and operand_type.width == 1:
            # Already boolean, just XOR with 1
            result = self.builder.xor(ensure_ir(operand), ir.Constant(ir.IntType(1), 1))
        else:
            # Convert to boolean first, then negate
            bool_val = self._to_boolean(operand)
            result = self.builder.xor(ensure_ir(bool_val), ir.Constant(ir.IntType(1), 1))
        return wrap_value(result, kind="value", type_hint=bool_type)
    
    def _unary_invert(self, operand: ValueRef) -> ValueRef:
        """Bitwise not: invert all bits"""
        result = self.builder.xor(ensure_ir(operand), ir.Constant(get_type(operand), -1))
        return wrap_value(result, kind="value", type_hint=operand.type_hint)
    

    def visit_Compare(self, node: ast.Compare):
        """Handle enhanced comparison operations with table-driven dispatch"""
        left = self.visit_expression(node.left)
        
        # Handle multiple comparisons (a < b < c)
        result = None
        current_left = left
        
        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit_expression(comparator)
            cmp_result = self._perform_comparison(op, current_left, right)
            
            # Chain comparisons with AND
            if result is None:
                result = cmp_result
            else:
                # Perform AND using _perform_comparison logic pathway
                # If both are Python values, do Python-level AND
                from ..builtin_entities.python_type import PythonType
                if (isinstance(result, ValueRef) and result.is_python_value() and
                    isinstance(cmp_result, ValueRef) and cmp_result.is_python_value()):
                    # Python-level AND
                    result_val = result.get_python_value() and cmp_result.get_python_value()
                    python_type_inst = PythonType.wrap(result_val, is_constant=True)
                    result = wrap_value(result_val, kind="python", type_hint=python_type_inst)
                else:
                    # IR-level AND
                    result_ir = ensure_ir(result)
                    cmp_result_ir = ensure_ir(cmp_result)
                    from ..builtin_entities import bool as bool_type
                    result = wrap_value(self.builder.and_(result_ir, cmp_result_ir), kind="value", type_hint=bool_type)
            
            current_left = right
        
        return result
    

    def _perform_comparison(self, op: ast.cmpop, left: ValueRef, right: ValueRef):
        """Unified comparison handler with table-driven dispatch"""
        # Handle Python value comparisons directly (constant folding)
        if left.is_python_value() and right.is_python_value():
            from ..builtin_entities.python_type import PythonType
            left_py = left.get_python_value()
            right_py = right.get_python_value()
            
            # Perform Python-level comparison
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
                raise NotImplementedError(f"Comparison operator {type(op).__name__} not supported for Python values")
            
            # Return as Python value (for constant propagation)
            python_type_inst = PythonType.wrap(result, is_constant=True)
            return wrap_value(result, kind="python", type_hint=python_type_inst)
        
        # Unified type promotion for numeric comparisons
        # (Python values are auto-promoted by TypeConverter)
        left, right, is_float_cmp = self.type_converter.unify_binop_types(left, right)
        
        # Comparison dispatch table: maps (op_type, is_float) to comparison predicate
        cmp_dispatch = {
            (ast.Lt, False): '<',
            (ast.Lt, True): '<',
            (ast.LtE, False): '<=',
            (ast.LtE, True): '<=',
            (ast.Gt, False): '>',
            (ast.Gt, True): '>',
            (ast.GtE, False): '>=',
            (ast.GtE, True): '>=',
            (ast.Eq, False): '==',
            (ast.Eq, True): '==',
            (ast.NotEq, False): '!=',
            (ast.NotEq, True): '!=',
            (ast.Is, False): '==',
            (ast.IsNot, False): '!=',
        }
        
        op_key = (type(op), is_float_cmp)
        if op_key not in cmp_dispatch:
            if isinstance(op, ast.In):
                raise NotImplementedError("'in' operator not yet supported")
            elif isinstance(op, ast.NotIn):
                raise NotImplementedError("'not in' operator not yet supported")
            else:
                raise NotImplementedError(f"Comparison operator {type(op).__name__} not supported")
        
        predicate = cmp_dispatch[op_key]
        
        # Execute comparison
        from ..builtin_entities import bool as bool_type
        if is_float_cmp:
            result = self.builder.fcmp_ordered(predicate, ensure_ir(left), ensure_ir(right))
            return wrap_value(result, kind="value", type_hint=bool_type)
        else:
            # Choose signed or unsigned based on operand type hints
            left_hint = get_type_hint(left)
            right_hint = get_type_hint(right)
            use_unsigned = (is_unsigned_int(left_hint) or is_unsigned_int(right_hint))
            icmp = self.builder.icmp_unsigned if use_unsigned else self.builder.icmp_signed
            result = icmp(predicate, ensure_ir(left), ensure_ir(right))
            return wrap_value(result, kind="value", type_hint=bool_type)
    

    def visit_BoolOp(self, node: ast.BoolOp):
        """Handle boolean operations (and, or) with table-driven dispatch"""
        # Dispatch table for boolean operations
        bool_op_dispatch = {
            ast.And: ('and', False, 0),  # (label_prefix, short_circuit_on_true, short_circuit_value)
            ast.Or: ('or', True, 1),
        }
        
        op_key = type(node.op)
        if op_key not in bool_op_dispatch:
            raise NotImplementedError(f"Boolean operator {type(node.op).__name__} not supported")
        
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
            result = self._to_boolean(self.visit_expression(values[0]))
            return wrap_value(result, kind="value")
        
        # Create end block
        end_block = self.current_function.append_basic_block(self.get_next_label(f"{label_prefix}_end"))
        
        # Track all blocks that jump to end_block and their values
        phi_incoming = []
        
        # Evaluate first value
        val = self._to_boolean(self.visit_expression(values[0]))
        first_block = self.builder.block
        
        if len(values) == 2:
            # Simple case: just two values
            continue_block = self.current_function.append_basic_block(
                self.get_next_label(f"{label_prefix}_continue")
            )
            
            # Branch: if short_circuit condition met, jump to end; otherwise continue
            if short_circuit_on_true:
                self.builder.cbranch(val, end_block, continue_block)
            else:
                self.builder.cbranch(val, continue_block, end_block)
            
            # Short-circuit path: use short_circuit_value
            phi_incoming.append((ir.Constant(ir.IntType(1), short_circuit_value), first_block))
            
            # Continue path: evaluate second value
            self.builder.position_at_end(continue_block)
            val2 = self._to_boolean(self.visit_expression(values[1]))
            second_block = self.builder.block
            self.builder.branch(end_block)
            
            # Result is second value if we didn't short-circuit
            phi_incoming.append((ensure_ir(val2), second_block))
        else:
            # Multiple values - chain them
            next_block = self.current_function.append_basic_block(self.get_next_label(f"{label_prefix}_next"))
            
            # First value: check for short-circuit
            if short_circuit_on_true:
                self.builder.cbranch(val, end_block, next_block)
            else:
                self.builder.cbranch(val, next_block, end_block)
            phi_incoming.append((ir.Constant(ir.IntType(1), short_circuit_value), first_block))
            
            # Middle values
            for i in range(1, len(values) - 1):
                self.builder.position_at_end(next_block)
                val = self._to_boolean(self.visit_expression(values[i]))
                current_block = self.builder.block
                
                next_block = self.current_function.append_basic_block(self.get_next_label(f"{label_prefix}_next"))
                if short_circuit_on_true:
                    self.builder.cbranch(val, end_block, next_block)
                else:
                    self.builder.cbranch(val, next_block, end_block)
                phi_incoming.append((ir.Constant(ir.IntType(1), short_circuit_value), current_block))
            
            # Last value
            self.builder.position_at_end(next_block)
            val = self._to_boolean(self.visit_expression(values[-1]))
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
        condition = self._to_boolean(self.visit_expression(node.test))
        
        # Create basic blocks
        then_block = self.current_function.append_basic_block(self.get_next_label("ternary_then"))
        else_block = self.current_function.append_basic_block(self.get_next_label("ternary_else"))
        merge_block = self.current_function.append_basic_block(self.get_next_label("ternary_merge"))
        
        # Branch based on condition
        self.builder.cbranch(condition, then_block, else_block)
        
        # Generate then block
        self.builder.position_at_end(then_block)
        then_val = self.visit_expression(node.body)
        then_block = self.builder.block  # Update in case of nested blocks
        self.builder.branch(merge_block)
        
        # Generate else block
        self.builder.position_at_end(else_block)
        else_val = self.visit_expression(node.orelse)
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
        
        If all elements are Python values, return as Python list
        Otherwise, return as list of ValueRefs
        """
        from ..builtin_entities.python_type import PythonType
    
        elements = [self.visit_expression(elt) for elt in node.elts]

        if all(elem.is_python_value() for elem in elements):
            python_list = [elem.get_python_value() for elem in elements]
            python_type_inst = PythonType.wrap(python_list, is_constant=True)
            return wrap_value(python_list, kind="python", type_hint=python_type_inst)
        return elements
    

    def visit_Tuple(self, node: ast.Tuple):
        """Handle tuple expressions - returns struct[element_types...]
        
        Creates an anonymous struct type from the tuple elements.
        For Python values, uses pyconst[value] as the field type.
        This allows tuples to be used as lightweight structs in PC code.
        """
        from ..builtin_entities.struct import struct
        
        # Evaluate all elements
        elements = [self.visit_expression(elt) for elt in node.elts]
        
        # Build struct type from element types
        # Format: ((None, type1), (None, type2), ...) for unnamed fields
        # For Python values, use pyconst[value] as the type
        field_specs = [(None, elem.get_pc_type()) for elem in elements]
        # Create anonymous struct type: struct[type1, type2, ...]
        struct_type = struct.handle_type_subscript(tuple(field_specs))
        
        # Allocate struct and store elements (only non-zero-sized fields)
        llvm_struct_type = struct_type.get_llvm_type()
        struct_alloca = self.builder.alloca(llvm_struct_type, name="tuple_struct")
        
        # Store each non-pyconst element into the struct
        # pyconst fields are zero-sized and don't need storage
        llvm_field_idx = 0
        for i, elem in enumerate(elements):
            field_ptr = self.builder.gep(
                struct_alloca,
                [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), llvm_field_idx)],
                name=f"tuple_field_{i}"
            )
            self.builder.store(ensure_ir(elem), field_ptr)
            llvm_field_idx += 1
        
        # Load the struct value
        struct_val = self.builder.load(struct_alloca, name="tuple_val")
        # logger.error(f"struct_type: {struct_type}, llvm_struct_type: {llvm_struct_type}, struct_alloca: {struct_alloca}, struct_val: {struct_val}", node)
        
        return wrap_value(struct_val, kind="address", type_hint=struct_type, address=struct_alloca)
    

    def visit_JoinedStr(self, node: ast.JoinedStr):
        """Handle f-string expressions by converting to simple string"""
        raise NotImplementedError("F-string expressions not implemented")
    

    def _promote_to_float(self, value, target_type):
        """Promote integer value to floating point"""
        return self.type_converter.promote_to_float(value, target_type)
    

    def _to_boolean(self, value):
        """Convert value to boolean (i1)"""
        # Handle Python values - convert to IR first
        if isinstance(value, ValueRef) and value.is_python_value():
            # Convert Python value to i1 (bool)
            from ..builtin_entities import bool as bool_type
            value = self.type_converter.convert(value, bool_type)
        
        vtype = get_type(value)
        if isinstance(vtype, ir.IntType) and vtype.width == 1:
            return ensure_ir(value)
        elif isinstance(vtype, ir.IntType):
            return self.builder.icmp_signed('!=', ensure_ir(value), ir.Constant(vtype, 0))
        elif isinstance(vtype, (ir.FloatType, ir.DoubleType)):
            return self.builder.fcmp_ordered('!=', ensure_ir(value), ir.Constant(vtype, 0.0))
        else:
            raise TypeError(f"Cannot convert {vtype} to boolean")

    
    def _perform_binary_operation(self, op: ast.operator, left: ValueRef, right: ValueRef) -> ValueRef:
        """Unified binary operation handler with table-driven dispatch
        
        This method handles all binary operations with automatic type promotion
        and unified dispatch logic, eliminating isinstance chains.
        """
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
            return wrap_value(result, kind="value", type_hint=left.type_hint)
        
        # Lookup and execute operation
        op_key = (type(op), is_float_op)
        if op_key not in op_dispatch:
            raise NotImplementedError(f"Binary operator {type(op).__name__} not supported")
        
        method_name, needs_intrinsic = op_dispatch[op_key]
        builder_method = getattr(self.builder, method_name)
        result = builder_method(ensure_ir(left), ensure_ir(right))
        
        # Special handling for floor division with floats
        if needs_intrinsic and isinstance(op, ast.FloorDiv):
            result = self.builder.call(self._get_floor_intrinsic(get_type(result)), [result])
        
        # Result type is the unified type (left after type promotion)
        return wrap_value(result, kind="value", type_hint=left.type_hint)