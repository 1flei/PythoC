"""
Calls mixin for LLVMIRVisitor
"""

import ast
import builtins
from typing import Optional, Any
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
)
from ..builtin_entities import bool as pc_bool
from ..registry import get_unified_registry, infer_struct_from_access
from ..logger import logger


class _MethodCallWrapper:
    """Wrapper for method calls to support unified handle_call interface"""
    def __init__(self, base_type, method_name):
        self.base_type = base_type
        self.method_name = method_name
    
    def handle_call(self, visitor, func_ref, args, node):
        """Delegate to base_type's handle_method_call"""
        return self.base_type.handle_method_call(visitor, node, self.method_name)


class CallsMixin:
    """Mixin containing calls-related visitor methods"""
    
    def visit_Call(self, node: ast.Call):
        """Handle function calls with unified duck typing approach
        
        Design principle (unified protocol):
        1. Get callable object and func_ref from node.func
        2. Pre-evaluate arguments (node.args)
        3. Delegate to handle_call(visitor, func_ref, args, node)
        
        All callables implement: handle_call(self, visitor, func_ref, args, node) -> ValueRef
        where:
        - func_ref: ValueRef of the callable (for func pointers, this is the pointer value)
        - args: list of pre-evaluated ValueRef objects
        """
        callable_obj, func_ref = self._get_callable(node.func)
        
        # Pre-evaluate arguments (unified behavior)
        # Handle struct unpacking (*struct_instance)
        args = []
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                # Struct unpacking: *struct_instance -> expand to fields
                expanded_args = self._expand_starred_struct(arg.value)
                args.extend(expanded_args)
            else:
                arg_value = self.visit_expression(arg)
                args.append(arg_value)
            
        for arg in args:
            # Transfer linear ownership for function arguments
            self._transfer_linear_ownership(arg, reason="function argument")
        
        return callable_obj.handle_call(self, func_ref, args, node)
    
    def _get_callable(self, func_node):
        """Get callable object and func_ref from function expression
        
        Extracts the object that implements handle_call protocol.
        
        Returns:
            Tuple of (callable_obj, func_ref) where:
            - callable_obj: Object with handle_call method
            - func_ref: ValueRef of the callable (for func pointers, etc.)
            
        Protocol implementers:
            - @compile/@inline/@extern functions: wrapper with handle_call
            - Function pointers: func type with handle_call
            - Builtin types: type class with handle_call (for casting)
            - Python types: PythonType instance with handle_call
            - Methods: _MethodCallWrapper with handle_call
        """
        # Evaluate the callable expression
        result = self.visit_expression(func_node)
        
        # Check if result is a type class (not ValueRef) with handle_call
        # This happens for type expressions like array[T, N], struct[...], etc.
        if isinstance(result, type) and hasattr(result, 'handle_call'):
            return result, result
    
        # Check value for handle_call (e.g., ExternFunctionWrapper, @compile wrapper)
        if hasattr(result, 'value') and hasattr(result.value, 'handle_call'):
            return result.value, result
            
        # Check type_hint for handle_call (e.g., BuiltinType, PythonType, func type)
        if hasattr(result, 'type_hint') and result.type_hint and hasattr(result.type_hint, 'handle_call'):
            return result.type_hint, result
        
        logger.error(f"Object does not support calling: {result}", node=func_node, exc_type=TypeError)
    
    def _expand_starred_struct(self, struct_expr):
        """Expand *struct_instance to individual field values
        
        Transforms:
            f(*my_struct)
        Into:
            f(my_struct.field0, my_struct.field1, ...)
        
        Args:
            struct_expr: AST node for the struct expression
        
        Returns:
            List of ValueRef objects for each field
        """
        # Evaluate the struct instance
        struct_val = self.visit_expression(struct_expr)
        
        # Get struct type information
        struct_type_hint = struct_val.type_hint
        if not struct_type_hint:
            logger.error(f"Cannot unpack value without type information", node=struct_expr, exc_type=TypeError)
        
        # Check if it's a struct type by checking for _field_types attribute
        if not hasattr(struct_type_hint, '_field_types'):
            logger.error(f"Cannot unpack non-struct type: {struct_type_hint}", node=struct_expr, exc_type=TypeError)
        
        # Get field information from the struct type directly
        field_types = struct_type_hint._field_types
        field_names = getattr(struct_type_hint, '_field_names', None)
        
        # Empty struct unpacks to no arguments
        if not field_types:
            return []
        
        # Extract each field value
        expanded_args = []
        struct_ir = ensure_ir(struct_val)
        
        for field_index, field_type in enumerate(field_types):
            # Use extractvalue to get the field directly from struct value
            field_val = self.builder.extract_value(struct_ir, field_index, name=f"field_{field_index}")
            
            # Create ValueRef with field type and tracking info
            field_ref = wrap_value(field_val, kind="value", type_hint=field_type)
            
            # If the original struct has tracking info, propagate it to the field
            if hasattr(struct_val, 'var_name') and struct_val.var_name:
                field_ref.var_name = struct_val.var_name
                # Build the linear path for this field
                base_path = getattr(struct_val, 'linear_path', ())
                field_ref.linear_path = base_path + (field_index,)
            
            expanded_args.append(field_ref)
        
        return expanded_args
    

    def _perform_call(self, node: ast.Call, func_callable, param_types, return_type_hint=None, evaluated_args=None):
        """Unified function call handler
        
        Args:
            node: ast.Call node
            func_callable: ir.Function or loaded function pointer
            param_types: List of expected parameter types (LLVM types)
            return_type_hint: Optional PC type hint for return value
            evaluated_args: Optional pre-evaluated arguments (for overloading)
        
        Returns:
            ValueRef with call result
            
        Note: ABI coercion for struct returns is handled by LLVMBuilder.call().
        """
        # Evaluate arguments (unless already evaluated for overloading)
        if evaluated_args is not None:
            args = evaluated_args
        else:
            args = [self.visit_expression(arg) for arg in node.args]
        
        # Type conversion for arguments using PC type hints when available
        converted_args = []
        for idx, (arg, expected_type) in enumerate(zip(args, param_types)):
            # Try to get PC type hint for this parameter from function registry
            target_pc_type = None
            try:
                func_name = getattr(func_callable, 'name', None)
                func_info = None
                if func_name:
                    # Prefer lookup by mangled name to preserve specialization
                    func_info = get_unified_registry().get_function_info_by_mangled(func_name) or get_unified_registry().get_function_info(func_name)
                if func_info and func_info.param_type_hints:
                    # param_types order follows function definition
                    param_names = list(func_info.param_type_hints.keys())
                    if idx < len(param_names):
                        target_pc_type = func_info.param_type_hints[param_names[idx]]
            except Exception:
                pass
            
            if target_pc_type is not None:
                converted = self.type_converter.convert(arg, target_pc_type)
                converted_args.append(ensure_ir(converted))
            else:
                # No PC hint: do not attempt LLVM-driven conversion; pass-through only if already matching
                if ensure_ir(arg).type == expected_type:
                    converted_args.append(ensure_ir(arg))
                else:
                    func_name_dbg = getattr(func_callable, 'name', '<unknown>')
                    logger.error(f"Function '{func_name_dbg}' parameter {idx} missing PC type hint; cannot convert",
                                node=node, exc_type=TypeError)
        
        # Debug: check argument count
        func_name = getattr(func_callable, 'name', '<unknown>')
        expected_param_count = len(func_callable.function_type.args)
        actual_arg_count = len(converted_args)
        if expected_param_count != actual_arg_count:
            logger.error(f"Function '{func_name}' expects {expected_param_count} arguments, got {actual_arg_count}",
                        node=node, exc_type=TypeError)
        
        # Try to get return type from FunctionInfo if not provided
        if return_type_hint is None:
            func_name = getattr(func_callable, 'name', None)
            if func_name:
                func_info = get_unified_registry().get_function_info(func_name)
                if func_info and func_info.return_type_hint:
                    return_type_hint = func_info.return_type_hint
        
        # Return type must be available
        if return_type_hint is None:
            func_name = getattr(func_callable, 'name', '<unknown>')
            logger.error(f"Cannot infer return type for function '{func_name}' - missing type hint",
                        node=node, exc_type=TypeError)
        
        # Make the call - LLVMBuilder.call() handles ABI coercion for struct returns
        logger.debug(f"_perform_call: calling {getattr(func_callable, 'name', func_callable)}, args={len(converted_args)}, return_type_hint={return_type_hint}")
        call_result = self.builder.call(func_callable, converted_args,
                                        return_type_hint=return_type_hint)
        
        # Return with type hint (tracking happens in visit_expression if needed)
        return wrap_value(call_result, kind="value", type_hint=return_type_hint)
