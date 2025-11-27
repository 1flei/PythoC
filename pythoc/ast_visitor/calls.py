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
    
    def handle_call(self, visitor, node):
        """Delegate to base_type's handle_method_call"""
        return self.base_type.handle_method_call(visitor, node, self.method_name)


class CallsMixin:
    """Mixin containing calls-related visitor methods"""
    
    def visit_Call(self, node: ast.Call):
        """Handle function calls with unified duck typing approach
        
        Design principle (unified protocol):
        1. Get callable object from node.func
        2. Pre-evaluate arguments (node.args)
        3. Delegate to handle_call(visitor, args, node)
        
        All callables implement: handle_call(self, visitor, args, node) -> ValueRef
        where args is a list of pre-evaluated ValueRef objects.
        """
        callable_obj = self._get_callable(node.func)
        
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
        
        return callable_obj.handle_call(self, args, node)
    
    def _get_callable(self, func_node):
        """Get callable object from function expression
        
        Extracts the object that implements handle_call protocol.
        
        Returns:
            Object with handle_call method, or None if not found
            
        Protocol implementers:
            - @compile/@inline/@extern functions: wrapper with handle_call
            - Function pointers: func type with handle_call
            - Builtin types: type class with handle_call (for casting)
            - Python types: PythonType instance with handle_call
            - Methods: _MethodCallWrapper with handle_call
        """
        # Evaluate the callable expression
        result = self.visit_expression(func_node)
        
        import ast as ast_module
        # Check if result is a type class (not ValueRef) with handle_call
        # This happens for type expressions like array[T, N], struct[...], etc.
        if isinstance(result, type) and hasattr(result, 'handle_call'):
            return result
    
        # Check value for handle_call (e.g., ExternFunctionWrapper)
        if hasattr(result, 'value') and hasattr(result.value, 'handle_call'):
            return result.value
            
        # Check type_hint for handle_call (e.g., BuiltinType, PythonType)
        if hasattr(result, 'type_hint') and result.type_hint and hasattr(result.type_hint, 'handle_call'):
            return result.type_hint
        
        raise TypeError(f"Object does not support calling: {result}")
    
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
            raise TypeError(f"Cannot unpack value without type information")
        
        # Check if it's a struct type by checking for _field_types attribute
        if not hasattr(struct_type_hint, '_field_types'):
            raise TypeError(f"Cannot unpack non-struct type: {struct_type_hint}")
        
        # Get field information from the struct type directly
        field_types = struct_type_hint._field_types
        field_names = getattr(struct_type_hint, '_field_names', None)
        
        if not field_types:
            raise TypeError(f"Struct type {struct_type_hint} has no fields")
        
        # Extract each field value
        expanded_args = []
        struct_ir = ensure_ir(struct_val)
        
        for field_index, field_type in enumerate(field_types):
            # Use extractvalue to get the field directly from struct value
            field_val = self.builder.extract_value(struct_ir, field_index, name=f"field_{field_index}")
            
            # Create ValueRef with field type
            expanded_args.append(wrap_value(field_val, kind="value", type_hint=field_type))
        
        return expanded_args
    
    def _fallback_function_call(self, node: ast.Call):
        """Fallback for traditional function calls without handle_call"""
        func_name = None
        
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        else:
            func_name = str(node.func)
        
        # Check if it's a struct constructor
        if get_unified_registry().has_struct(func_name):
            return self._handle_struct_construction(node, func_name)
        
        # Look up variable info
        var_info = self.lookup_variable(func_name)
        if not var_info:
            raise NameError(f"Function '{func_name}' not found")
        
        # Check if function has overloading enabled
        registry = get_unified_registry()
        func_info = registry.get_function_info(func_name)
        
        # Determine the actual function name to call (mangled or original)
        actual_func_name = func_name
        evaluated_args = None  # Cache evaluated arguments for overloading
        param_llvm_types = []
        return_type_hint = None
        
        if func_info and func_info.overload_enabled:
            # Evaluate arguments to get their types
            evaluated_args = [self.visit_expression(arg) for arg in node.args]
            arg_types = []
            for arg in evaluated_args:
                if hasattr(arg, 'type_hint') and arg.type_hint:
                    arg_types.append(arg.type_hint)
                else:
                    # Missing PC type hint: cannot infer from LLVM; surface error
                    raise TypeError("Overloaded call requires PC type hints for arguments; missing type_hint")
            
            # Generate mangled name based on argument types
            from ..decorators import _mangle_function_name
            mangled_name = _mangle_function_name(func_name, arg_types)
            
            # Resolve FunctionInfo by mangled name
            mangled_info = registry.get_function_info_by_mangled(mangled_name)
            if not mangled_info:
                raise NameError(f"Function '{func_name}' with signature {[getattr(t,'get_name',lambda:str(t))() for t in arg_types]} not registered")
            
            actual_func_name = mangled_info.mangled_name
            
            # Get function from module
            try:
                func = self.module.get_global(actual_func_name)
            except KeyError:
                # If not declared yet, rely on dynamic declaration in wrappers/visitors
                func = None
            
            # Build param LLVM types in correct order
            for pn in mangled_info.param_names:
                pt = mangled_info.param_type_hints.get(pn)
                if not hasattr(pt, 'get_llvm_type'):
                    raise TypeError(f"Invalid parameter type hint for '{pn}' in function '{func_name}'")
                param_llvm_types.append(pt.get_llvm_type(self.module.context))
            return_type_hint = mangled_info.return_type_hint
        else:
            # Non-overloaded: get from module and type hints
            try:
                func = self.module.get_global(actual_func_name)
            except KeyError:
                raise NameError(f"Function '{actual_func_name}' not found in module")
            
            # Extract parameter and return types from var_info
            if not var_info.type_hint:
                raise NameError(f"Function '{func_name}' has no type hints")
            func_hints = var_info.type_hint
            if 'params' not in func_hints:
                raise TypeError(f"No parameter type hints found for function '{func_name}'")
            for param_name, pc_param_type in func_hints['params'].items():
                if not hasattr(pc_param_type, 'get_llvm_type'):
                    raise TypeError(f"Invalid parameter type hint for '{param_name}' in function '{func_name}'")
                param_llvm_types.append(pc_param_type.get_llvm_type(self.module.context))
            return_type_hint = func_hints.get('return', None)
        
        # Perform the call (pass evaluated_args if available from overloading)
        return self._perform_call(node, func or self.module.get_global(actual_func_name), param_llvm_types, return_type_hint, evaluated_args=evaluated_args)
    

    def _handle_struct_construction(self, node: ast.Call, struct_name: str):
        """Handle struct construction like TestStruct()"""
        # Get struct type from registry
        struct_info = get_unified_registry().get_struct(struct_name)
        
        if not struct_info:
            raise NameError(f"Struct '{struct_name}' not found in registry")
        
        # Get the LLVM struct type
        struct_type = self.module.context.get_identified_type(struct_name)
        
        # Create an undef value of the struct type
        # This represents an uninitialized struct value
        struct_value = ir.Constant(struct_type, ir.Undefined)
        
        # Use the struct's Python class as the type hint if available
        type_hint = None
        if struct_info.python_class is not None:
            type_hint = struct_info.python_class
        else:
            # Struct must have python_class
            raise TypeError(f"Struct '{struct_info.name}' missing python_class - cannot determine type hint")
        
        return wrap_value(struct_value, kind="value", type_hint=type_hint)
    

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
        """
        from ..logger import logger
        func_name = getattr(func_callable, 'name', '<unknown>')
        logger.debug(f"_perform_call: func={func_name}, return_type_hint={return_type_hint}, is_linear={hasattr(return_type_hint, 'is_linear') and return_type_hint.is_linear() if return_type_hint else False}")
        
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
                    raise TypeError(f"Function '{func_name_dbg}' parameter {idx} missing PC type hint; cannot convert")
        
        # Make the call
        # Debug: check argument count
        func_name = getattr(func_callable, 'name', '<unknown>')
        expected_param_count = len(func_callable.function_type.args)
        actual_arg_count = len(converted_args)
        if expected_param_count != actual_arg_count:
            raise TypeError(f"Function '{func_name}' expects {expected_param_count} arguments, got {actual_arg_count}")
        
        call_result = self.builder.call(func_callable, converted_args)
        
        # Try to get return type from FunctionInfo if not provided
        if return_type_hint is None:
            # Try to get function name from func_callable
            func_name = getattr(func_callable, 'name', None)
            if func_name:
                func_info = get_unified_registry().get_function_info(func_name)
                if func_info and func_info.return_type_hint:
                    return_type_hint = func_info.return_type_hint
        
        # Return type must be available
        if return_type_hint is None:
            func_name = getattr(func_callable, 'name', '<unknown>')
            raise TypeError(f"Cannot infer return type for function '{func_name}' - missing type hint")
        
        # Return with type hint (tracking happens in visit_expression if needed)
        result = wrap_value(call_result, kind="value", type_hint=return_type_hint)
        from ..logger import logger
        logger.debug(f"Function call result: type={return_type_hint}")
        return result
    

    def _handle_function_pointer_call(self, node: ast.Call, func_name: str, var_info):
        """Handle indirect function calls through function pointers
        
        Example: f(x, y) where f is a function pointer variable
        """
        # Load the function pointer from the variable
        func_ptr = self.builder.load(var_info.alloca)
        
        # Get the function type from the PC type hint
        pc_type = var_info.type_hint
        
        # Extract parameter and return types from func type
        if hasattr(pc_type, 'param_types') and hasattr(pc_type, 'return_type'):
            param_types = pc_type.param_types
            return_type = pc_type.return_type
        else:
            raise TypeError(f"Variable '{func_name}' is not a function pointer type")
        
        # Convert PC types to LLVM types
        param_llvm_types = []
        for pc_param_type in param_types:
            if hasattr(pc_param_type, 'get_llvm_type'):
                param_llvm_types.append(pc_param_type.get_llvm_type(self.module.context))
            else:
                raise TypeError(f"Cannot get LLVM type from {pc_param_type}")
        
        # Unified call handling
        return self._perform_call(node, func_ptr, param_llvm_types, return_type)

    def _parse_subscript_entity(self, node: ast.Subscript):
        """Parse subscript expression to get builtin entity (e.g., ptr[TreeNode])
        
        Unified implementation: delegates to type_resolver for all type parsing.
        """
        # Use type_resolver to parse the entire subscript expression
        # This handles ptr[T], array[T, N], and other parameterized types uniformly
        parsed_type = self.type_resolver.parse_annotation(node)
        return parsed_type

