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


class CallsMixin:
    """Mixin containing calls-related visitor methods"""
    
    def visit_Call(self, node: ast.Call):
        """Handle function calls with unified call protocol.

        The visitor evaluates the callee expression and arguments, then hands the
        resulting `ValueRef` objects to `ValueRefDispatcher.handle_call()` for
        protocol resolution, rvalue materialization, and linear-transfer policy.
        """
        func_ref = self.visit_expression(node.func)

        # Pre-evaluate arguments (unified behavior)
        # Handle struct unpacking (*struct_instance)
        args = []
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                # Struct unpacking: *struct_instance -> expand to fields
                expanded_args = self._expand_starred_struct(arg.value)
                args.extend(expanded_args)
            else:
                arg_value = self.visit_rvalue_expression(arg)
                args.append(arg_value)

        return self.value_dispatcher.handle_call(
            func_ref,
            args,
            node,
            name=getattr(node.func, 'id', None),
        )
    
    def _expand_starred_struct(self, struct_expr):
        """Expand a starred carrier through the shared unpack protocol."""
        from ..literal_protocol import get_unpack_values

        struct_val = self.visit_rvalue_expression(struct_expr)
        return get_unpack_values(self, struct_val, struct_expr)
    

    def _perform_call(self, node: ast.Call, func_callable, param_types, return_type_hint=None, evaluated_args=None, param_pc_types=None):
        """Unified function call handler
        
        Args:
            node: ast.Call node
            func_callable: ir.Function or loaded function pointer
            param_types: List of expected parameter types (LLVM types)
            return_type_hint: Optional PC type hint for return value
            evaluated_args: Optional pre-evaluated arguments (for overloading)
            param_pc_types: Optional list of PC types for parameters (for type conversion)
        
        Returns:
            ValueRef with call result
            
        Note: ABI coercion for struct returns is handled by LLVMBuilder.call().
        """
        # Evaluate arguments (unless already evaluated for overloading)
        if evaluated_args is not None:
            args = evaluated_args
        else:
            args = [self.visit_expression(arg) for arg in node.args]
        
        # Type conversion for arguments using PC type hints
        converted_args = []
        for idx, (arg, expected_type) in enumerate(zip(args, param_types)):
            # Get PC type hint from param_pc_types if provided
            target_pc_type = None
            if param_pc_types and idx < len(param_pc_types):
                target_pc_type = param_pc_types[idx]
            
            if target_pc_type is not None:
                converted = self.implicit_coercer.coerce(arg, target_pc_type, node)
                converted_args.append(ensure_ir(converted))
            else:
                # No PC hint: pass-through only if already matching
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
