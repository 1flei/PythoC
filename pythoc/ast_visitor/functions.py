"""
Functions mixin for LLVMIRVisitor
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
from ..builder import LLVMBuilder


class FunctionsMixin:
    """Mixin containing functions-related visitor methods"""
    
    def visit_function(self, node: ast.FunctionDef):
        """Visit and generate LLVM IR for a function definition"""
        # Get function from module (should already be declared)
        # Check if function has mangled name (for overloading)
        from ..registry import get_unified_registry
        registry = get_unified_registry()
        func_info = registry.get_function_info(node.name)
        func_name = func_info.mangled_name if func_info and func_info.mangled_name else node.name
        
        func = self.module.get_global(func_name)
        if func is None:
            from ..logger import logger
            logger.error(f"Function {func_name} not found in module", node=node, exc_type=ValueError)
        
        # Store current function for block creation
        self.current_function = func
        
        # Detect and store varargs information for this function
        from ..ast_visitor.varargs import detect_varargs
        varargs_kind, element_types, varargs_name = detect_varargs(node, self.type_resolver)
        self.current_varargs_info = None
        if varargs_kind in ('union', 'enum', 'none'):
            # This function has runtime varargs - store info for args[i] access
            element_pc_types = []
            if element_types:
                for elem_type_ast in element_types:
                    pc_type = self.type_resolver.parse_annotation(elem_type_ast)
                    if pc_type:
                        element_pc_types.append(pc_type)
            self.current_varargs_info = {
                'kind': varargs_kind,
                'name': varargs_name,
                'element_types': element_pc_types,
                'va_list': None  # Will be initialized on first access
            }
        
        # Create entry block
        entry_block = func.append_basic_block('entry')
        ir_builder = ir.IRBuilder(entry_block)
        self.builder = LLVMBuilder(ir_builder)
        
        # Set ABI context for struct returns
        sret_info = self.func_type_hints.get('_sret_info')
        param_coercion_info = self.func_type_hints.get('_param_coercion_info', {})
        self.builder.set_return_abi_context(func, sret_info)
        
        # Wrap function to hide sret offset and handle ABI coercion
        from ..builder.llvm_builder import FunctionWrapper
        func_wrapper = FunctionWrapper(func, sret_info, param_coercion_info)
        
        # Store parameter values in local variables
        for i, arg in enumerate(node.args.args):
            # Get unpacked parameter value (handles ABI coercion transparently)
            param_val, param_type = func_wrapper.get_user_arg_unpacked(i, self.builder.ir_builder)
            func_wrapper.get_user_arg(i).name = arg.arg
            
            # Allocate and store parameter
            alloca = self._create_alloca_in_entry(param_type, f"{arg.arg}_addr")
            self.builder.store(param_val, alloca)
            
            # Register parameter in variable registry with type hint
            type_hint = None
            if arg.annotation:
                type_hint = self.get_pc_type_from_annotation(arg.annotation)
            
            # Create ValueRef with proper wrapper for function pointers
            from ..registry import VariableInfo
            from ..valueref import ValueRef
            from ..builtin_entities.func import func
            
            # Check if this is a function pointer parameter
            if type_hint and isinstance(type_hint, type) and issubclass(type_hint, func):
                # Store alloca directly, func.handle_call will load it when needed
                value_ref = wrap_value(
                    alloca,
                    kind='address',
                    type_hint=type_hint,
                    address=alloca
                )
            else:
                value_ref = wrap_value(
                    alloca,
                    kind='address',
                    type_hint=type_hint,
                    address=alloca
                )
            
            var_info = VariableInfo(
                name=arg.arg,
                value_ref=value_ref,
                alloca=alloca,
                source="parameter",
                is_parameter=True,
                is_mutable=True
            )
            self.ctx.var_registry.declare(var_info, allow_shadow=True)
            
            # Initialize linear states for parameters (active state = ownership transferred)
            if self._is_linear_type(type_hint):
                self._init_linear_states(var_info, type_hint, initial_state='active')
        
        # Visit function body statements
        for stmt in node.body:
            self.visit(stmt)
        
        # Ensure function has a return statement
        if not self.builder.block.is_terminated:
            if func.return_value.type == ir.VoidType():
                self.builder.ret_void()
            else:
                # Add a default return for non-void functions
                ret_type = func.return_value.type
                if isinstance(ret_type, ir.PointerType):
                    # Return null pointer for pointer types
                    self.builder.ret(ir.Constant(ret_type, None))
                elif isinstance(ret_type, ir.IntType):
                    # Return 0 for integer types
                    self.builder.ret(ir.Constant(ret_type, 0))
                elif isinstance(ret_type, (ir.FloatType, ir.DoubleType)):
                    # Return 0.0 for float types
                    self.builder.ret(ir.Constant(ret_type, 0.0))
                elif isinstance(ret_type, (ir.LiteralStructType, ir.IdentifiedStructType)):
                    # Return undefined for struct types
                    self.builder.ret(ir.Constant(ret_type, ir.Undefined))
                elif isinstance(ret_type, ir.ArrayType):
                    # Return undefined for array types
                    self.builder.ret(ir.Constant(ret_type, ir.Undefined))
                else:
                    # Fallback: return undefined
                    self.builder.ret(ir.Constant(ret_type, ir.Undefined))
        
        # Check for any unterminated blocks in the function and fix them
        for block in func.blocks:
            if not block.is_terminated:
                # Position builder at the unterminated block
                temp_builder = ir.IRBuilder(block)
                # Add unreachable instruction to make it valid
                temp_builder.unreachable()
    

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Handle function definition in AST traversal
        
        If encountered within a function body (nested), treat as closure.
        Otherwise, treat as top-level function definition.
        """
        # Check if this is a nested function (closure)
        if self.current_function is not None:
            # This is a closure - register it as a callable
            self._register_closure(node)
            return None
        else:
            # Top-level function
            return self.visit_function(node)
    
    def _register_closure(self, node: ast.FunctionDef):
        """Register a closure function for inline execution
        
        Creates a handle_call wrapper that uses ClosureAdapter to inline
        the closure body when called.
        """
        from ..inline import ClosureAdapter
        from ..valueref import ValueRef
        from ..registry import VariableInfo
        
        func_name = node.name
        
        # Capture the current user_globals at closure definition time
        # This is the caller's globals context
        closure_globals = self.ctx.user_globals
        
        # Create a closure wrapper with handle_call
        class ClosureWrapper:
            def __init__(self, func_ast, visitor, func_globals):
                self.func_ast = func_ast
                self.visitor = visitor
                self.func_globals = func_globals
            
            def handle_call(self, visitor, func_ref, args, call_node):
                """Execute closure inline using ClosureAdapter"""
                # Build parameter bindings
                param_names = [arg.arg for arg in self.func_ast.args.args]
                if len(args) != len(param_names):
                    from ...logger import logger
                    logger.error(
                        f"Closure {self.func_ast.name}() takes {len(param_names)} "
                        f"arguments, got {len(args)}",
                        node=call_node, exc_type=TypeError
                    )
                param_bindings = dict(zip(param_names, args))
                
                # Use ClosureAdapter to inline the closure with captured globals
                adapter = ClosureAdapter(visitor, param_bindings, func_globals=self.func_globals)
                return adapter.execute_closure(self.func_ast)
        
        # Create wrapper instance with captured globals
        wrapper = ClosureWrapper(node, self, closure_globals)
        
        # Register as a variable in current scope
        var_info = VariableInfo(
            name=func_name,
            value_ref=wrap_value(
                wrapper,
                kind='python',
                type_hint=wrapper,
            ),
            alloca=None,
            source='closure',
            is_mutable=False,
        )
        
        self.ctx.var_registry.declare(var_info, allow_shadow=True)
    
    def _calculate_struct_size(self, struct_type):
        """Calculate the size of a struct type in bytes"""
        if isinstance(struct_type, (ir.LiteralStructType, ir.IdentifiedStructType)):
            total_size = 0
            for element_type in struct_type.elements:
                if isinstance(element_type, ir.PointerType):
                    total_size += 8  # Pointer size on 64-bit systems
                elif isinstance(element_type, ir.IntType):
                    total_size += element_type.width // 8  # Convert bits to bytes
                elif isinstance(element_type, ir.FloatType):
                    total_size += 4  # 32-bit float
                elif isinstance(element_type, ir.DoubleType):
                    total_size += 8  # 64-bit double
                else:
                    total_size += 8  # Default size for unknown types
            return total_size
        return 16  # Default fallback size
    

