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

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Handle function definition in AST traversal
        
        If encountered within a function body (nested), treat as closure.
        Otherwise, treat as top-level function definition.
        """
        # This is a closure - register it as a callable
        self._register_closure(node)
        return None
    
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
