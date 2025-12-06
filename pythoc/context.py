"""
Context Management System

This module provides:
1. Variable registry with scope management (now from registry)
2. Enhanced ValueRef with type information (now from ir_value)
3. Integration with type inference system

Type classes (i32, f64, etc.) are now used directly instead of PCType instances.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List, Union, TYPE_CHECKING
from llvmlite import ir
import ast

if TYPE_CHECKING:
    from .builtin_entities import BuiltinType
    from .backend import AbstractBackend

# Backward compatibility: import type registry from builtin_entities
from .builtin_entities import TYPE_REGISTRY as PC_TYPE_MAP

# Backward compatibility: PCType is now just a type alias for Any (BuiltinType class)
# Old code using PCType will now work with BuiltinType classes directly
# PCType = Any  # Type alias for backward compatibility

# Import registry components
from .registry import (
    VariableInfo,
    VariableRegistry
)

# EnhancedValueRef has been merged into ValueRef in ir_value.py
# Import it here for backward compatibility
from .valueref import ValueRef as EnhancedValueRef


class CompilationContext:
    """Global compilation context
    
    This holds all context information for a compilation unit:
    - Variable registry
    - Type information
    - Backend (module and builder access)
    - Compilation options
    
    Supports two initialization modes:
    1. Legacy: module + builder (backward compatible)
    2. New: backend (preferred)
    """
    
    def __init__(self, module: ir.Module = None, builder: ir.IRBuilder = None,
                 user_globals: Dict[str, Any] = None, backend: "AbstractBackend" = None):
        # Support both legacy (module/builder) and new (backend) initialization
        if backend is not None:
            self._backend = backend
            self._module = backend.get_module()
            self._builder = backend.get_llvm_builder() if hasattr(backend, 'get_llvm_builder') else None
        else:
            self._backend = None
            self._module = module
            self._builder = builder
        
        # Variable management
        self.var_registry = VariableRegistry()
        
        # Type inference integration
        self.type_inference_ctx: Optional[Any] = None
        
        # Current function being compiled
        self.current_function: Optional[ir.Function] = None
        
        # Struct types registry
        self.struct_types: Dict[str, ir.Type] = {}
        
        # Source globals (for type resolution)
        self.source_globals: Dict[str, Any] = {}
        
        # User globals (from the module where @compile is used)
        # This allows accessing constants, type aliases, imported names, etc.
        self.user_globals: Dict[str, Any] = user_globals or {}
        
        # Loop context (for break/continue)
        self.loop_stack: List[tuple] = []
        
        # Label counter for unique names
        self.label_counter = 0
    
    @property
    def module(self) -> Optional[ir.Module]:
        """Get LLVM module (backward compatible property)"""
        return self._module
    
    @property
    def builder(self) -> Optional[ir.IRBuilder]:
        """Get LLVM builder (backward compatible property)"""
        return self._builder
    
    @property
    def backend(self) -> Optional["AbstractBackend"]:
        """Get the backend (if initialized with one)"""
        return self._backend
    
    def is_constexpr(self) -> bool:
        """Check if this context is for constexpr evaluation"""
        if self._backend is not None:
            return self._backend.is_constexpr()
        return self._module is None
    
    def get_next_label(self, prefix: str = "label") -> str:
        """Generate unique label name"""
        label = f"{prefix}{self.label_counter}"
        self.label_counter += 1
        return label
    
    def enter_function(self, func: ir.Function):
        """Enter a function scope"""
        self.current_function = func
        self.var_registry.enter_scope()
    
    def exit_function(self):
        """Exit function scope"""
        self.var_registry.exit_scope()
        self.current_function = None
    
    def __repr__(self) -> str:
        if self._module is not None:
            return f"CompilationContext(module={self._module.name})"
        elif self._backend is not None:
            return f"CompilationContext(backend={type(self._backend).__name__})"
        return "CompilationContext(constexpr)"
