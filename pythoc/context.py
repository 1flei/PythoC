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
    VariableInfo
)
from .scope_manager import ScopeManager

# EnhancedValueRef has been merged into ValueRef in ir_value.py
# Import it here for backward compatibility
from .valueref import ValueRef as EnhancedValueRef


@dataclass
class FunctionBindingState:
    """Long-lived wrapper binding state.

    This tracks which compiled implementation a wrapper points to, along with
    the effect/materialization data needed to continue specialization later.
    """
    source_file: Optional[str] = None
    original_name: Optional[str] = None
    actual_func_name: Optional[str] = None
    mangled_name: Optional[str] = None
    group_key: Optional[tuple] = None
    compile_suffix: Optional[str] = None
    effect_suffix: Optional[str] = None
    compiler: Optional[Any] = None
    so_file: Optional[str] = None
    is_template: bool = False
    captured_effect_context: Optional[Any] = None
    captured_symbols: Optional[Any] = None
    effect_specialized_cache: Dict[str, Any] = field(default_factory=dict)
    template_compile_callback: Optional[Any] = None
    compilation_globals: Dict[str, Any] = field(default_factory=dict)
    wrapper: Optional[Any] = None


@dataclass
class ActiveCompileFrame:
    """Ephemeral state for one compile_function_from_ast() invocation."""
    current_function: Optional[ir.Function] = None
    varargs_info: Optional[dict] = None
    all_inlined_stmts: list = field(default_factory=list)


# Backward compatibility alias for older imports.
FunctionCompileState = FunctionBindingState


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
        
        # Unified scope manager for variables, defers, and linear types
        self.scope_manager = ScopeManager()

        # Type inference integration
        self.type_inference_ctx: Optional[Any] = None

        # Struct types registry
        self.struct_types: Dict[str, ir.Type] = {}
        
        # Source globals (for type resolution)
        self.source_globals: Dict[str, Any] = {}
        
        # User globals (from the module where @compile is used)
        # This allows accessing constants, type aliases, imported names, etc.
        self.user_globals: Dict[str, Any] = user_globals or {}

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
    
    def __repr__(self) -> str:
        if self._module is not None:
            return f"CompilationContext(module={self._module.name})"
        elif self._backend is not None:
            return f"CompilationContext(backend={type(self._backend).__name__})"
        return "CompilationContext(constexpr)"
