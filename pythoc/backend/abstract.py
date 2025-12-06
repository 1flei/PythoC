"""
Abstract Backend interface for code generation.

This defines the interface that all code generation backends must implement.
The Backend abstracts away:
1. IR builder operations
2. Module/context management
3. Variable storage allocation
4. Type resolution context

This separation allows the visitor to be backend-agnostic, enabling:
- LLVM IR generation for actual compilation
- Constexpr evaluation for type resolution and constant folding
- Future backends (C, MLIR, etc.)
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict

from ..builder.abstract import AbstractBuilder


class AbstractBackend(ABC):
    """Abstract interface for code generation backends.
    
    A Backend encapsulates all backend-specific functionality:
    - Builder for IR generation
    - Module context for type resolution
    - Variable allocation strategy
    - Function/global management
    
    The visitor uses this interface without knowing the concrete backend.
    """
    
    @abstractmethod
    def get_builder(self) -> AbstractBuilder:
        """Get the IR builder for this backend.
        
        Returns:
            AbstractBuilder instance (LLVMBuilder, NullBuilder, etc.)
        """
        pass
    
    @abstractmethod
    def get_module_context(self) -> Optional[Any]:
        """Get module context for type resolution.
        
        For LLVM: returns ir.Context for struct type resolution
        For Constexpr: returns None (no LLVM types needed)
        
        Returns:
            Backend-specific module context, or None
        """
        pass
    
    @abstractmethod
    def get_module(self) -> Optional[Any]:
        """Get the module object.
        
        For LLVM: returns ir.Module
        For Constexpr: returns None
        
        Returns:
            Backend-specific module, or None
        """
        pass
    
    @abstractmethod
    def allocate_variable(self, name: str, llvm_type: Any, pc_type: Any = None) -> Optional[Any]:
        """Allocate storage for a variable.
        
        For LLVM: creates alloca instruction
        For Constexpr: returns None (variables stored as Python values)
        
        Args:
            name: Variable name
            llvm_type: LLVM type (or None for constexpr)
            pc_type: PC type hint
            
        Returns:
            Allocated storage reference, or None for constexpr
        """
        pass
    
    @abstractmethod
    def get_global(self, name: str) -> Optional[Any]:
        """Get a global value by name.
        
        For LLVM: looks up in module globals
        For Constexpr: returns None
        
        Args:
            name: Global name
            
        Returns:
            Global value, or None if not found
        """
        pass
    
    @abstractmethod
    def declare_function(self, name: str, func_type: Any) -> Any:
        """Declare a function.
        
        For LLVM: creates ir.Function
        For Constexpr: raises error (not supported)
        
        Args:
            name: Function name
            func_type: Function type
            
        Returns:
            Function object
        """
        pass
    
    @abstractmethod
    def create_block(self, name: str = "") -> Any:
        """Create a new basic block.
        
        For LLVM: creates ir.Block
        For Constexpr: raises error (not supported)
        
        Args:
            name: Block name
            
        Returns:
            Block object
        """
        pass
    
    @abstractmethod
    def is_constexpr(self) -> bool:
        """Check if this is a constexpr (compile-time only) backend.
        
        Returns:
            True if this backend is for compile-time evaluation only
        """
        pass
    
    @property
    def builder(self) -> AbstractBuilder:
        """Convenience property to access builder."""
        return self.get_builder()
