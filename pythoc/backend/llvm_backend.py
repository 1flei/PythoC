"""
LLVM Backend implementation.

This backend uses llvmlite for actual LLVM IR generation.
It wraps ir.Module and ir.IRBuilder to provide the Backend interface.
"""

from typing import Any, Optional

from llvmlite import ir

from .abstract import AbstractBackend
from ..builder import LLVMBuilder
from ..builder.abstract import AbstractBuilder


class LLVMBackend(AbstractBackend):
    """LLVM-based backend for actual code generation.
    
    This backend generates LLVM IR using llvmlite. It's used for
    actual compilation of PC code to native executables.
    
    Attributes:
        module: The LLVM module
        _builder: The wrapped LLVM IR builder
        _current_function: Currently compiling function (if any)
    """
    
    def __init__(self, module: ir.Module, builder: ir.IRBuilder = None):
        """Initialize LLVM backend.
        
        Args:
            module: LLVM module for code generation
            builder: Optional LLVM IR builder (created lazily if not provided)
        """
        self.module = module
        self._llvm_builder = builder
        self._builder = LLVMBuilder(builder) if builder else None
        self._current_function: Optional[ir.Function] = None
    
    def get_builder(self) -> AbstractBuilder:
        """Get the LLVM IR builder wrapper."""
        if self._builder is None:
            raise RuntimeError("LLVMBackend: builder not initialized. "
                             "Call set_builder() or position builder in a block first.")
        return self._builder
    
    def set_builder(self, builder: ir.IRBuilder) -> None:
        """Set or update the LLVM IR builder.
        
        Args:
            builder: LLVM IR builder instance
        """
        self._llvm_builder = builder
        self._builder = LLVMBuilder(builder)
    
    def get_llvm_builder(self) -> Optional[ir.IRBuilder]:
        """Get the raw LLVM IR builder (for compatibility)."""
        return self._llvm_builder
    
    def get_module_context(self) -> ir.Context:
        """Get LLVM module context for type resolution."""
        return self.module.context
    
    def get_module(self) -> ir.Module:
        """Get the LLVM module."""
        return self.module
    
    def allocate_variable(self, name: str, llvm_type: Any, pc_type: Any = None) -> ir.AllocaInstr:
        """Allocate stack storage for a variable.
        
        Args:
            name: Variable name
            llvm_type: LLVM type for the variable
            pc_type: PC type hint (unused here, for interface compatibility)
            
        Returns:
            Alloca instruction
        """
        if self._llvm_builder is None:
            raise RuntimeError("LLVMBackend: cannot allocate variable without builder")
        return self._llvm_builder.alloca(llvm_type, name=name)
    
    def get_global(self, name: str) -> Optional[ir.GlobalValue]:
        """Get a global value by name.
        
        Args:
            name: Global name
            
        Returns:
            Global value, or None if not found
        """
        try:
            return self.module.get_global(name)
        except KeyError:
            return None
    
    def declare_function(self, name: str, func_type: ir.FunctionType) -> ir.Function:
        """Declare a function in the module.
        
        Args:
            name: Function name
            func_type: LLVM function type
            
        Returns:
            LLVM Function object
        """
        return ir.Function(self.module, func_type, name=name)
    
    def create_block(self, name: str = "") -> ir.Block:
        """Create a new basic block in the current function.
        
        Args:
            name: Block name
            
        Returns:
            LLVM Block object
        """
        if self._current_function is None:
            raise RuntimeError("LLVMBackend: cannot create block without current function")
        return self._current_function.append_basic_block(name=name)
    
    def set_current_function(self, func: ir.Function) -> None:
        """Set the current function being compiled.
        
        Args:
            func: LLVM Function object
        """
        self._current_function = func
    
    def get_current_function(self) -> Optional[ir.Function]:
        """Get the current function being compiled."""
        return self._current_function
    
    def is_constexpr(self) -> bool:
        """LLVM backend is not constexpr."""
        return False
    
    def get_identified_type(self, name: str) -> ir.IdentifiedStructType:
        """Get or create an identified struct type.
        
        Args:
            name: Struct type name
            
        Returns:
            Identified struct type
        """
        return self.module.context.get_identified_type(name)
