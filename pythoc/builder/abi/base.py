"""
Base classes for ABI handling.

Defines the interface for ABI coercion of struct types.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List, Tuple, Any

from llvmlite import ir


class PassingKind(Enum):
    """How a type should be passed/returned."""
    DIRECT = auto()      # Pass directly (no coercion needed)
    COERCE = auto()      # Coerce to different type
    INDIRECT = auto()    # Pass via pointer (sret for returns)
    EXPAND = auto()      # Expand struct fields as separate args


@dataclass
class CoercedType:
    """Result of ABI coercion for a type.
    
    Attributes:
        kind: How the type should be passed
        coerced_type: The LLVM type to use (for COERCE kind)
        original_type: The original LLVM type
        is_return: Whether this is for a return value
    """
    kind: PassingKind
    coerced_type: Optional[ir.Type] = None
    original_type: Optional[ir.Type] = None
    is_return: bool = False
    
    @property
    def needs_coercion(self) -> bool:
        """Check if coercion is needed."""
        return self.kind in (PassingKind.COERCE, PassingKind.INDIRECT)
    
    @property
    def is_indirect(self) -> bool:
        """Check if indirect passing (sret) is needed."""
        return self.kind == PassingKind.INDIRECT


class ABIInfo(ABC):
    """Abstract base class for ABI information.
    
    Subclasses implement target-specific ABI rules for:
    - Struct return type coercion
    - Struct argument passing
    - Alignment requirements
    """
    
    @abstractmethod
    def classify_return_type(self, llvm_type: ir.Type) -> CoercedType:
        """Classify how a return type should be handled.
        
        Args:
            llvm_type: The LLVM type being returned
            
        Returns:
            CoercedType describing how to handle the return
        """
        pass
    
    @abstractmethod
    def classify_argument_type(self, llvm_type: ir.Type) -> CoercedType:
        """Classify how an argument type should be passed.
        
        Args:
            llvm_type: The LLVM type being passed
            
        Returns:
            CoercedType describing how to pass the argument
        """
        pass
    
    def get_type_size(self, llvm_type: ir.Type) -> int:
        """Get the target allocation size of an LLVM ABI type."""
        if isinstance(llvm_type, ir.VoidType):
            return 0
        from ...target_layout import get_target_layout
        return get_target_layout().primitive(llvm_type).allocation_size
    
    def get_type_alignment(self, llvm_type: ir.Type) -> int:
        """Get the target ABI alignment of an LLVM ABI type."""
        if isinstance(llvm_type, ir.VoidType):
            return 1
        from ...target_layout import get_target_layout
        return get_target_layout().primitive(llvm_type).alignment
    
    def is_struct_type(self, llvm_type: ir.Type) -> bool:
        """Check if type is a struct type."""
        return isinstance(llvm_type, (ir.LiteralStructType, ir.IdentifiedStructType))
    
    def is_aggregate_type(self, llvm_type: ir.Type) -> bool:
        """Check if type is an aggregate type (struct or array).
        
        Aggregate types may need ABI coercion for passing/returning.
        """
        return isinstance(llvm_type, (ir.LiteralStructType, ir.IdentifiedStructType, ir.ArrayType))
    
    def uses_byval_for_indirect_args(self) -> bool:
        """Check if this ABI uses byval attribute for indirect aggregate arguments.
        
        x86-64: Uses byval attribute on pointer parameters
        ARM64: Does NOT use byval - just passes pointer directly
        
        Returns:
            True if byval attribute should be added to indirect arg pointers
        """
        return True  # Default for x86-64
