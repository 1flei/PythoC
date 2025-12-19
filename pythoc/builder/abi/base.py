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
        """Get size of LLVM type in bytes.
        
        Args:
            llvm_type: LLVM type
            
        Returns:
            Size in bytes
        """
        if isinstance(llvm_type, ir.IntType):
            return (llvm_type.width + 7) // 8
        elif isinstance(llvm_type, ir.FloatType):
            return 4
        elif isinstance(llvm_type, ir.DoubleType):
            return 8
        elif isinstance(llvm_type, ir.PointerType):
            return 8  # Assume 64-bit pointers
        elif isinstance(llvm_type, ir.ArrayType):
            return llvm_type.count * self.get_type_size(llvm_type.element)
        elif isinstance(llvm_type, (ir.LiteralStructType, ir.IdentifiedStructType)):
            # Sum of field sizes with alignment
            total = 0
            for elem in llvm_type.elements:
                elem_size = self.get_type_size(elem)
                elem_align = self.get_type_alignment(elem)
                # Align current offset
                if total % elem_align != 0:
                    total += elem_align - (total % elem_align)
                total += elem_size
            # Final struct alignment
            struct_align = self.get_type_alignment(llvm_type)
            if total % struct_align != 0:
                total += struct_align - (total % struct_align)
            return total
        elif isinstance(llvm_type, ir.VoidType):
            return 0
        else:
            # Default to 8 bytes
            return 8
    
    def get_type_alignment(self, llvm_type: ir.Type) -> int:
        """Get alignment of LLVM type in bytes.
        
        Args:
            llvm_type: LLVM type
            
        Returns:
            Alignment in bytes
        """
        if isinstance(llvm_type, ir.IntType):
            return min((llvm_type.width + 7) // 8, 8)
        elif isinstance(llvm_type, ir.FloatType):
            return 4
        elif isinstance(llvm_type, ir.DoubleType):
            return 8
        elif isinstance(llvm_type, ir.PointerType):
            return 8
        elif isinstance(llvm_type, ir.ArrayType):
            return self.get_type_alignment(llvm_type.element)
        elif isinstance(llvm_type, (ir.LiteralStructType, ir.IdentifiedStructType)):
            # Struct alignment is max of field alignments
            max_align = 1
            for elem in llvm_type.elements:
                max_align = max(max_align, self.get_type_alignment(elem))
            return max_align
        else:
            return 8
    
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
