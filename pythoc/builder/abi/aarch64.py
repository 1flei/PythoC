"""
AArch64 (ARM64) AAPCS ABI implementation.

This implements the struct passing/returning conventions for ARM64.
Based on the AAPCS64 specification.

Key rules for struct returns:
- <= 16 bytes: pass in registers (coerce to i128 or {i64, i64})
- > 16 bytes: use sret (indirect return via pointer)
- HFA (Homogeneous Floating-point Aggregate): up to 4 same-type floats in SIMD regs
"""

from typing import Optional, Tuple
from llvmlite import ir

from .base import ABIInfo, CoercedType, PassingKind


class AArch64ABI(ABIInfo):
    """AArch64 AAPCS ABI implementation."""
    
    def uses_byval_for_indirect_args(self) -> bool:
        """ARM64 does NOT use byval attribute for indirect aggregate arguments.
        
        Unlike x86-64, ARM64 passes large structs by simply passing a pointer
        to a caller-allocated copy. The function signature uses a plain pointer
        parameter without byval attribute.
        
        Returns:
            False - ARM64 does not use byval
        """
        return False
    
    def classify_return_type(self, llvm_type: ir.Type) -> CoercedType:
        """Classify return type for AArch64 ABI.
        
        Args:
            llvm_type: The LLVM type being returned
            
        Returns:
            CoercedType with coercion info
        """
        # Non-aggregate types don't need coercion
        if not self.is_aggregate_type(llvm_type):
            return CoercedType(
                kind=PassingKind.DIRECT,
                original_type=llvm_type,
                is_return=True
            )
        
        size = self.get_type_size(llvm_type)
        
        # Empty struct returns void
        if size == 0:
            return CoercedType(
                kind=PassingKind.DIRECT,
                coerced_type=ir.VoidType(),
                original_type=llvm_type,
                is_return=True
            )
        
        # > 16 bytes: use sret (indirect)
        if size > 16:
            return CoercedType(
                kind=PassingKind.INDIRECT,
                original_type=llvm_type,
                is_return=True
            )
        
        # Check for HFA (Homogeneous Floating-point Aggregate)
        hfa_type, hfa_count = self._check_hfa(llvm_type)
        if hfa_type is not None and hfa_count <= 4:
            # Return as array of floats/doubles
            if hfa_count == 1:
                coerced = hfa_type
            else:
                coerced = ir.ArrayType(hfa_type, hfa_count)
            return CoercedType(
                kind=PassingKind.COERCE,
                coerced_type=coerced,
                original_type=llvm_type,
                is_return=True
            )
        
        # <= 16 bytes integer-like struct: coerce to integer(s)
        if size <= 8:
            coerced = ir.IntType(size * 8)
        else:
            # Two registers
            coerced = ir.LiteralStructType([ir.IntType(64), ir.IntType((size - 8) * 8)])
        
        return CoercedType(
            kind=PassingKind.COERCE,
            coerced_type=coerced,
            original_type=llvm_type,
            is_return=True
        )
    
    def classify_argument_type(self, llvm_type: ir.Type) -> CoercedType:
        """Classify argument type for AArch64 ABI.
        
        AAPCS64 rules for composite type arguments:
        - <= 16 bytes: pass in registers (coerce to integers)
        - > 16 bytes: caller copies to memory, callee receives pointer
        - HFA (up to 4 floats/doubles): pass in SIMD registers
        """
        # Non-aggregate types don't need coercion
        if not self.is_aggregate_type(llvm_type):
            return CoercedType(
                kind=PassingKind.DIRECT,
                original_type=llvm_type,
                is_return=False
            )
        
        size = self.get_type_size(llvm_type)
        
        # Empty struct
        if size == 0:
            return CoercedType(
                kind=PassingKind.DIRECT,
                coerced_type=ir.VoidType(),
                original_type=llvm_type,
                is_return=False
            )
        
        # > 16 bytes: pass by reference (caller copies, callee gets pointer)
        # This is different from sret - the callee receives a pointer to a copy
        if size > 16:
            return CoercedType(
                kind=PassingKind.INDIRECT,
                original_type=llvm_type,
                is_return=False
            )
        
        # Check for HFA (Homogeneous Floating-point Aggregate)
        hfa_type, hfa_count = self._check_hfa(llvm_type)
        if hfa_type is not None and hfa_count <= 4:
            # Pass as array of floats/doubles in SIMD regs
            if hfa_count == 1:
                coerced = hfa_type
            else:
                coerced = ir.ArrayType(hfa_type, hfa_count)
            return CoercedType(
                kind=PassingKind.COERCE,
                coerced_type=coerced,
                original_type=llvm_type,
                is_return=False
            )
        
        # <= 16 bytes integer-like: coerce to integer(s)
        if size <= 8:
            coerced = ir.IntType(size * 8)
        else:
            # Two registers
            coerced = ir.LiteralStructType([ir.IntType(64), ir.IntType((size - 8) * 8)])
        
        return CoercedType(
            kind=PassingKind.COERCE,
            coerced_type=coerced,
            original_type=llvm_type,
            is_return=False
        )
    
    def _check_hfa(self, llvm_type: ir.Type) -> Tuple[Optional[ir.Type], int]:
        """Check if type is a Homogeneous Floating-point Aggregate.
        
        Args:
            llvm_type: Type to check
            
        Returns:
            Tuple of (base_float_type, count) or (None, 0) if not HFA
        """
        if not self.is_struct_type(llvm_type):
            return None, 0
        
        elements = llvm_type.elements if hasattr(llvm_type, 'elements') else []
        if not elements:
            return None, 0
        
        # Check if all elements are the same float type
        base_type = None
        count = 0
        
        for elem in elements:
            if isinstance(elem, ir.FloatType):
                if base_type is None:
                    base_type = elem
                elif not isinstance(base_type, ir.FloatType):
                    return None, 0
                count += 1
            elif isinstance(elem, ir.DoubleType):
                if base_type is None:
                    base_type = elem
                elif not isinstance(base_type, ir.DoubleType):
                    return None, 0
                count += 1
            elif isinstance(elem, (ir.LiteralStructType, ir.IdentifiedStructType)):
                # Recursively check nested struct
                nested_type, nested_count = self._check_hfa(elem)
                if nested_type is None:
                    return None, 0
                if base_type is None:
                    base_type = nested_type
                elif type(base_type) != type(nested_type):
                    return None, 0
                count += nested_count
            else:
                # Non-float element breaks HFA
                return None, 0
        
        return base_type, count
