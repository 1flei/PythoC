"""
x86-64 ABI implementation.

This implements the struct passing/returning conventions for x86-64.

System V ABI (Linux/macOS):
- <= 8 bytes: coerce to single integer (i8/i16/i32/i64)
- 9-16 bytes: coerce to {i64, iN} or similar two-element struct
- > 16 bytes: use sret (indirect return via pointer)

Windows x64 ABI:
- <= 8 bytes: coerce to single integer
- > 8 bytes: use sret (indirect return via pointer)

For floating-point structs, SSE registers are used, but for simplicity
we focus on integer-class structs first.
"""

from typing import Optional, List, Tuple
from llvmlite import ir

from .base import ABIInfo, CoercedType, PassingKind


class FieldClass:
    """Classification of an 8-byte region (eightbyte) in x86-64 ABI."""
    NO_CLASS = 0
    INTEGER = 1
    SSE = 2
    SSEUP = 3
    X87 = 4
    X87UP = 5
    COMPLEX_X87 = 6
    MEMORY = 7  # Pass in memory (sret for returns)


class X86_64ABI(ABIInfo):
    """x86-64 ABI implementation.
    
    Supports both System V (Linux/macOS) and Windows x64 calling conventions.
    The key difference is the threshold for indirect passing:
    - System V: > 16 bytes
    - Windows x64: > 8 bytes
    """
    
    # Default to System V ABI (16 bytes threshold)
    # Windows x64 uses 8 bytes threshold
    MAX_REGISTER_SIZE = 16
    
    def __init__(self, max_register_size: int = 16):
        """Initialize x86-64 ABI.
        
        Args:
            max_register_size: Maximum struct size that can be passed in registers.
                              16 for System V (Linux/macOS), 8 for Windows x64.
        """
        super().__init__()
        self.max_register_size = max_register_size
    
    def classify_return_type(self, llvm_type: ir.Type) -> CoercedType:
        """Classify return type for x86-64 ABI.
        
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
        
        # Use sret (indirect) for structs larger than max_register_size
        # System V: > 16 bytes, Windows x64: > 8 bytes
        if size > self.max_register_size:
            return CoercedType(
                kind=PassingKind.INDIRECT,
                original_type=llvm_type,
                is_return=True
            )
        
        # Classify the struct's eightbytes
        lo_class, hi_class = self._classify_struct(llvm_type)
        
        # If either class is MEMORY, use sret
        if lo_class == FieldClass.MEMORY or hi_class == FieldClass.MEMORY:
            return CoercedType(
                kind=PassingKind.INDIRECT,
                original_type=llvm_type,
                is_return=True
            )
        
        # Build coerced type based on classification
        coerced = self._build_coerced_type(llvm_type, size, lo_class, hi_class)
        
        if coerced is None:
            # Fallback: use sret
            return CoercedType(
                kind=PassingKind.INDIRECT,
                original_type=llvm_type,
                is_return=True
            )
        
        return CoercedType(
            kind=PassingKind.COERCE,
            coerced_type=coerced,
            original_type=llvm_type,
            is_return=True
        )
    
    def classify_argument_type(self, llvm_type: ir.Type) -> CoercedType:
        """Classify argument type for x86-64 ABI.
        
        Similar to return type but arguments have slightly different rules.
        """
        # For now, use same logic as return type
        # TODO: Implement argument-specific rules if needed
        result = self.classify_return_type(llvm_type)
        result.is_return = False
        return result
    
    def _classify_aggregate(self, llvm_type: ir.Type) -> Tuple[int, int]:
        """Classify aggregate type (struct or array) into two eightbyte classes.
        
        Args:
            llvm_type: Struct or array type to classify
            
        Returns:
            Tuple of (lo_class, hi_class) for the two eightbytes
        """
        size = self.get_type_size(llvm_type)
        
        # Initialize classes
        lo_class = FieldClass.NO_CLASS
        hi_class = FieldClass.NO_CLASS
        
        # Use sret for structs larger than max_register_size
        if size > self.max_register_size:
            return FieldClass.MEMORY, FieldClass.MEMORY
        
        # Handle array type - expand as repeated elements
        if isinstance(llvm_type, ir.ArrayType):
            elem_type = llvm_type.element
            elem_size = self.get_type_size(elem_type)
            elem_align = self.get_type_alignment(elem_type)
            field_class = self._classify_field(elem_type)
            
            offset = 0
            for _ in range(llvm_type.count):
                # Align offset
                if elem_align > 0 and offset % elem_align != 0:
                    offset += elem_align - (offset % elem_align)
                
                if offset < 8:
                    lo_class = self._merge_classes(lo_class, field_class)
                    if offset + elem_size > 8:
                        hi_class = self._merge_classes(hi_class, field_class)
                else:
                    hi_class = self._merge_classes(hi_class, field_class)
                
                offset += elem_size
            
            return lo_class, hi_class
        
        # Handle struct type
        offset = 0
        elements = llvm_type.elements if hasattr(llvm_type, 'elements') else []
        
        for elem in elements:
            elem_size = self.get_type_size(elem)
            elem_align = self.get_type_alignment(elem)
            
            # Align offset
            if elem_align > 0 and offset % elem_align != 0:
                offset += elem_align - (offset % elem_align)
            
            # Determine which eightbyte this field falls into
            field_class = self._classify_field(elem)
            
            if offset < 8:
                lo_class = self._merge_classes(lo_class, field_class)
                # If field spans both eightbytes
                if offset + elem_size > 8:
                    hi_class = self._merge_classes(hi_class, field_class)
            else:
                hi_class = self._merge_classes(hi_class, field_class)
            
            offset += elem_size
        
        return lo_class, hi_class
    
    # Alias for backward compatibility
    _classify_struct = _classify_aggregate
    
    def _classify_field(self, llvm_type: ir.Type) -> int:
        """Classify a single field type.
        
        Args:
            llvm_type: Field type
            
        Returns:
            Field class (INTEGER, SSE, etc.)
        """
        if isinstance(llvm_type, ir.IntType):
            return FieldClass.INTEGER
        elif isinstance(llvm_type, ir.PointerType):
            return FieldClass.INTEGER
        elif isinstance(llvm_type, ir.FloatType):
            return FieldClass.SSE
        elif isinstance(llvm_type, ir.DoubleType):
            return FieldClass.SSE
        elif isinstance(llvm_type, (ir.LiteralStructType, ir.IdentifiedStructType)):
            # Recursively classify nested struct
            lo, hi = self._classify_struct(llvm_type)
            # Return the more restrictive class
            return self._merge_classes(lo, hi)
        elif isinstance(llvm_type, ir.ArrayType):
            return self._classify_field(llvm_type.element)
        else:
            return FieldClass.INTEGER
    
    def _merge_classes(self, c1: int, c2: int) -> int:
        """Merge two field classes according to ABI rules.
        
        Args:
            c1, c2: Field classes to merge
            
        Returns:
            Merged class
        """
        if c1 == c2:
            return c1
        if c1 == FieldClass.NO_CLASS:
            return c2
        if c2 == FieldClass.NO_CLASS:
            return c1
        if c1 == FieldClass.MEMORY or c2 == FieldClass.MEMORY:
            return FieldClass.MEMORY
        if c1 == FieldClass.INTEGER or c2 == FieldClass.INTEGER:
            return FieldClass.INTEGER
        # Both are SSE-related
        return FieldClass.SSE
    
    def _build_coerced_type(self, llvm_type: ir.Type, size: int,
                           lo_class: int, hi_class: int) -> Optional[ir.Type]:
        """Build the coerced LLVM type based on classification.
        
        Args:
            llvm_type: Original struct type
            size: Size in bytes
            lo_class: Classification of first eightbyte
            hi_class: Classification of second eightbyte
            
        Returns:
            Coerced LLVM type, or None if sret should be used
        """
        # Handle single eightbyte (size <= 8)
        if size <= 8:
            if lo_class == FieldClass.INTEGER:
                # Coerce to integer of appropriate size
                return self._get_integer_type_for_size(size)
            elif lo_class == FieldClass.SSE:
                # For SSE class, use appropriate float type
                return self._get_sse_type_for_struct(llvm_type, size)
            elif lo_class == FieldClass.NO_CLASS:
                return ir.VoidType()
            else:
                return None
        
        # Handle two eightbytes (8 < size <= 16)
        lo_type = self._get_type_for_class_with_struct(lo_class, 8, llvm_type, 0)
        
        # Calculate size of second eightbyte
        hi_size = size - 8
        hi_type = self._get_type_for_class_with_struct(hi_class, hi_size, llvm_type, 8)
        
        if lo_type is None or hi_type is None:
            return None
        
        return ir.LiteralStructType([lo_type, hi_type])
    
    def _get_integer_type_for_size(self, size: int) -> ir.Type:
        """Get integer type for a given size.
        
        Args:
            size: Size in bytes
            
        Returns:
            Appropriate integer type
        """
        if size <= 1:
            return ir.IntType(8)
        elif size <= 2:
            return ir.IntType(16)
        elif size <= 4:
            return ir.IntType(32)
        else:
            return ir.IntType(64)
    
    def _get_sse_type_for_struct(self, llvm_type: ir.Type, size: int) -> ir.Type:
        """Get SSE type for a struct based on its float fields.
        
        Args:
            llvm_type: Original struct type
            size: Size in bytes
            
        Returns:
            Appropriate SSE type (float, double, or vector)
        """
        elements = llvm_type.elements if hasattr(llvm_type, 'elements') else []
        
        # Count float types
        float_count = 0
        double_count = 0
        for elem in elements:
            if isinstance(elem, ir.FloatType):
                float_count += 1
            elif isinstance(elem, ir.DoubleType):
                double_count += 1
        
        # Single float
        if float_count == 1 and double_count == 0 and size == 4:
            return ir.FloatType()
        
        # Two floats -> <2 x float> vector
        if float_count == 2 and double_count == 0 and size == 8:
            return ir.VectorType(ir.FloatType(), 2)
        
        # Single double
        if double_count == 1 and float_count == 0 and size == 8:
            return ir.DoubleType()
        
        # Fallback: use integer coercion for mixed or complex cases
        return self._get_integer_type_for_size(size)
    
    def _get_type_for_class_with_struct(self, field_class: int, size: int,
                                        llvm_type: ir.Type, offset: int) -> Optional[ir.Type]:
        """Get LLVM type for a field class, considering original struct layout.
        
        Args:
            field_class: Field classification
            size: Size in bytes for this eightbyte
            llvm_type: Original struct type
            offset: Byte offset of this eightbyte in the struct
            
        Returns:
            LLVM type, or None if not representable
        """
        if field_class == FieldClass.NO_CLASS:
            return None
        elif field_class == FieldClass.INTEGER:
            return self._get_integer_type_for_size(size)
        elif field_class == FieldClass.SSE:
            # For SSE, try to match the original struct's float layout
            elements = llvm_type.elements if hasattr(llvm_type, 'elements') else []
            
            # Find float elements in this eightbyte
            current_offset = 0
            float_count = 0
            has_double = False
            
            for elem in elements:
                elem_size = self.get_type_size(elem)
                elem_align = self.get_type_alignment(elem)
                
                # Align
                if current_offset % elem_align != 0:
                    current_offset += elem_align - (current_offset % elem_align)
                
                # Check if element is in this eightbyte
                if current_offset >= offset and current_offset < offset + 8:
                    if isinstance(elem, ir.FloatType):
                        float_count += 1
                    elif isinstance(elem, ir.DoubleType):
                        has_double = True
                
                current_offset += elem_size
            
            if has_double:
                return ir.DoubleType()
            elif float_count == 2:
                return ir.VectorType(ir.FloatType(), 2)
            elif float_count == 1:
                return ir.FloatType()
            else:
                return ir.DoubleType()  # Default for SSE
        elif field_class == FieldClass.MEMORY:
            return None
        else:
            return self._get_integer_type_for_size(size)
    
    def _get_type_for_class(self, field_class: int, size: int) -> Optional[ir.Type]:
        """Get LLVM type for a field class and size.
        
        Args:
            field_class: Field classification
            size: Size in bytes
            
        Returns:
            LLVM type, or None if not representable
        """
        if field_class == FieldClass.NO_CLASS:
            return None
        elif field_class == FieldClass.INTEGER:
            return self._get_integer_type_for_size(size)
        elif field_class == FieldClass.SSE:
            if size <= 4:
                return ir.FloatType()
            else:
                return ir.DoubleType()
        elif field_class == FieldClass.MEMORY:
            return None
        else:
            return self._get_integer_type_for_size(size)
