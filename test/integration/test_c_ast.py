"""
Test c_ast module: linear ownership, refined types, and type construction.

Tests cover:
1. Linear allocation: alloc returns (proof, ptr), must consume proof
2. Refined types: CTypeRef guarantees non-null
3. Type construction: make_* functions with ownership transfer
4. Free functions: recursive free with proof consumption
5. Match-case dispatch on CType variants
"""

import unittest
from pythoc import compile, i32, i8, ptr, sizeof, struct, consume, assume, nullptr, bool
from pythoc.libc.stdlib import malloc, free
from pythoc.libc.stdio import printf

# Import from c_ast module
from pythoc.bindings.c_ast import (
    # Types
    Span, CType, QualType, PtrType, ArrayType, FuncType, 
    StructType, EnumType, EnumValue, FieldInfo, ParamInfo,
    Decl, DeclKind,
    # Refined types
    CTypeRef, QualTypeRef, ctype_nonnull,
    # Proof types
    CTypeProof, QualTypeProof,
    # Allocation
    alloc, ctype_alloc, qualtype_alloc, ptrtype_alloc,
    paraminfo_alloc, fieldinfo_alloc, enumvalue_alloc,
    # Type constructors
    prim, make_qualtype, make_ptr_type, make_array_type,
    make_func_type, make_struct_type, make_typedef_type,
    make_union_type, make_enum_type,
    # Free functions
    ctype_free, qualtype_free,
    # Span helpers
    span_empty, span_is_empty,
    # Constants
    QUAL_NONE, QUAL_CONST,
)


# =============================================================================
# Test 1: Linear allocation - proof must be consumed
# =============================================================================

@compile
def test_linear_alloc_and_free() -> i32:
    """Test that linear alloc returns proof and ptr, free consumes proof."""
    # Allocate with linear tracking
    prf, ty = ctype_alloc()
    
    # Use the pointer
    ty[0] = CType(CType.Int)
    
    # Free must consume proof
    ctype_free(prf, ty)
    
    return 0


@compile
def test_linear_qualtype() -> i32:
    """Test QualType linear allocation."""
    qt_prf, qt = qualtype_alloc()
    ty_prf, ty = ctype_alloc()
    
    ty[0] = CType(CType.Char)
    qt.type = ty
    qt.quals = QUAL_CONST
    
    # Free QualType (which owns CType)
    # We need to consume both proofs since we're manually managing
    consume(ty_prf)  # CType ownership transferred to QualType
    qualtype_free(qt_prf, qt)
    
    return 0


# =============================================================================
# Test 2: Primitive type constructors via prim API
# =============================================================================

@compile
def test_prim_api() -> i32:
    """Test primitive type construction via prim API."""
    # Create int type
    int_prf, int_ty = prim.int()
    
    # Verify tag
    result: i32 = 0
    match int_ty[0]:
        case (CType.Int):
            result = 1
        case _:
            result = 0
    
    ctype_free(int_prf, int_ty)
    return result


@compile
def test_multiple_prims() -> i32:
    """Test creating multiple primitive types."""
    void_prf, void_ty = prim.void()
    char_prf, char_ty = prim.char()
    long_prf, long_ty = prim.long()
    
    count: i32 = 0
    
    match void_ty[0]:
        case (CType.Void):
            count = count + 1
        case _:
            pass
    
    match char_ty[0]:
        case (CType.Char):
            count = count + 1
        case _:
            pass
    
    match long_ty[0]:
        case (CType.Long):
            count = count + 1
        case _:
            pass
    
    ctype_free(void_prf, void_ty)
    ctype_free(char_prf, char_ty)
    ctype_free(long_prf, long_ty)
    
    return count  # Should be 3


# =============================================================================
# Test 3: Compound type construction with ownership transfer
# =============================================================================

@compile
def test_make_qualtype_ownership() -> i32:
    """Test make_qualtype transfers CType ownership."""
    # Create a CType
    ty_prf, ty = prim.int()
    
    # make_qualtype takes ownership of ty (consumes ty_prf)
    qt_prf, qt = make_qualtype(ty_prf, ty, QUAL_CONST)
    
    # Verify
    result: i32 = 0
    if qt.quals == QUAL_CONST:
        match qt.type[0]:
            case (CType.Int):
                result = 1
            case _:
                result = 0
    
    # Free QualType (owns the CType now)
    qualtype_free(qt_prf, qt)
    
    return result


@compile
def test_make_ptr_type() -> i32:
    """Test pointer type construction."""
    # Create pointee: const int
    int_prf, int_ty = prim.int()
    qt_prf, qt = make_qualtype(int_prf, int_ty, QUAL_CONST)
    
    # Create pointer to const int
    ptr_prf, ptr_ty = make_ptr_type(qt_prf, qt, QUAL_NONE)
    
    # Verify it's a pointer type
    result: i32 = 0
    match ptr_ty[0]:
        case (CType.Ptr, pt):
            # Check pointee is const int
            if pt.pointee.quals == QUAL_CONST:
                match pt.pointee.type[0]:
                    case (CType.Int):
                        result = 1
                    case _:
                        result = 0
        case _:
            result = 0
    
    ctype_free(ptr_prf, ptr_ty)
    return result


@compile
def test_make_array_type() -> i32:
    """Test array type construction."""
    # Create element type: int
    int_prf, int_ty = prim.int()
    qt_prf, qt = make_qualtype(int_prf, int_ty, QUAL_NONE)
    
    # Create array[10] of int
    arr_prf, arr_ty = make_array_type(qt_prf, qt, 10)
    
    # Verify
    result: i32 = 0
    match arr_ty[0]:
        case (CType.Array, at):
            if at.size == 10:
                match at.elem.type[0]:
                    case (CType.Int):
                        result = 1
                    case _:
                        result = 0
        case _:
            result = 0
    
    ctype_free(arr_prf, arr_ty)
    return result


# =============================================================================
# Test 4: Span functions
# =============================================================================

@compile
def test_span_empty() -> i32:
    """Test span_empty and span_is_empty."""
    s: Span = span_empty()
    
    if span_is_empty(s):
        return 1
    return 0


# =============================================================================
# Test 5: Match-case dispatch on CType
# =============================================================================

@compile
def test_ctype_match() -> i32:
    """Test match-case dispatch on different CType variants."""
    int_prf, int_ty = prim.int()
    void_prf, void_ty = prim.void()
    
    # Create a pointer type
    qt_prf, qt = make_qualtype(void_prf, void_ty, QUAL_NONE)
    ptr_prf, ptr_ty = make_ptr_type(qt_prf, qt, QUAL_NONE)
    
    count: i32 = 0
    
    # Match int
    match int_ty[0]:
        case (CType.Int):
            count = count + 1
        case _:
            pass
    
    # Match pointer
    match ptr_ty[0]:
        case (CType.Ptr, pt):
            count = count + 1
        case _:
            pass
    
    ctype_free(int_prf, int_ty)
    ctype_free(ptr_prf, ptr_ty)
    
    return count  # Should be 2


# =============================================================================
# Test 6: Typedef type
# =============================================================================

@compile
def test_typedef_type() -> i32:
    """Test typedef type construction."""
    name: Span = span_empty()
    
    td_prf, td_ty = make_typedef_type(name)
    
    result: i32 = 0
    match td_ty[0]:
        case (CType.Typedef, span):
            if span_is_empty(span):
                result = 1
        case _:
            result = 0
    
    ctype_free(td_prf, td_ty)
    return result


# =============================================================================
# Test 7: FuncType construction
# =============================================================================

@compile
def test_make_func_type() -> i32:
    """Test function type construction with parameters."""
    # Create return type: int
    ret_prf, ret_ty = prim.int()
    ret_qt_prf, ret_qt = make_qualtype(ret_prf, ret_ty, QUAL_NONE)
    
    # Create parameter array with 2 params
    params: ptr[ParamInfo] = paraminfo_alloc(2)
    
    # Param 0: int x
    p0_prf, p0_ty = prim.int()
    p0_qt_prf, p0_qt = make_qualtype(p0_prf, p0_ty, QUAL_NONE)
    params[0].name = span_empty()
    params[0].type = p0_qt
    consume(p0_qt_prf)  # Transfer ownership to params array
    
    # Param 1: const char* s (simplified as char for test)
    p1_prf, p1_ty = prim.char()
    p1_qt_prf, p1_qt = make_qualtype(p1_prf, p1_ty, QUAL_CONST)
    params[1].name = span_empty()
    params[1].type = p1_qt
    consume(p1_qt_prf)  # Transfer ownership to params array
    
    # Create function type: int(int, const char)
    func_prf, func_ty = make_func_type(ret_qt_prf, ret_qt, params, 2, 0)
    
    # Verify
    result: i32 = 0
    match func_ty[0]:
        case (CType.Func, ft):
            if ft.param_count == 2:
                if ft.is_variadic == 0:
                    # Check return type is int
                    match ft.ret.type[0]:
                        case (CType.Int):
                            result = 1
                        case _:
                            result = 0
        case _:
            result = 0
    
    ctype_free(func_prf, func_ty)
    return result


@compile
def test_make_func_type_variadic() -> i32:
    """Test variadic function type (e.g., printf-like)."""
    # Create return type: int
    ret_prf, ret_ty = prim.int()
    ret_qt_prf, ret_qt = make_qualtype(ret_prf, ret_ty, QUAL_NONE)
    
    # Create function type with no fixed params, variadic
    func_prf, func_ty = make_func_type(ret_qt_prf, ret_qt, nullptr, 0, 1)
    
    result: i32 = 0
    match func_ty[0]:
        case (CType.Func, ft):
            if ft.is_variadic == 1:
                if ft.param_count == 0:
                    result = 1
        case _:
            result = 0
    
    ctype_free(func_prf, func_ty)
    return result


# =============================================================================
# Test 8: StructType construction
# =============================================================================

@compile
def test_make_struct_type() -> i32:
    """Test struct type construction with fields."""
    # Create fields array with 2 fields
    fields: ptr[FieldInfo] = fieldinfo_alloc(2)
    
    # Field 0: int x
    f0_prf, f0_ty = prim.int()
    f0_qt_prf, f0_qt = make_qualtype(f0_prf, f0_ty, QUAL_NONE)
    fields[0].name = span_empty()
    fields[0].type = f0_qt
    fields[0].bit_width = -1  # Not a bitfield
    consume(f0_qt_prf)
    
    # Field 1: double y
    f1_prf, f1_ty = prim.double()
    f1_qt_prf, f1_qt = make_qualtype(f1_prf, f1_ty, QUAL_NONE)
    fields[1].name = span_empty()
    fields[1].type = f1_qt
    fields[1].bit_width = -1
    consume(f1_qt_prf)
    
    # Create struct type
    struct_prf, struct_ty = make_struct_type(span_empty(), fields, 2, 1)
    
    # Verify
    result: i32 = 0
    match struct_ty[0]:
        case (CType.Struct, st):
            if st.field_count == 2:
                if st.is_complete == 1:
                    # Check first field is int
                    match st.fields[0].type.type[0]:
                        case (CType.Int):
                            result = 1
                        case _:
                            result = 0
        case _:
            result = 0
    
    ctype_free(struct_prf, struct_ty)
    return result


@compile
def test_make_struct_incomplete() -> i32:
    """Test incomplete (forward-declared) struct."""
    # Create incomplete struct (no fields)
    struct_prf, struct_ty = make_struct_type(span_empty(), nullptr, 0, 0)
    
    result: i32 = 0
    match struct_ty[0]:
        case (CType.Struct, st):
            if st.is_complete == 0:
                if st.field_count == 0:
                    result = 1
        case _:
            result = 0
    
    ctype_free(struct_prf, struct_ty)
    return result


# =============================================================================
# Test 9: EnumType construction
# =============================================================================

@compile
def test_make_enum_type() -> i32:
    """Test enum type construction with values."""
    # Create enum values array with 3 values
    values: ptr[EnumValue] = enumvalue_alloc(3)
    
    # Value 0: RED = 0 (implicit)
    values[0].name = span_empty()
    values[0].value = 0
    values[0].has_explicit_value = 0
    
    # Value 1: GREEN = 1 (implicit)
    values[1].name = span_empty()
    values[1].value = 1
    values[1].has_explicit_value = 0
    
    # Value 2: BLUE = 100 (explicit)
    values[2].name = span_empty()
    values[2].value = 100
    values[2].has_explicit_value = 1
    
    # Create enum type
    enum_prf, enum_ty = make_enum_type(span_empty(), values, 3, 1)
    
    # Verify
    result: i32 = 0
    match enum_ty[0]:
        case (CType.Enum, et):
            if et.value_count == 3:
                if et.is_complete == 1:
                    # Check third value is 100 with explicit flag
                    if et.values[2].value == 100:
                        if et.values[2].has_explicit_value == 1:
                            result = 1
        case _:
            result = 0
    
    ctype_free(enum_prf, enum_ty)
    return result


# =============================================================================
# Test 10: Union type (uses same StructType payload)
# =============================================================================

@compile
def test_make_union_type() -> i32:
    """Test union type construction."""
    # Create fields array with 2 fields
    fields: ptr[FieldInfo] = fieldinfo_alloc(2)
    
    # Field 0: int i
    f0_prf, f0_ty = prim.int()
    f0_qt_prf, f0_qt = make_qualtype(f0_prf, f0_ty, QUAL_NONE)
    fields[0].name = span_empty()
    fields[0].type = f0_qt
    fields[0].bit_width = -1
    consume(f0_qt_prf)
    
    # Field 1: float f
    f1_prf, f1_ty = prim.float()
    f1_qt_prf, f1_qt = make_qualtype(f1_prf, f1_ty, QUAL_NONE)
    fields[1].name = span_empty()
    fields[1].type = f1_qt
    fields[1].bit_width = -1
    consume(f1_qt_prf)
    
    # Create union type
    union_prf, union_ty = make_union_type(span_empty(), fields, 2, 1)
    
    # Verify it's a Union, not Struct
    result: i32 = 0
    match union_ty[0]:
        case (CType.Union, st):
            if st.field_count == 2:
                result = 1
        case (CType.Struct, st):
            result = 0  # Wrong - should be Union
        case _:
            result = 0
    
    ctype_free(union_prf, union_ty)
    return result


# =============================================================================
# Test runner
# =============================================================================

class TestCAst(unittest.TestCase):
    def test_linear_alloc_and_free(self):
        result = test_linear_alloc_and_free()
        self.assertEqual(result, 0)
    
    def test_linear_qualtype(self):
        result = test_linear_qualtype()
        self.assertEqual(result, 0)
    
    def test_prim_api(self):
        result = test_prim_api()
        self.assertEqual(result, 1)
    
    def test_multiple_prims(self):
        result = test_multiple_prims()
        self.assertEqual(result, 3)
    
    def test_make_qualtype_ownership(self):
        result = test_make_qualtype_ownership()
        self.assertEqual(result, 1)
    
    def test_make_ptr_type(self):
        result = test_make_ptr_type()
        self.assertEqual(result, 1)
    
    def test_make_array_type(self):
        result = test_make_array_type()
        self.assertEqual(result, 1)
    
    def test_span_empty(self):
        result = test_span_empty()
        self.assertEqual(result, 1)
    
    def test_ctype_match(self):
        result = test_ctype_match()
        self.assertEqual(result, 2)
    
    def test_typedef_type(self):
        result = test_typedef_type()
        self.assertEqual(result, 1)
    
    def test_make_func_type(self):
        result = test_make_func_type()
        self.assertEqual(result, 1)
    
    def test_make_func_type_variadic(self):
        result = test_make_func_type_variadic()
        self.assertEqual(result, 1)
    
    def test_make_struct_type(self):
        result = test_make_struct_type()
        self.assertEqual(result, 1)
    
    def test_make_struct_incomplete(self):
        result = test_make_struct_incomplete()
        self.assertEqual(result, 1)
    
    def test_make_enum_type(self):
        result = test_make_enum_type()
        self.assertEqual(result, 1)
    
    def test_make_union_type(self):
        result = test_make_union_type()
        self.assertEqual(result, 1)


if __name__ == '__main__':
    print("=== C AST Module Tests ===\n")
    unittest.main(verbosity=2)
