#!/usr/bin/env python3
"""
Comprehensive tests for struct and union types including complex layouts,
nested structures, alignment, and edge cases.
"""

import unittest
from pythoc import (
    i8, i16, i32, i64,
    u8, u16, u32, u64,
    f32, f64, bool, ptr, array, struct, union, compile
)


# =============================================================================
# Basic Struct Operations
# =============================================================================

@compile
def test_struct_create_tuple() -> i32:
    """Test struct creation from tuple"""
    s: struct[i32, i32] = (10, 20)
    return s[0] + s[1]  # 30


@compile
def test_struct_create_default() -> i32:
    """Test struct default creation"""
    s: struct[i32, i32] = struct[i32, i32]()
    s[0] = 10
    s[1] = 20
    return s[0] + s[1]  # 30


@compile
def test_struct_named_fields() -> i32:
    """Test struct with named fields"""
    s: struct[x: i32, y: i32] = (10, 20)
    return s.x + s.y  # 30


@compile
def test_struct_mixed_access() -> i32:
    """Test struct with mixed index and name access"""
    s: struct[a: i32, b: i32, c: i32] = (10, 20, 30)
    return s.a + s[1] + s.c  # 60


@compile
def test_struct_modify_field() -> i32:
    """Test modifying struct fields"""
    s: struct[i32, i32] = (10, 20)
    s[0] = 100
    s[1] = 200
    return s[0] + s[1]  # 300


@compile
def test_struct_modify_named() -> i32:
    """Test modifying named struct fields"""
    s: struct[x: i32, y: i32] = (10, 20)
    s.x = 100
    s.y = 200
    return s.x + s.y  # 300


# =============================================================================
# Struct with Different Types
# =============================================================================

@compile
def test_struct_mixed_types() -> i32:
    """Test struct with different types"""
    s: struct[i8, i16, i32, i64] = (1, 2, 3, 4)
    a: i8 = s[0]
    b: i16 = s[1]
    c: i32 = s[2]
    d: i64 = s[3]
    return i32(a) + i32(b) + c + i32(d)  # 10


@compile
def test_struct_with_float() -> i32:
    """Test struct with float fields"""
    s: struct[i32, f64, i32] = (10, 3.14, 20)
    a: i32 = s[0]
    b: f64 = s[1]
    c: i32 = s[2]
    return a + i32(b) + c  # 10 + 3 + 20 = 33


@compile
def test_struct_with_ptr() -> i32:
    """Test struct with pointer field"""
    x: i32 = 42
    s: struct[ptr[i32], i32] = (ptr(x), 10)
    p: ptr[i32] = s[0]
    return p[0] + s[1]  # 52


@compile
def test_struct_with_bool() -> i32:
    """Test struct with bool field"""
    s: struct[bool, i32, bool] = (True, 42, False)
    result: i32 = 0
    if s[0]:
        result = result + 1
    result = result + s[1]
    if not s[2]:
        result = result + 1
    return result  # 44


# =============================================================================
# Nested Structs
# =============================================================================

@compile
def test_nested_struct_basic() -> i32:
    """Test basic nested struct"""
    inner: struct[i32, i32] = (10, 20)
    outer: struct[i32, struct[i32, i32]] = (5, inner)
    return outer[0] + outer[1][0] + outer[1][1]  # 35


@compile
def test_nested_struct_deep() -> i32:
    """Test deeply nested structs (3 levels)"""
    l1: struct[i32, i32] = (1, 2)
    l2: struct[i32, struct[i32, i32]] = (3, l1)
    l3: struct[i32, struct[i32, struct[i32, i32]]] = (4, l2)
    
    v1: i32 = l3[0]
    v2: i32 = l3[1][0]
    v3: i32 = l3[1][1][0]
    v4: i32 = l3[1][1][1]
    return v1 + v2 + v3 + v4  # 10


@compile
def test_nested_struct_modify() -> i32:
    """Test modifying nested struct"""
    inner: struct[i32, i32] = (10, 20)
    outer: struct[struct[i32, i32], i32] = (inner, 30)
    
    # Modify inner struct
    inner2: struct[i32, i32] = outer[0]
    inner2[0] = 100
    
    return inner2[0] + inner2[1] + outer[1]  # 150


@compile
def test_nested_struct_named() -> i32:
    """Test nested struct with named fields"""
    point: struct[x: i32, y: i32] = (10, 20)
    rect: struct[origin: struct[x: i32, y: i32], width: i32, height: i32] = (point, 100, 50)
    
    ox: i32 = rect.origin.x
    oy: i32 = rect.origin.y
    w: i32 = rect.width
    h: i32 = rect.height
    return ox + oy + w + h  # 180


# =============================================================================
# Struct with Arrays
# =============================================================================

@compile
def test_struct_with_array() -> i32:
    """Test struct containing array"""
    s: struct[i32, array[i32, 3]] = struct[i32, array[i32, 3]]()
    s[0] = 10
    s[1][0] = 1
    s[1][1] = 2
    s[1][2] = 3
    return s[0] + s[1][0] + s[1][1] + s[1][2]  # 16


@compile
def test_struct_multiple_arrays() -> i32:
    """Test struct with multiple array fields"""
    s: struct[array[i32, 2], array[i32, 3]] = struct[array[i32, 2], array[i32, 3]]()
    s[0][0] = 1
    s[0][1] = 2
    s[1][0] = 10
    s[1][1] = 20
    s[1][2] = 30
    return s[0][0] + s[0][1] + s[1][0] + s[1][1] + s[1][2]  # 63


@compile
def test_array_of_structs() -> i32:
    """Test array of structs"""
    arr: array[struct[i32, i32], 3] = array[struct[i32, i32], 3]()
    arr[0][0] = 1
    arr[0][1] = 2
    arr[1][0] = 3
    arr[1][1] = 4
    arr[2][0] = 5
    arr[2][1] = 6
    
    sum: i32 = 0
    i: i32 = 0
    while i < 3:
        sum = sum + arr[i][0] + arr[i][1]
        i = i + 1
    return sum  # 21


@compile
def test_array_of_nested_structs() -> i32:
    """Test array of nested structs"""
    arr: array[struct[i32, struct[i32, i32]], 2] = array[struct[i32, struct[i32, i32]], 2]()
    
    inner1: struct[i32, i32] = (10, 20)
    inner2: struct[i32, i32] = (30, 40)
    
    arr[0][0] = 1
    arr[0][1] = inner1
    arr[1][0] = 2
    arr[1][1] = inner2
    
    s1: struct[i32, i32] = arr[0][1]
    s2: struct[i32, i32] = arr[1][1]
    
    return arr[0][0] + arr[1][0] + s1[0] + s1[1] + s2[0] + s2[1]  # 103


# =============================================================================
# Struct Assignment and Copy
# =============================================================================

@compile
def test_struct_assignment() -> i32:
    """Test struct assignment"""
    s1: struct[i32, i32] = (10, 20)
    s2: struct[i32, i32] = s1
    return s2[0] + s2[1]  # 30


@compile
def test_struct_copy_independence() -> i32:
    """Test that struct copy is independent"""
    s1: struct[i32, i32] = (10, 20)
    s2: struct[i32, i32] = s1
    s2[0] = 100
    return s1[0] + s2[0]  # 110 (s1 unchanged)


@compile
def test_struct_return() -> struct[i32, i32]:
    """Test returning struct from function"""
    s: struct[i32, i32] = (42, 24)
    return s


@compile
def test_struct_return_use() -> i32:
    """Test using returned struct"""
    s: struct[i32, i32] = test_struct_return()
    return s[0] + s[1]  # 66


@compile
def test_struct_param(s: struct[i32, i32]) -> i32:
    """Test struct as parameter"""
    return s[0] + s[1]


@compile
def test_struct_param_call() -> i32:
    """Test calling function with struct parameter"""
    s: struct[i32, i32] = (10, 20)
    return test_struct_param(s)  # 30


# =============================================================================
# Union Basic Operations
# =============================================================================

@compile
def test_union_basic() -> i32:
    """Test basic union operations"""
    u: union[i32, f64] = union[i32, f64]()
    u[0] = 42
    return u[0]


@compile
def test_union_named() -> i32:
    """Test union with named fields"""
    u: union[i: i32, f: f64] = union[i: i32, f: f64]()
    u.i = 42
    return u.i


@compile
def test_union_type_punning() -> i32:
    """Test union type punning (reinterpret bits)"""
    u: union[i32, f32] = union[i32, f32]()
    u[1] = f32(1.0)  # Store as float
    i_val: i32 = u[0]  # Read as int
    # IEEE 754: 1.0f = 0x3F800000
    return i_val


@compile
def test_union_overwrite() -> i32:
    """Test union field overwrite"""
    u: union[i32, i64] = union[i32, i64]()
    u[1] = 0x123456789ABCDEF0
    u[0] = 42  # Overwrites lower bits
    return u[0]


# =============================================================================
# Union with Different Types
# =============================================================================

@compile
def test_union_mixed_sizes() -> i32:
    """Test union with different sized types"""
    u: union[i8, i16, i32, i64] = union[i8, i16, i32, i64]()
    u[3] = 0x0102030405060708  # Write as i64
    a: i8 = u[0]  # Read lower 8 bits
    b: i16 = u[1]  # Read lower 16 bits
    c: i32 = u[2]  # Read lower 32 bits
    return i32(u8(a))  # 0x08 = 8


@compile
def test_union_with_ptr() -> i32:
    """Test union with pointer type"""
    x: i32 = 42
    u: union[ptr[i32], i64] = union[ptr[i32], i64]()
    u[0] = ptr(x)
    p: ptr[i32] = u[0]
    return p[0]


@compile
def test_union_with_struct() -> i32:
    """Test union containing struct"""
    u: union[i32, struct[i16, i16]] = union[i32, struct[i16, i16]]()
    u[0] = 0x00020001  # Lower 16 bits = 1, upper 16 bits = 2
    s: struct[i16, i16] = u[1]
    return i32(s[0]) + i32(s[1])  # 1 + 2 = 3 (little-endian)


# =============================================================================
# Struct in Union and Vice Versa
# =============================================================================

@compile
def test_struct_with_union() -> i32:
    """Test struct containing union"""
    s: struct[i32, union[i32, f64]] = struct[i32, union[i32, f64]]()
    s[0] = 10
    u: union[i32, f64] = s[1]
    u[0] = 42
    return s[0] + u[0]  # 52


@compile
def test_union_with_nested_struct() -> i32:
    """Test union containing nested struct"""
    u: union[i64, struct[i32, i32]] = union[i64, struct[i32, i32]]()
    s: struct[i32, i32] = u[1]
    s[0] = 10
    s[1] = 20
    return s[0] + s[1]  # 30


# =============================================================================
# Large Structs
# =============================================================================

@compile
def test_large_struct() -> i32:
    """Test struct with many fields"""
    s: struct[i32, i32, i32, i32, i32, i32, i32, i32] = (1, 2, 3, 4, 5, 6, 7, 8)
    return s[0] + s[1] + s[2] + s[3] + s[4] + s[5] + s[6] + s[7]  # 36


@compile
def test_large_struct_with_array() -> i32:
    """Test large struct with array field"""
    s: struct[i32, array[i32, 10], i32] = struct[i32, array[i32, 10], i32]()
    s[0] = 100
    s[2] = 200
    i: i32 = 0
    while i < 10:
        s[1][i] = i + 1
        i = i + 1
    
    sum: i32 = s[0] + s[2]
    i = 0
    while i < 10:
        sum = sum + s[1][i]
        i = i + 1
    return sum  # 100 + 200 + 55 = 355


# =============================================================================
# Struct Alignment and Size
# =============================================================================

@compile
def test_struct_different_alignments() -> i32:
    """Test struct with different alignment requirements"""
    # i8 (1), i32 (4), i8 (1), i64 (8)
    # With padding: 1 + 3(pad) + 4 + 1 + 7(pad) + 8 = 24
    s: struct[i8, i32, i8, i64] = struct[i8, i32, i8, i64]()
    s[0] = 1
    s[1] = 2
    s[2] = 3
    s[3] = 4
    return i32(s[0]) + s[1] + i32(s[2]) + i32(s[3])  # 10


@compile
def test_struct_packed_like() -> i32:
    """Test struct with same-size fields (naturally packed)"""
    s: struct[i32, i32, i32, i32] = (10, 20, 30, 40)
    return s[0] + s[1] + s[2] + s[3]  # 100


# =============================================================================
# Struct Pointer Operations
# =============================================================================

@compile
def test_ptr_to_struct() -> i32:
    """Test pointer to struct"""
    s: struct[i32, i32] = (10, 20)
    p: ptr[struct[i32, i32]] = ptr(s)
    ps: struct[i32, i32] = p[0]
    return ps[0] + ps[1]  # 30


@compile
def test_ptr_to_struct_field() -> i32:
    """Test pointer to struct field"""
    s: struct[i32, i32, i32] = (10, 20, 30)
    p: ptr[i32] = ptr(s[1])
    return p[0]  # 20


@compile
def test_modify_struct_via_ptr() -> i32:
    """Test modifying struct through pointer"""
    s: struct[i32, i32] = (10, 20)
    p: ptr[i32] = ptr(s[0])
    p[0] = 100
    return s[0]  # 100


@compile
def test_struct_ptr_arithmetic() -> i32:
    """Test pointer arithmetic on struct array"""
    arr: array[struct[i32, i32], 3] = array[struct[i32, i32], 3]()
    arr[0][0] = 1
    arr[0][1] = 2
    arr[1][0] = 3
    arr[1][1] = 4
    arr[2][0] = 5
    arr[2][1] = 6
    
    p: ptr[struct[i32, i32]] = arr
    s1: struct[i32, i32] = p[1]  # Second struct
    return s1[0] + s1[1]  # 7


# =============================================================================
# Edge Cases
# =============================================================================

@compile
def test_struct_single_field() -> i32:
    """Test struct with single field"""
    s: struct[i32] = (42,)
    return s[0]


@compile
def test_struct_all_same_type() -> i32:
    """Test struct where all fields are same type"""
    s: struct[i32, i32, i32, i32, i32] = (1, 2, 3, 4, 5)
    return s[0] + s[1] + s[2] + s[3] + s[4]  # 15


@compile
def test_union_single_type() -> i32:
    """Test union with single type"""
    u: union[i32] = union[i32]()
    u[0] = 42
    return u[0]


@compile
def test_nested_struct_in_array() -> i32:
    """Test deeply nested struct in array"""
    arr: array[struct[struct[i32, i32], i32], 2] = array[struct[struct[i32, i32], i32], 2]()
    
    inner: struct[i32, i32] = (10, 20)
    arr[0][0] = inner
    arr[0][1] = 30
    
    inner2: struct[i32, i32] = (40, 50)
    arr[1][0] = inner2
    arr[1][1] = 60
    
    s0: struct[i32, i32] = arr[0][0]
    s1: struct[i32, i32] = arr[1][0]
    
    return s0[0] + s0[1] + arr[0][1] + s1[0] + s1[1] + arr[1][1]  # 210


# =============================================================================
# Struct Comparison Patterns
# =============================================================================

@compile
def test_struct_field_compare() -> i32:
    """Test comparing struct fields"""
    s1: struct[i32, i32] = (10, 20)
    s2: struct[i32, i32] = (10, 30)
    
    result: i32 = 0
    if s1[0] == s2[0]:
        result = result + 1
    if s1[1] != s2[1]:
        result = result + 1
    if s1[1] < s2[1]:
        result = result + 1
    return result  # 3


@compile
def test_struct_conditional_init() -> i32:
    """Test conditional struct initialization"""
    flag: i32 = 1
    s: struct[i32, i32]
    if flag == 1:
        s = (10, 20)
    else:
        s = (30, 40)
    return s[0] + s[1]  # 30


# =============================================================================
# Test Runner
# =============================================================================

class TestBasicStruct(unittest.TestCase):
    def test_create_tuple(self):
        self.assertEqual(test_struct_create_tuple(), 30)
    
    def test_create_default(self):
        self.assertEqual(test_struct_create_default(), 30)
    
    def test_named_fields(self):
        self.assertEqual(test_struct_named_fields(), 30)
    
    def test_mixed_access(self):
        self.assertEqual(test_struct_mixed_access(), 60)
    
    def test_modify_field(self):
        self.assertEqual(test_struct_modify_field(), 300)
    
    def test_modify_named(self):
        self.assertEqual(test_struct_modify_named(), 300)


class TestStructTypes(unittest.TestCase):
    def test_mixed_types(self):
        self.assertEqual(test_struct_mixed_types(), 10)
    
    def test_with_float(self):
        self.assertEqual(test_struct_with_float(), 33)
    
    def test_with_ptr(self):
        self.assertEqual(test_struct_with_ptr(), 52)
    
    def test_with_bool(self):
        self.assertEqual(test_struct_with_bool(), 44)


class TestNestedStruct(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(test_nested_struct_basic(), 35)
    
    def test_deep(self):
        self.assertEqual(test_nested_struct_deep(), 10)
    
    def test_modify(self):
        self.assertEqual(test_nested_struct_modify(), 150)
    
    def test_named(self):
        self.assertEqual(test_nested_struct_named(), 180)


class TestStructArray(unittest.TestCase):
    def test_with_array(self):
        self.assertEqual(test_struct_with_array(), 16)
    
    def test_multiple_arrays(self):
        self.assertEqual(test_struct_multiple_arrays(), 63)
    
    def test_array_of_structs(self):
        self.assertEqual(test_array_of_structs(), 21)
    
    def test_array_of_nested(self):
        self.assertEqual(test_array_of_nested_structs(), 103)


class TestStructAssignment(unittest.TestCase):
    def test_assignment(self):
        self.assertEqual(test_struct_assignment(), 30)
    
    def test_copy_independence(self):
        self.assertEqual(test_struct_copy_independence(), 110)
    
    def test_return_use(self):
        self.assertEqual(test_struct_return_use(), 66)
    
    def test_param_call(self):
        self.assertEqual(test_struct_param_call(), 30)


class TestUnion(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(test_union_basic(), 42)
    
    def test_named(self):
        self.assertEqual(test_union_named(), 42)
    
    def test_type_punning(self):
        self.assertEqual(test_union_type_punning(), 0x3F800000)
    
    def test_overwrite(self):
        self.assertEqual(test_union_overwrite(), 42)


class TestUnionTypes(unittest.TestCase):
    def test_mixed_sizes(self):
        self.assertEqual(test_union_mixed_sizes(), 8)
    
    def test_with_ptr(self):
        self.assertEqual(test_union_with_ptr(), 42)
    
    def test_with_struct(self):
        self.assertEqual(test_union_with_struct(), 3)


class TestStructUnionCombo(unittest.TestCase):
    def test_struct_with_union(self):
        self.assertEqual(test_struct_with_union(), 52)
    
    def test_union_with_nested_struct(self):
        self.assertEqual(test_union_with_nested_struct(), 30)


class TestLargeStruct(unittest.TestCase):
    def test_many_fields(self):
        self.assertEqual(test_large_struct(), 36)
    
    def test_with_array(self):
        self.assertEqual(test_large_struct_with_array(), 355)


class TestStructAlignment(unittest.TestCase):
    def test_different_alignments(self):
        self.assertEqual(test_struct_different_alignments(), 10)
    
    def test_packed_like(self):
        self.assertEqual(test_struct_packed_like(), 100)


class TestStructPointer(unittest.TestCase):
    def test_ptr_to_struct(self):
        self.assertEqual(test_ptr_to_struct(), 30)
    
    def test_ptr_to_field(self):
        self.assertEqual(test_ptr_to_struct_field(), 20)
    
    def test_modify_via_ptr(self):
        self.assertEqual(test_modify_struct_via_ptr(), 100)
    
    def test_ptr_arithmetic(self):
        self.assertEqual(test_struct_ptr_arithmetic(), 7)


class TestEdgeCases(unittest.TestCase):
    def test_single_field(self):
        self.assertEqual(test_struct_single_field(), 42)
    
    def test_all_same_type(self):
        self.assertEqual(test_struct_all_same_type(), 15)
    
    def test_union_single(self):
        self.assertEqual(test_union_single_type(), 42)
    
    def test_nested_in_array(self):
        self.assertEqual(test_nested_struct_in_array(), 210)


class TestStructPatterns(unittest.TestCase):
    def test_field_compare(self):
        self.assertEqual(test_struct_field_compare(), 3)
    
    def test_conditional_init(self):
        self.assertEqual(test_struct_conditional_init(), 30)


if __name__ == '__main__':
    unittest.main()
