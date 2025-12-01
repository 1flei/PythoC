#!/usr/bin/env python3
"""
Comprehensive tests for pointer operations including edge cases,
complex pointer arithmetic, multi-level pointers, and aliasing.
"""

import unittest
from pythoc import (
    i8, i16, i32, i64,
    u8, u16, u32, u64,
    f32, f64, bool, ptr, array, struct, compile, nullptr
)


# =============================================================================
# Basic Pointer Operations
# =============================================================================

@compile
def test_ptr_create_and_deref() -> i32:
    """Test basic pointer creation and dereference"""
    x: i32 = 42
    p: ptr[i32] = ptr(x)
    return p[0]


@compile
def test_ptr_write_through() -> i32:
    """Test writing through pointer"""
    x: i32 = 10
    p: ptr[i32] = ptr(x)
    p[0] = 42
    return x  # Should be 42


@compile
def test_ptr_multiple_refs() -> i32:
    """Test multiple pointers to same variable"""
    x: i32 = 10
    p1: ptr[i32] = ptr(x)
    p2: ptr[i32] = ptr(x)
    p1[0] = 20
    return p2[0]  # Should be 20


@compile
def test_ptr_chain_assignment() -> i32:
    """Test pointer assignment chain"""
    x: i32 = 42
    p1: ptr[i32] = ptr(x)
    p2: ptr[i32] = p1
    p3: ptr[i32] = p2
    return p3[0]


# =============================================================================
# Pointer Arithmetic
# =============================================================================

@compile
def test_ptr_add_offset() -> i32:
    """Test pointer addition with offset"""
    arr: array[i32, 5] = [10, 20, 30, 40, 50]
    p: ptr[i32] = arr
    p2: ptr[i32] = p + 2
    return p2[0]  # 30


@compile
def test_ptr_sub_offset() -> i32:
    """Test pointer subtraction with offset"""
    arr: array[i32, 5] = [10, 20, 30, 40, 50]
    p: ptr[i32] = arr + 4
    p2: ptr[i32] = p - 2
    return p2[0]  # 30


@compile
def test_ptr_increment_loop() -> i32:
    """Test pointer increment in loop"""
    arr: array[i32, 5] = [1, 2, 3, 4, 5]
    p: ptr[i32] = arr
    sum: i32 = 0
    i: i32 = 0
    while i < 5:
        sum = sum + p[0]
        p = p + 1
        i = i + 1
    return sum  # 15


@compile
def test_ptr_diff() -> i64:
    """Test pointer difference"""
    arr: array[i32, 10] = array[i32, 10]()
    p1: ptr[i32] = arr
    p2: ptr[i32] = arr + 5
    diff: i64 = i64(p2) - i64(p1)
    return diff / 4  # Should be 5 (5 * sizeof(i32))


@compile
def test_ptr_negative_offset() -> i32:
    """Test pointer with negative offset"""
    arr: array[i32, 5] = [10, 20, 30, 40, 50]
    p: ptr[i32] = arr + 3
    return p[-2]  # arr[1] = 20


@compile
def test_ptr_arithmetic_chain() -> i32:
    """Test chained pointer arithmetic"""
    arr: array[i32, 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    p: ptr[i32] = arr
    p = p + 3  # points to arr[3]
    p = p + 2  # points to arr[5]
    p = p - 1  # points to arr[4]
    return p[0]  # 4


# =============================================================================
# Multi-level Pointers
# =============================================================================

@compile
def test_ptr_to_ptr() -> i32:
    """Test pointer to pointer"""
    x: i32 = 42
    p: ptr[i32] = ptr(x)
    pp: ptr[ptr[i32]] = ptr(p)
    return pp[0][0]


@compile
def test_ptr_to_ptr_write() -> i32:
    """Test writing through pointer to pointer"""
    x: i32 = 10
    p: ptr[i32] = ptr(x)
    pp: ptr[ptr[i32]] = ptr(p)
    pp[0][0] = 42
    return x  # Should be 42


@compile
def test_ptr_to_ptr_reassign() -> i32:
    """Test reassigning through pointer to pointer"""
    x: i32 = 10
    y: i32 = 20
    px: ptr[i32] = ptr(x)
    py: ptr[i32] = ptr(y)
    pp: ptr[ptr[i32]] = ptr(px)
    pp[0] = py  # Now pp[0] points to y
    return pp[0][0]  # Should be 20


@compile
def test_triple_ptr() -> i32:
    """Test triple pointer (ptr to ptr to ptr)"""
    x: i32 = 42
    p1: ptr[i32] = ptr(x)
    p2: ptr[ptr[i32]] = ptr(p1)
    p3: ptr[ptr[ptr[i32]]] = ptr(p2)
    return p3[0][0][0]


# =============================================================================
# Pointer Type Casting
# =============================================================================

@compile
def test_ptr_cast_i32_to_i8() -> i32:
    """Test casting i32 pointer to i8 pointer"""
    x: i32 = 0x12345678
    p32: ptr[i32] = ptr(x)
    p8: ptr[i8] = ptr[i8](p32)
    # Access individual bytes (little-endian)
    b0: i8 = p8[0]  # 0x78
    b1: i8 = p8[1]  # 0x56
    b2: i8 = p8[2]  # 0x34
    b3: i8 = p8[3]  # 0x12
    return i32(u8(b0))  # 0x78 = 120


@compile
def test_ptr_cast_to_int_back() -> i32:
    """Test casting pointer to int and back"""
    x: i32 = 42
    p: ptr[i32] = ptr(x)
    addr: i64 = i64(p)
    p2: ptr[i32] = ptr[i32](addr)
    return p2[0]


@compile
def test_ptr_cast_different_sizes() -> i32:
    """Test casting between pointers of different sizes"""
    x: i64 = 0x0102030405060708
    p64: ptr[i64] = ptr(x)
    p32: ptr[i32] = ptr[i32](p64)
    # Access as two i32 values
    low: i32 = p32[0]  # 0x05060708
    high: i32 = p32[1]  # 0x01020304
    return low & 0xFF  # 0x08 = 8


@compile
def test_ptr_cast_struct_to_array() -> i32:
    """Test casting struct pointer to element pointer"""
    s: struct[i32, i32, i32] = (10, 20, 30)
    ps: ptr[struct[i32, i32, i32]] = ptr(s)
    p32: ptr[i32] = ptr[i32](ps)
    return p32[0] + p32[1] + p32[2]  # 60


# =============================================================================
# Pointer Comparisons
# =============================================================================

@compile
def test_ptr_equal() -> i32:
    """Test pointer equality"""
    x: i32 = 42
    p1: ptr[i32] = ptr(x)
    p2: ptr[i32] = p1
    if p1 == p2:
        return 1
    return 0


@compile
def test_ptr_not_equal() -> i32:
    """Test pointer inequality"""
    x: i32 = 42
    y: i32 = 42
    px: ptr[i32] = ptr(x)
    py: ptr[i32] = ptr(y)
    if px != py:
        return 1
    return 0


@compile
def test_ptr_less_than() -> i32:
    """Test pointer less than comparison"""
    arr: array[i32, 5] = array[i32, 5]()
    p1: ptr[i32] = arr
    p2: ptr[i32] = arr + 3
    if p1 < p2:
        return 1
    return 0


@compile
def test_ptr_greater_than() -> i32:
    """Test pointer greater than comparison"""
    arr: array[i32, 5] = array[i32, 5]()
    p1: ptr[i32] = arr + 3
    p2: ptr[i32] = arr
    if p1 > p2:
        return 1
    return 0


@compile
def test_ptr_null_check() -> i32:
    """Test null pointer check"""
    p: ptr[i32] = ptr[i32](nullptr)
    if p == ptr[i32](nullptr):
        return 1
    return 0


@compile
def test_ptr_not_null_check() -> i32:
    """Test non-null pointer check"""
    x: i32 = 42
    p: ptr[i32] = ptr(x)
    if p != ptr[i32](nullptr):
        return 1
    return 0


# =============================================================================
# Pointer with Arrays
# =============================================================================

@compile
def test_array_decay_to_ptr() -> i32:
    """Test array decay to pointer"""
    arr: array[i32, 5] = [1, 2, 3, 4, 5]
    p: ptr[i32] = arr  # Implicit decay
    return p[2]  # 3


@compile
def test_ptr_iterate_array() -> i32:
    """Test iterating array via pointer"""
    arr: array[i32, 5] = [10, 20, 30, 40, 50]
    p: ptr[i32] = arr
    sum: i32 = 0
    i: i32 = 0
    while i < 5:
        sum = sum + p[i]
        i = i + 1
    return sum  # 150


@compile
def test_ptr_modify_array() -> i32:
    """Test modifying array through pointer"""
    arr: array[i32, 5] = [1, 2, 3, 4, 5]
    p: ptr[i32] = arr
    p[2] = 100
    return arr[2]  # 100


@compile
def test_ptr_to_array_element() -> i32:
    """Test pointer to specific array element"""
    arr: array[i32, 5] = [10, 20, 30, 40, 50]
    p: ptr[i32] = ptr(arr[2])
    return p[0]  # 30


@compile
def test_ptr_array_bounds() -> i32:
    """Test pointer access at array bounds"""
    arr: array[i32, 5] = [1, 2, 3, 4, 5]
    p: ptr[i32] = arr
    first: i32 = p[0]
    last: i32 = p[4]
    return first + last  # 6


@compile
def test_2d_array_ptr() -> i32:
    """Test pointer with 2D array"""
    arr: array[i32, 3, 4] = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ]
    p: ptr[array[i32, 4]] = arr
    row1: ptr[i32] = p[1]
    return row1[2]  # 7


# =============================================================================
# Pointer with Structs
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
    p: ptr[struct[i32, i32]] = ptr(s)
    ps: struct[i32, i32] = p[0]
    ps[0] = 100
    return ps[0]  # 100


@compile
def test_array_of_struct_ptr() -> i32:
    """Test pointer to array of structs"""
    arr: array[struct[i32, i32], 3] = array[struct[i32, i32], 3]()
    arr[0][0] = 1
    arr[0][1] = 2
    arr[1][0] = 3
    arr[1][1] = 4
    arr[2][0] = 5
    arr[2][1] = 6
    
    p: ptr[struct[i32, i32]] = arr
    s1: struct[i32, i32] = p[1]
    return s1[0] + s1[1]  # 7


@compile
def test_struct_with_ptr_field() -> i32:
    """Test struct containing pointer field"""
    x: i32 = 42
    s: struct[ptr[i32], i32]
    s[0] = ptr(x)
    s[1] = 10
    p: ptr[i32] = s[0]
    return p[0] + s[1]  # 52


# =============================================================================
# Pointer Aliasing
# =============================================================================

@compile
def test_aliasing_same_type() -> i32:
    """Test aliasing with same type pointers"""
    x: i32 = 10
    p1: ptr[i32] = ptr(x)
    p2: ptr[i32] = ptr(x)
    p1[0] = 20
    return p2[0]  # 20


@compile
def test_aliasing_different_types() -> i32:
    """Test aliasing with different type pointers (type punning)"""
    x: i32 = 0x41424344  # "DCBA" in ASCII
    p32: ptr[i32] = ptr(x)
    p8: ptr[i8] = ptr[i8](p32)
    # Modify through i8 pointer
    p8[0] = 0x58  # 'X'
    return p32[0] & 0xFF  # Should be 0x58 = 88


@compile
def test_aliasing_array_elements() -> i32:
    """Test aliasing between array elements"""
    arr: array[i32, 5] = [1, 2, 3, 4, 5]
    p1: ptr[i32] = ptr(arr[2])
    p2: ptr[i32] = arr + 2
    p1[0] = 100
    return p2[0]  # 100


# =============================================================================
# Pointer Function Patterns
# =============================================================================

@compile
def swap_via_ptr(a: ptr[i32], b: ptr[i32]) -> i32:
    """Swap two values via pointers"""
    temp: i32 = a[0]
    a[0] = b[0]
    b[0] = temp
    return 0


@compile
def test_swap_pattern() -> i32:
    """Test swap pattern using pointers"""
    x: i32 = 10
    y: i32 = 20
    swap_via_ptr(ptr(x), ptr(y))
    return x * 100 + y  # 2010


@compile
def increment_via_ptr(p: ptr[i32]) -> i32:
    """Increment value via pointer"""
    p[0] = p[0] + 1
    return p[0]


@compile
def test_increment_pattern() -> i32:
    """Test increment pattern using pointer"""
    x: i32 = 10
    increment_via_ptr(ptr(x))
    increment_via_ptr(ptr(x))
    increment_via_ptr(ptr(x))
    return x  # 13


@compile
def sum_array_via_ptr(arr: ptr[i32], len: i32) -> i32:
    """Sum array elements via pointer"""
    sum: i32 = 0
    i: i32 = 0
    while i < len:
        sum = sum + arr[i]
        i = i + 1
    return sum


@compile
def test_sum_array_pattern() -> i32:
    """Test array sum via pointer parameter"""
    arr: array[i32, 5] = [1, 2, 3, 4, 5]
    return sum_array_via_ptr(arr, 5)  # 15


# =============================================================================
# Complex Pointer Scenarios
# =============================================================================

@compile
def test_ptr_in_loop_condition() -> i32:
    """Test pointer comparison in loop condition"""
    arr: array[i32, 5] = [1, 2, 3, 4, 5]
    p: ptr[i32] = arr
    end: ptr[i32] = arr + 5
    sum: i32 = 0
    while p < end:
        sum = sum + p[0]
        p = p + 1
    return sum  # 15


@compile
def test_ptr_conditional_access() -> i32:
    """Test conditional pointer access"""
    x: i32 = 10
    y: i32 = 20
    flag: i32 = 1
    p: ptr[i32]
    if flag == 1:
        p = ptr(x)
    else:
        p = ptr(y)
    return p[0]  # 10


@compile
def test_ptr_array_of_ptrs() -> i32:
    """Test array of pointers"""
    a: i32 = 10
    b: i32 = 20
    c: i32 = 30
    ptrs: array[ptr[i32], 3] = array[ptr[i32], 3]()
    ptrs[0] = ptr(a)
    ptrs[1] = ptr(b)
    ptrs[2] = ptr(c)
    sum: i32 = 0
    i: i32 = 0
    while i < 3:
        p: ptr[i32] = ptrs[i]
        sum = sum + p[0]
        i = i + 1
    return sum  # 60


@compile
def test_ptr_linked_access() -> i32:
    """Test linked pointer access pattern"""
    a: i32 = 10
    b: i32 = 20
    c: i32 = 30
    
    # Create a chain: pa -> pb -> pc
    pa: ptr[i32] = ptr(a)
    pb: ptr[i32] = ptr(b)
    pc: ptr[i32] = ptr(c)
    
    # Store pointers in array
    chain: array[ptr[i32], 3] = array[ptr[i32], 3]()
    chain[0] = pa
    chain[1] = pb
    chain[2] = pc
    
    # Access through chain
    sum: i32 = 0
    i: i32 = 0
    while i < 3:
        p: ptr[i32] = chain[i]
        sum = sum + p[0]
        i = i + 1
    return sum  # 60


@compile
def test_ptr_reinterpret_struct() -> i32:
    """Test reinterpreting struct as array of bytes"""
    s: struct[i32, i32] = (0x01020304, 0x05060708)
    p: ptr[i8] = ptr[i8](ptr(s))
    # Access individual bytes (little-endian)
    return i32(u8(p[0])) + i32(u8(p[4]))  # 0x04 + 0x08 = 12


# =============================================================================
# Test Runner
# =============================================================================

class TestBasicPointer(unittest.TestCase):
    def test_create_deref(self):
        self.assertEqual(test_ptr_create_and_deref(), 42)
    
    def test_write_through(self):
        self.assertEqual(test_ptr_write_through(), 42)
    
    def test_multiple_refs(self):
        self.assertEqual(test_ptr_multiple_refs(), 20)
    
    def test_chain_assignment(self):
        self.assertEqual(test_ptr_chain_assignment(), 42)


class TestPointerArithmetic(unittest.TestCase):
    def test_add_offset(self):
        self.assertEqual(test_ptr_add_offset(), 30)
    
    def test_sub_offset(self):
        self.assertEqual(test_ptr_sub_offset(), 30)
    
    def test_increment_loop(self):
        self.assertEqual(test_ptr_increment_loop(), 15)
    
    def test_diff(self):
        self.assertEqual(test_ptr_diff(), 5)
    
    def test_negative_offset(self):
        self.assertEqual(test_ptr_negative_offset(), 20)
    
    def test_arithmetic_chain(self):
        self.assertEqual(test_ptr_arithmetic_chain(), 4)


class TestMultiLevelPointer(unittest.TestCase):
    def test_ptr_to_ptr(self):
        self.assertEqual(test_ptr_to_ptr(), 42)
    
    def test_ptr_to_ptr_write(self):
        self.assertEqual(test_ptr_to_ptr_write(), 42)
    
    def test_ptr_to_ptr_reassign(self):
        self.assertEqual(test_ptr_to_ptr_reassign(), 20)
    
    def test_triple_ptr(self):
        self.assertEqual(test_triple_ptr(), 42)


class TestPointerCasting(unittest.TestCase):
    def test_cast_i32_to_i8(self):
        self.assertEqual(test_ptr_cast_i32_to_i8(), 120)
    
    def test_cast_to_int_back(self):
        self.assertEqual(test_ptr_cast_to_int_back(), 42)
    
    def test_cast_different_sizes(self):
        self.assertEqual(test_ptr_cast_different_sizes(), 8)
    
    def test_cast_struct_to_array(self):
        self.assertEqual(test_ptr_cast_struct_to_array(), 60)


class TestPointerComparison(unittest.TestCase):
    def test_equal(self):
        self.assertEqual(test_ptr_equal(), 1)
    
    def test_not_equal(self):
        self.assertEqual(test_ptr_not_equal(), 1)
    
    def test_less_than(self):
        self.assertEqual(test_ptr_less_than(), 1)
    
    def test_greater_than(self):
        self.assertEqual(test_ptr_greater_than(), 1)
    
    def test_null_check(self):
        self.assertEqual(test_ptr_null_check(), 1)
    
    def test_not_null_check(self):
        self.assertEqual(test_ptr_not_null_check(), 1)


class TestPointerWithArray(unittest.TestCase):
    def test_decay(self):
        self.assertEqual(test_array_decay_to_ptr(), 3)
    
    def test_iterate(self):
        self.assertEqual(test_ptr_iterate_array(), 150)
    
    def test_modify(self):
        self.assertEqual(test_ptr_modify_array(), 100)
    
    def test_to_element(self):
        self.assertEqual(test_ptr_to_array_element(), 30)
    
    def test_bounds(self):
        self.assertEqual(test_ptr_array_bounds(), 6)
    
    def test_2d_array(self):
        self.assertEqual(test_2d_array_ptr(), 7)


class TestPointerWithStruct(unittest.TestCase):
    def test_to_struct(self):
        self.assertEqual(test_ptr_to_struct(), 30)
    
    def test_to_field(self):
        self.assertEqual(test_ptr_to_struct_field(), 20)
    
    def test_array_of_struct(self):
        self.assertEqual(test_array_of_struct_ptr(), 7)
    
    def test_struct_with_ptr(self):
        self.assertEqual(test_struct_with_ptr_field(), 52)


class TestAliasing(unittest.TestCase):
    def test_same_type(self):
        self.assertEqual(test_aliasing_same_type(), 20)
    
    def test_different_types(self):
        self.assertEqual(test_aliasing_different_types(), 88)
    
    def test_array_elements(self):
        self.assertEqual(test_aliasing_array_elements(), 100)


class TestPointerPatterns(unittest.TestCase):
    def test_swap(self):
        self.assertEqual(test_swap_pattern(), 2010)
    
    def test_increment(self):
        self.assertEqual(test_increment_pattern(), 13)
    
    def test_sum_array(self):
        self.assertEqual(test_sum_array_pattern(), 15)


class TestComplexPointer(unittest.TestCase):
    def test_loop_condition(self):
        self.assertEqual(test_ptr_in_loop_condition(), 15)
    
    def test_conditional_access(self):
        self.assertEqual(test_ptr_conditional_access(), 10)
    
    def test_array_of_ptrs(self):
        self.assertEqual(test_ptr_array_of_ptrs(), 60)
    
    def test_linked_access(self):
        self.assertEqual(test_ptr_linked_access(), 60)
    
    def test_reinterpret_struct(self):
        self.assertEqual(test_ptr_reinterpret_struct(), 12)


if __name__ == '__main__':
    unittest.main()
