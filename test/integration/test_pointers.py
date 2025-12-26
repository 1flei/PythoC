#!/usr/bin/env python3
"""
Pointer operation tests
"""

import unittest
from pythoc import i32, i64, ptr, compile, nullptr, array


@compile
def test_ptr_basic() -> i32:
    """Test basic pointer operations"""
    x: i32 = 42
    p: ptr[i32] = ptr(x)
    
    return p[0]


@compile
def test_ptr_arithmetic() -> i32:
    """Test pointer arithmetic"""
    arr: ptr[i32] = ptr[i32](nullptr)
    p1: ptr[i32] = arr + 5
    p2: ptr[i32] = p1 - 2
    
    diff: i64 = i64(p2) - i64(arr)
    
    return i32(diff / 4)


@compile
def test_ptr_deref_assign() -> i32:
    """Test pointer dereference and assignment"""
    x: i32 = 10
    p: ptr[i32] = ptr(x)
    
    p[0] = 20
    
    return x


@compile
def test_ptr_to_ptr() -> i32:
    """Test pointer to pointer"""
    x: i32 = 42
    p: ptr[i32] = ptr(x)
    pp: ptr[ptr[i32]] = ptr(p)
    
    return pp[0][0]


@compile
def test_nullptr_check() -> i32:
    """Test nullptr"""
    p: ptr[i32] = ptr[i32](nullptr)
    
    if p == ptr[i32](nullptr):
        return 1
    else:
        return 0


@compile
def test_ptr_int() -> i32:
    arr: array[i32, 10] = [1, 2, 3, 4]
    p: ptr[i32] = ptr[i32](arr)
    x: i32 = p[3]
    return x


@compile
def test_ptr_int_decay() -> i32:
    arr: array[i32, 10] = [1, 2, 3, 4]
    p: ptr[i32] = arr
    x: i32 = p[3]
    return x


@compile
def test_ptr_array() -> i32:
    arr: array[i32, 10] = [1, 2, 3, 4]
    p: ptr[array[i32, 10]] = ptr(arr)
    x: i32 = p[0][3]
    return x


@compile
def test_ptr_array_arithmetic() -> i32:
    """Test pointer arithmetic: ptr(arr) vs decayed array pointer
    
    Key difference in C semantics:
    - arr decays to ptr[i32]: points to first element, +1 skips 4 bytes
    - ptr(arr) is ptr[array[i32, 10]]: points to whole array, +1 skips 40 bytes
    """
    # Create two adjacent arrays to demonstrate pointer arithmetic
    arr1: array[i32, 10] = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    arr2: array[i32, 10] = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    
    # Test 1: Decayed pointer (ptr[i32]) - points to element
    p_elem: ptr[i32] = arr1  # Decays to ptr[i32]
    elem_0: i32 = p_elem[0]   # Should be 10
    elem_1: i32 = p_elem[1]   # Should be 11
    elem_3: i32 = p_elem[3]   # Should be 13
    
    # Test 2: Pointer to array (ptr[array[i32, 10]]) - points to whole array
    p_arr: ptr[array[i32, 10]] = ptr(arr1)
    arr_elem: i32 = p_arr[0][3]  # p_arr[0] is the array, [3] gets element
    
    # Test 3: Cast ptr(arr) to get the underlying address
    # This demonstrates that ptr(arr) and arr have the same address
    # but different types
    p_elem_from_arr: ptr[i32] = ptr[i32](p_arr)
    same_elem: i32 = p_elem_from_arr[3]  # Should also be 13
    
    # Sum all results to verify correctness
    # 10 + 11 + 13 + 13 + 13 = 60
    return elem_0 + elem_1 + elem_3 + arr_elem + same_elem


@compile
def test_multidim_array_ptr() -> i32:
    """Test pointer arithmetic with multi-dimensional arrays
    
    For arr: array[i32, 3, 4]:
    - arr decays to ptr[array[i32, 4]]: +1 skips one row (4 i32s = 16 bytes)
    - ptr(arr) is ptr[array[i32, 3, 4]]: +1 skips entire array (12 i32s = 48 bytes)
    """
    arr: array[i32, 3, 4] = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ]
    
    # Test 1: Natural decay - arr decays to ptr[array[i32, 4]]
    p_row: ptr[array[i32, 4]] = arr  # Points to first row
    val_0_2: i32 = p_row[0][2]  # First row, third element = 3
    val_1_3: i32 = p_row[1][3]  # Second row, fourth element = 8
    
    # Test 2: Pointer to entire 2D array
    p_arr: ptr[array[i32, 3, 4]] = ptr(arr)
    val_whole: i32 = p_arr[0][2][1]  # [0] gets the array, [2] gets third row, [1] gets second element = 10
    p_arr_i64_1 = i64(p_arr + 1)
    p_arr_i64 = i64(p_arr)
    diff = p_arr_i64_1 - p_arr_i64   # Should be 48
    
    # 3 + 8 + 10 + 48 = 69
    return val_0_2 + val_1_3 + val_whole + diff


class TestPointers(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(test_ptr_basic(), 42)
    
    def test_arithmetic(self):
        self.assertEqual(test_ptr_arithmetic(), 3)
    
    def test_deref_assign(self):
        self.assertEqual(test_ptr_deref_assign(), 20)
    
    def test_to_ptr(self):
        self.assertEqual(test_ptr_to_ptr(), 42)
    
    def test_nullptr(self):
        self.assertEqual(test_nullptr_check(), 1)
    
    def test_int(self):
        self.assertEqual(test_ptr_int(), 4)
    
    def test_int_decay(self):
        self.assertEqual(test_ptr_int_decay(), 4)
    
    def test_array(self):
        self.assertEqual(test_ptr_array(), 4)
    
    def test_array_arithmetic(self):
        self.assertEqual(test_ptr_array_arithmetic(), 60)
    
    def test_multidim(self):
        self.assertEqual(test_multidim_array_ptr(), 69)


if __name__ == "__main__":
    unittest.main()
