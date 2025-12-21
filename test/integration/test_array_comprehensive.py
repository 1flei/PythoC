#!/usr/bin/env python3
"""
Comprehensive tests for arrays including multi-dimensional arrays,
array operations, edge cases, and complex array patterns.
"""

import unittest
from pythoc import (
    i8, i16, i32, i64,
    u8, u16, u32, u64,
    f32, f64, bool, ptr, array, struct, compile
)


# =============================================================================
# Basic Array Operations
# =============================================================================

@compile
def test_array_init_full() -> i32:
    """Test array with full initialization"""
    arr: array[i32, 5] = [1, 2, 3, 4, 5]
    return arr[0] + arr[1] + arr[2] + arr[3] + arr[4]  # 15


@compile
def test_array_init_partial() -> i32:
    """Test array with partial initialization"""
    arr: array[i32, 5] = [1, 2, 3]
    # Remaining elements should be zero
    return arr[0] + arr[1] + arr[2] + arr[3] + arr[4]  # 6


@compile
def test_array_init_default() -> i32:
    """Test array default initialization"""
    arr: array[i32, 5] = array[i32, 5]()
    return arr[0] + arr[1] + arr[2] + arr[3] + arr[4]  # 0


@compile
def test_array_modify() -> i32:
    """Test array element modification"""
    arr: array[i32, 5] = [1, 2, 3, 4, 5]
    arr[2] = 100
    return arr[2]  # 100


@compile
def test_array_read_write() -> i32:
    """Test array read and write"""
    arr: array[i32, 5] = array[i32, 5]()
    i: i32 = 0
    while i < 5:
        arr[i] = i * 10
        i = i + 1
    return arr[0] + arr[1] + arr[2] + arr[3] + arr[4]  # 0+10+20+30+40 = 100


# =============================================================================
# Array with Different Types
# =============================================================================

@compile
def test_array_i8() -> i32:
    """Test i8 array"""
    arr: array[i8, 5] = [10, 20, 30, 40, 50]
    return i32(arr[0]) + i32(arr[2]) + i32(arr[4])  # 90


@compile
def test_array_i16() -> i32:
    """Test i16 array"""
    arr: array[i16, 5] = [100, 200, 300, 400, 500]
    return i32(arr[1]) + i32(arr[3])  # 600


@compile
def test_array_i64() -> i64:
    """Test i64 array"""
    arr: array[i64, 3] = [1000000000, 2000000000, 3000000000]
    return arr[0] + arr[1] + arr[2]  # 6000000000


@compile
def test_array_f32() -> i32:
    """Test f32 array"""
    arr: array[f32, 3] = [f32(1.5), f32(2.5), f32(3.5)]
    sum: f32 = arr[0] + arr[1] + arr[2]
    return i32(sum)  # 7


@compile
def test_array_f64() -> i32:
    """Test f64 array"""
    arr: array[f64, 3] = [1.1, 2.2, 3.3]
    sum: f64 = arr[0] + arr[1] + arr[2]
    return i32(sum)  # 6


@compile
def test_array_bool() -> i32:
    """Test bool array"""
    arr: array[bool, 4] = [True, False, True, False]
    count: i32 = 0
    i: i32 = 0
    while i < 4:
        if arr[i]:
            count = count + 1
        i = i + 1
    return count  # 2


# =============================================================================
# Multi-dimensional Arrays
# =============================================================================

@compile
def test_2d_array_init() -> i32:
    """Test 2D array initialization"""
    arr: array[i32, 2, 3] = [[1, 2, 3], [4, 5, 6]]
    return arr[0][0] + arr[0][1] + arr[0][2] + arr[1][0] + arr[1][1] + arr[1][2]  # 21


@compile
def test_2d_array_modify() -> i32:
    """Test 2D array modification"""
    arr: array[i32, 2, 3] = [[1, 2, 3], [4, 5, 6]]
    arr[0][1] = 100
    arr[1][2] = 200
    return arr[0][1] + arr[1][2]  # 300


@compile
def test_2d_array_tuple_index() -> i32:
    """Test 2D array with tuple indexing"""
    arr: array[i32, 3, 4] = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ]
    return arr[0, 0] + arr[1, 2] + arr[2, 3]  # 1 + 7 + 12 = 20


@compile
def test_2d_array_row_access() -> i32:
    """Test 2D array row access"""
    arr: array[i32, 2, 3] = [[1, 2, 3], [4, 5, 6]]
    row: ptr[i32] = arr[1]
    return row[0] + row[1] + row[2]  # 15


@compile
def test_3d_array() -> i32:
    """Test 3D array"""
    arr: array[i32, 2, 3, 4] = [
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
        [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]
    ]
    return arr[0][0][0] + arr[1][2][3]  # 1 + 24 = 25


@compile
def test_3d_array_tuple_index() -> i32:
    """Test 3D array with tuple indexing"""
    arr: array[i32, 2, 3, 4] = [
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
        [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]
    ]
    return arr[0, 1, 2] + arr[1, 0, 3]  # 7 + 16 = 23


@compile
def test_4d_array() -> i32:
    """Test 4D array"""
    arr: array[i32, 2, 2, 2, 2] = [
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]
    ]
    return arr[0, 0, 0, 0] + arr[1, 1, 1, 1]  # 1 + 16 = 17


# =============================================================================
# Array Iteration Patterns
# =============================================================================

@compile
def test_array_sum() -> i32:
    """Test array sum with loop"""
    arr: array[i32, 10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sum: i32 = 0
    i: i32 = 0
    while i < 10:
        sum = sum + arr[i]
        i = i + 1
    return sum  # 55


@compile
def test_array_find_max() -> i32:
    """Test finding maximum in array"""
    arr: array[i32, 5] = [3, 7, 2, 9, 4]
    max_val: i32 = arr[0]
    i: i32 = 1
    while i < 5:
        if arr[i] > max_val:
            max_val = arr[i]
        i = i + 1
    return max_val  # 9


@compile
def test_array_find_min() -> i32:
    """Test finding minimum in array"""
    arr: array[i32, 5] = [3, 7, 2, 9, 4]
    min_val: i32 = arr[0]
    i: i32 = 1
    while i < 5:
        if arr[i] < min_val:
            min_val = arr[i]
        i = i + 1
    return min_val  # 2


@compile
def test_array_count() -> i32:
    """Test counting elements matching condition"""
    arr: array[i32, 10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    count: i32 = 0
    i: i32 = 0
    while i < 10:
        if arr[i] % 2 == 0:  # Count even numbers
            count = count + 1
        i = i + 1
    return count  # 5


@compile
def test_array_reverse_copy() -> i32:
    """Test reversing array into another"""
    src: array[i32, 5] = [1, 2, 3, 4, 5]
    dst: array[i32, 5] = array[i32, 5]()
    i: i32 = 0
    while i < 5:
        dst[4 - i] = src[i]
        i = i + 1
    return dst[0] + dst[1] + dst[2] + dst[3] + dst[4]  # 15


# =============================================================================
# Array with Pointers
# =============================================================================

@compile
def test_array_decay() -> i32:
    """Test array decay to pointer"""
    arr: array[i32, 5] = [10, 20, 30, 40, 50]
    p: ptr[i32] = arr
    return p[0] + p[2] + p[4]  # 90


@compile
def test_array_ptr_arithmetic() -> i32:
    """Test pointer arithmetic on array"""
    arr: array[i32, 5] = [10, 20, 30, 40, 50]
    p: ptr[i32] = arr
    p = p + 2
    return p[0] + p[1] + p[2]  # 30 + 40 + 50 = 120


@compile
def test_array_ptr_iterate() -> i32:
    """Test iterating array via pointer"""
    arr: array[i32, 5] = [1, 2, 3, 4, 5]
    p: ptr[i32] = arr
    end: ptr[i32] = arr + 5
    sum: i32 = 0
    while p < end:
        sum = sum + p[0]
        p = p + 1
    return sum  # 15


@compile
def test_ptr_to_array_element() -> i32:
    """Test pointer to specific array element"""
    arr: array[i32, 5] = [10, 20, 30, 40, 50]
    p: ptr[i32] = ptr(arr[2])
    p[0] = 100
    return arr[2]  # 100


@compile
def test_array_of_ptrs() -> i32:
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


# =============================================================================
# Array with Structs
# =============================================================================

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
def test_struct_with_array() -> i32:
    """Test struct containing array"""
    s: struct[i32, array[i32, 3]] = struct[i32, array[i32, 3]]()
    s[0] = 100
    s[1][0] = 1
    s[1][1] = 2
    s[1][2] = 3
    return s[0] + s[1][0] + s[1][1] + s[1][2]  # 106


@compile
def test_nested_array_struct() -> i32:
    """Test nested array and struct combination"""
    arr: array[struct[i32, array[i32, 2]], 2] = array[struct[i32, array[i32, 2]], 2]()
    arr[0][0] = 10
    arr[0][1][0] = 1
    arr[0][1][1] = 2
    arr[1][0] = 20
    arr[1][1][0] = 3
    arr[1][1][1] = 4
    
    return arr[0][0] + arr[0][1][0] + arr[0][1][1] + arr[1][0] + arr[1][1][0] + arr[1][1][1]  # 40


# =============================================================================
# Array Algorithms
# =============================================================================

@compile
def test_bubble_sort_partial() -> i32:
    """Test partial bubble sort (one pass)"""
    arr: array[i32, 5] = [5, 3, 4, 1, 2]
    # One pass of bubble sort
    i: i32 = 0
    while i < 4:
        if arr[i] > arr[i + 1]:
            temp: i32 = arr[i]
            arr[i] = arr[i + 1]
            arr[i + 1] = temp
        i = i + 1
    # After one pass, largest element is at end
    return arr[4]  # 5


@compile
def test_linear_search() -> i32:
    """Test linear search"""
    arr: array[i32, 5] = [10, 20, 30, 40, 50]
    target: i32 = 30
    found_idx: i32 = -1
    i: i32 = 0
    while i < 5:
        if arr[i] == target:
            found_idx = i
            break
        i = i + 1
    return found_idx  # 2


@compile
def test_binary_search() -> i32:
    """Test binary search on sorted array"""
    arr: array[i32, 8] = [10, 20, 30, 40, 50, 60, 70, 80]
    target: i32 = 50
    left: i32 = 0
    right: i32 = 7
    found_idx: i32 = -1
    
    while left <= right:
        mid: i32 = (left + right) / 2
        if arr[mid] == target:
            found_idx = mid
            break
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return found_idx  # 4


@compile
def test_array_copy() -> i32:
    """Test array copy"""
    src: array[i32, 5] = [1, 2, 3, 4, 5]
    dst: array[i32, 5] = array[i32, 5]()
    i: i32 = 0
    while i < 5:
        dst[i] = src[i]
        i = i + 1
    return dst[0] + dst[1] + dst[2] + dst[3] + dst[4]  # 15


@compile
def test_array_fill() -> i32:
    """Test array fill"""
    arr: array[i32, 5] = array[i32, 5]()
    i: i32 = 0
    while i < 5:
        arr[i] = 42
        i = i + 1
    return arr[0] + arr[2] + arr[4]  # 126


# =============================================================================
# 2D Array Operations
# =============================================================================

@compile
def test_2d_array_sum() -> i32:
    """Test 2D array sum"""
    arr: array[i32, 3, 4] = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ]
    sum: i32 = 0
    i: i32 = 0
    while i < 3:
        j: i32 = 0
        while j < 4:
            sum = sum + arr[i][j]
            j = j + 1
        i = i + 1
    return sum  # 78


@compile
def test_2d_array_row_sum() -> i32:
    """Test 2D array row sums"""
    arr: array[i32, 3, 4] = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ]
    row_sums: array[i32, 3] = array[i32, 3]()
    i: i32 = 0
    while i < 3:
        sum: i32 = 0
        j: i32 = 0
        while j < 4:
            sum = sum + arr[i][j]
            j = j + 1
        row_sums[i] = sum
        i = i + 1
    return row_sums[0] + row_sums[1] + row_sums[2]  # 10 + 26 + 42 = 78


@compile
def test_2d_array_col_sum() -> i32:
    """Test 2D array column sums"""
    arr: array[i32, 3, 4] = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ]
    col_sums: array[i32, 4] = array[i32, 4]()
    j: i32 = 0
    while j < 4:
        sum: i32 = 0
        i: i32 = 0
        while i < 3:
            sum = sum + arr[i][j]
            i = i + 1
        col_sums[j] = sum
        j = j + 1
    return col_sums[0] + col_sums[1] + col_sums[2] + col_sums[3]  # 15+18+21+24 = 78


@compile
def test_2d_array_diagonal() -> i32:
    """Test 2D array diagonal sum"""
    arr: array[i32, 3, 3] = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    diag_sum: i32 = 0
    i: i32 = 0
    while i < 3:
        diag_sum = diag_sum + arr[i][i]
        i = i + 1
    return diag_sum  # 1 + 5 + 9 = 15


@compile
def test_2d_array_transpose() -> i32:
    """Test 2D array transpose"""
    src: array[i32, 2, 3] = [[1, 2, 3], [4, 5, 6]]
    dst: array[i32, 3, 2] = array[i32, 3, 2]()
    
    i: i32 = 0
    while i < 2:
        j: i32 = 0
        while j < 3:
            dst[j][i] = src[i][j]
            j = j + 1
        i = i + 1
    
    return dst[0][0] + dst[0][1] + dst[1][0] + dst[1][1] + dst[2][0] + dst[2][1]  # 21


# =============================================================================
# Edge Cases
# =============================================================================

@compile
def test_array_single_element() -> i32:
    """Test single element array"""
    arr: array[i32, 1] = [42]
    return arr[0]


@compile
def test_array_large() -> i32:
    """Test larger array"""
    arr: array[i32, 100] = array[i32, 100]()
    i: i32 = 0
    while i < 100:
        arr[i] = i
        i = i + 1
    return arr[0] + arr[50] + arr[99]  # 0 + 50 + 99 = 149


@compile
def test_array_first_last() -> i32:
    """Test first and last element access"""
    arr: array[i32, 10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    return arr[0] + arr[9]  # 11


@compile
def test_array_negative_values() -> i32:
    """Test array with negative values"""
    arr: array[i32, 5] = [-5, -3, 0, 3, 5]
    sum: i32 = 0
    i: i32 = 0
    while i < 5:
        sum = sum + arr[i]
        i = i + 1
    return sum  # 0


@compile
def test_array_all_same() -> i32:
    """Test array with all same values"""
    arr: array[i32, 5] = [7, 7, 7, 7, 7]
    return arr[0] + arr[1] + arr[2] + arr[3] + arr[4]  # 35


@compile
def test_array_alternating() -> i32:
    """Test array with alternating pattern"""
    arr: array[i32, 6] = [1, -1, 1, -1, 1, -1]
    sum: i32 = 0
    i: i32 = 0
    while i < 6:
        sum = sum + arr[i]
        i = i + 1
    return sum  # 0


# =============================================================================
# Array in Control Flow
# =============================================================================

@compile
def test_array_in_if() -> i32:
    """Test array access in if condition"""
    arr: array[i32, 5] = [10, 20, 30, 40, 50]
    result: i32 = 0
    if arr[2] > 25:
        result = 1
    return result  # 1


@compile
def test_array_conditional_modify() -> i32:
    """Test conditional array modification"""
    arr: array[i32, 5] = [1, 2, 3, 4, 5]
    i: i32 = 0
    while i < 5:
        if arr[i] % 2 == 0:
            arr[i] = arr[i] * 10
        i = i + 1
    return arr[0] + arr[1] + arr[2] + arr[3] + arr[4]  # 1 + 20 + 3 + 40 + 5 = 69


@compile
def test_array_early_exit() -> i32:
    """Test early exit when finding element"""
    arr: array[i32, 10] = [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
    i: i32 = 0
    while i < 10:
        if arr[i] == 2:
            return i
        i = i + 1
    return -1  # 5


# =============================================================================
# Test Runner
# =============================================================================

class TestBasicArray(unittest.TestCase):
    def test_init_full(self):
        self.assertEqual(test_array_init_full(), 15)
    
    def test_init_partial(self):
        self.assertEqual(test_array_init_partial(), 6)
    
    def test_init_default(self):
        self.assertEqual(test_array_init_default(), 0)
    
    def test_modify(self):
        self.assertEqual(test_array_modify(), 100)
    
    def test_read_write(self):
        self.assertEqual(test_array_read_write(), 100)


class TestArrayTypes(unittest.TestCase):
    def test_i8(self):
        self.assertEqual(test_array_i8(), 90)
    
    def test_i16(self):
        self.assertEqual(test_array_i16(), 600)
    
    def test_i64(self):
        self.assertEqual(test_array_i64(), 6000000000)
    
    def test_f32(self):
        self.assertEqual(test_array_f32(), 7)
    
    def test_f64(self):
        self.assertEqual(test_array_f64(), 6)
    
    def test_bool(self):
        self.assertEqual(test_array_bool(), 2)


class TestMultiDimArray(unittest.TestCase):
    def test_2d_init(self):
        self.assertEqual(test_2d_array_init(), 21)
    
    def test_2d_modify(self):
        self.assertEqual(test_2d_array_modify(), 300)
    
    def test_2d_tuple_index(self):
        self.assertEqual(test_2d_array_tuple_index(), 20)
    
    def test_2d_row_access(self):
        self.assertEqual(test_2d_array_row_access(), 15)
    
    def test_3d(self):
        self.assertEqual(test_3d_array(), 25)
    
    def test_3d_tuple_index(self):
        self.assertEqual(test_3d_array_tuple_index(), 23)
    
    def test_4d(self):
        self.assertEqual(test_4d_array(), 17)


class TestArrayIteration(unittest.TestCase):
    def test_sum(self):
        self.assertEqual(test_array_sum(), 55)
    
    def test_find_max(self):
        self.assertEqual(test_array_find_max(), 9)
    
    def test_find_min(self):
        self.assertEqual(test_array_find_min(), 2)
    
    def test_count(self):
        self.assertEqual(test_array_count(), 5)
    
    def test_reverse_copy(self):
        self.assertEqual(test_array_reverse_copy(), 15)


class TestArrayPointer(unittest.TestCase):
    def test_decay(self):
        self.assertEqual(test_array_decay(), 90)
    
    def test_ptr_arithmetic(self):
        self.assertEqual(test_array_ptr_arithmetic(), 120)
    
    def test_ptr_iterate(self):
        self.assertEqual(test_array_ptr_iterate(), 15)
    
    def test_ptr_to_element(self):
        self.assertEqual(test_ptr_to_array_element(), 100)
    
    def test_array_of_ptrs(self):
        self.assertEqual(test_array_of_ptrs(), 60)


class TestArrayStruct(unittest.TestCase):
    def test_array_of_structs(self):
        self.assertEqual(test_array_of_structs(), 21)
    
    def test_struct_with_array(self):
        self.assertEqual(test_struct_with_array(), 106)
    
    def test_nested(self):
        self.assertEqual(test_nested_array_struct(), 40)


class TestArrayAlgorithms(unittest.TestCase):
    def test_bubble_partial(self):
        self.assertEqual(test_bubble_sort_partial(), 5)
    
    def test_linear_search(self):
        self.assertEqual(test_linear_search(), 2)
    
    def test_binary_search(self):
        self.assertEqual(test_binary_search(), 4)
    
    def test_copy(self):
        self.assertEqual(test_array_copy(), 15)
    
    def test_fill(self):
        self.assertEqual(test_array_fill(), 126)


class Test2DOperations(unittest.TestCase):
    def test_sum(self):
        self.assertEqual(test_2d_array_sum(), 78)
    
    def test_row_sum(self):
        self.assertEqual(test_2d_array_row_sum(), 78)
    
    def test_col_sum(self):
        self.assertEqual(test_2d_array_col_sum(), 78)
    
    def test_diagonal(self):
        self.assertEqual(test_2d_array_diagonal(), 15)
    
    def test_transpose(self):
        self.assertEqual(test_2d_array_transpose(), 21)


class TestEdgeCases(unittest.TestCase):
    def test_single(self):
        self.assertEqual(test_array_single_element(), 42)
    
    def test_large(self):
        self.assertEqual(test_array_large(), 149)
    
    def test_first_last(self):
        self.assertEqual(test_array_first_last(), 11)
    
    def test_negative(self):
        self.assertEqual(test_array_negative_values(), 0)
    
    def test_all_same(self):
        self.assertEqual(test_array_all_same(), 35)
    
    def test_alternating(self):
        self.assertEqual(test_array_alternating(), 0)


class TestArrayControlFlow(unittest.TestCase):
    def test_in_if(self):
        self.assertEqual(test_array_in_if(), 1)
    
    def test_conditional_modify(self):
        self.assertEqual(test_array_conditional_modify(), 69)
    
    def test_early_exit(self):
        self.assertEqual(test_array_early_exit(), 5)


if __name__ == '__main__':
    unittest.main()
