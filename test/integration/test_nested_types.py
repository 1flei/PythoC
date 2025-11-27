"""
Test nested type combinations and address correctness.

This test file focuses on:
- Complex combinations of ptr/array/struct types
- Verifying that field addresses are correctly computed
- Testing various levels of type nesting

All @compile functions are defined at module level.
"""

import unittest
from pythoc import compile, i32, i64, i8, ptr, struct, array


# ============================================================================
# Array of structs - basic operations
# ============================================================================

@compile
def test_array_of_structs_basic() -> i32:
    """Array containing struct elements"""
    points: array[struct[i32, i32], 3] = array[struct[i32, i32], 3]()
    points[0][0] = 10
    points[0][1] = 20
    points[1][0] = 30
    points[1][1] = 40
    points[2][0] = 50
    points[2][1] = 60
    return points[1][0] + points[2][1]  # 30 + 60 = 90


@compile
def test_array_of_structs_modify() -> i32:
    """Modify elements in array of structs"""
    points: array[struct[i32, i32], 2] = array[struct[i32, i32], 2]()
    points[0][0] = 1
    points[0][1] = 2
    points[1][0] = 3
    points[1][1] = 4
    # Modify values
    points[0][0] = 100
    points[1][1] = 200
    return points[0][0] + points[1][1]  # 100 + 200 = 300


@compile
def test_array_of_structs_addresses() -> i32:
    """Verify we can get addresses of struct fields in array"""
    points: array[struct[i32, i32], 3] = array[struct[i32, i32], 3]()
    points[0][0] = 1
    points[1][0] = 3
    points[2][0] = 5
    
    # Get addresses and verify access
    addr0: ptr[i32] = ptr(points[0][0])
    addr1: ptr[i32] = ptr(points[1][0])
    addr2: ptr[i32] = ptr(points[2][0])
    
    if addr0[0] == 1 and addr1[0] == 3 and addr2[0] == 5:
        return 1
    return 0


# ============================================================================
# Struct containing arrays
# ============================================================================
# TODO: These tests require struct[..., array[...], ...] initialization support
# See test_nested_types_TODO.md for details

@compile
def test_struct_with_array() -> i32:
    """Struct containing an array field"""
    data: struct[i32, array[i32, 3]] = struct[i32, array[i32, 3]]()
    data[0] = 100
    data[1][0] = 10
    data[1][1] = 20
    data[1][2] = 30
    return data[0] + data[1][0] + data[1][1] + data[1][2]  # 100 + 10 + 20 + 30 = 160


@compile
def test_struct_with_multiple_arrays() -> i32:
    """Struct with multiple array fields"""
    data: struct[array[i32, 2], array[i32, 3]] = struct[array[i32, 2], array[i32, 3]]()
    data[0][0] = 10
    data[0][1] = 20
    data[1][0] = 30
    data[1][1] = 40
    data[1][2] = 50
    return data[0][0] + data[0][1] + data[1][0] + data[1][1] + data[1][2]  # 10+20+30+40+50 = 150


# ============================================================================
# Struct containing pointers
# ============================================================================

@compile
def test_struct_with_pointer() -> i32:
    """Struct containing pointer field"""
    value: i32 = 42
    data: struct[ptr[i32], i32]
    data[0] = ptr(value)
    data[1] = 10
    p: ptr[i32] = data[0]
    return p[0] + data[1]  # 42 + 10 = 52


@compile
def test_modify_through_struct_pointer() -> i32:
    """Modify value through pointer stored in struct"""
    value: i32 = 5
    data: struct[ptr[i32], i32] = struct[ptr[i32], i32]()
    data[0] = ptr(value)
    data[1] = 100
    p: ptr[i32] = data[0]
    p[0] = 99
    return value  # Should be 99


@compile
def test_struct_with_multiple_pointers() -> i32:
    """Struct with multiple pointer fields"""
    x: i32 = 10
    y: i32 = 20
    data: struct[ptr[i32], ptr[i32]] = struct[ptr[i32], ptr[i32]]()
    data[0] = ptr(x)
    data[1] = ptr(y)
    p1: ptr[i32] = data[0]
    p2: ptr[i32] = data[1]
    return p1[0] + p2[0]  # 10 + 20 = 30


# ============================================================================
# Nested structs
# ============================================================================

@compile
def test_nested_struct_two_levels() -> i32:
    """Two levels of struct nesting"""
    inner: struct[i32, i32] = (10, 20)
    outer: struct[i32, struct[i32, i32]] = struct[i32, struct[i32, i32]]()
    outer[0] = 5
    outer[1] = inner
    nested: struct[i32, i32] = outer[1]
    return outer[0] + nested[0] + nested[1]  # 5 + 10 + 20 = 35


@compile
def test_nested_struct_modify() -> i32:
    """Modify nested struct fields"""
    inner: struct[i32, i32] = (10, 20)
    outer: struct[struct[i32, i32], i32] = struct[struct[i32, i32], i32]()
    outer[0] = inner
    outer[1] = 30
    
    nested: struct[i32, i32] = outer[0]
    nested[0] = 100
    
    return nested[0] + nested[1] + outer[1]  # 100 + 20 + 30 = 150


@compile
def test_three_level_struct_nesting() -> i32:
    """Three levels of struct nesting"""
    level1: struct[i32, i32] = (1, 2)
    level2: struct[i32, struct[i32, i32]] = struct[i32, struct[i32, i32]]()
    level2[0] = 3
    level2[1] = level1
    
    level3: struct[i32, struct[i32, struct[i32, i32]]] = struct[i32, struct[i32, struct[i32, i32]]]()
    level3[0] = 4
    level3[1] = level2
    
    l2: struct[i32, struct[i32, i32]] = level3[1]
    l1: struct[i32, i32] = l2[1]
    
    return level3[0] + l2[0] + l1[0] + l1[1]  # 4 + 3 + 1 + 2 = 10


# ============================================================================
# Pointer to array
# ============================================================================

@compile
def test_pointer_to_array_access() -> i32:
    """Access array through pointer"""
    arr: array[i32, 5] = [10, 20, 30, 40, 50]
    p: ptr[i32] = ptr(arr[0])
    return p[0] + p[2] + p[4]  # 10 + 30 + 50 = 90


@compile
def test_pointer_to_array_modify() -> i32:
    """Modify array through pointer"""
    arr: array[i32, 3] = [1, 2, 3]
    p: ptr[i32] = ptr(arr[0])
    p[1] = 99
    return arr[0] + arr[1] + arr[2]  # 1 + 99 + 3 = 103


# ============================================================================
# Pointer to struct
# ============================================================================

@compile
def test_pointer_to_struct_field() -> i32:
    """Get pointer to specific struct field"""
    point: struct[i32, i32, i32] = (10, 20, 30)
    p_field: ptr[i32] = ptr(point[1])
    p_field[0] = 99
    return point[0] + point[1] + point[2]  # 10 + 99 + 30 = 139


@compile
def test_pointer_to_struct_access() -> i32:
    """Access struct through pointer"""
    point: struct[i32, i32] = (10, 20)
    p: ptr[struct[i32, i32]] = ptr(point)
    s: struct[i32, i32] = p[0]
    return s[0] + s[1]  # 10 + 20 = 30


# ============================================================================
# Array of pointers
# ============================================================================

@compile
def test_array_of_pointers() -> i32:
    """Array containing pointers"""
    x: i32 = 10
    y: i32 = 20
    z: i32 = 30
    
    ptrs: array[ptr[i32], 3] = array[ptr[i32], 3]()
    ptrs[0] = ptr(x)
    ptrs[1] = ptr(y)
    ptrs[2] = ptr(z)
    
    p0: ptr[i32] = ptrs[0]
    p1: ptr[i32] = ptrs[1]
    p2: ptr[i32] = ptrs[2]
    
    return p0[0] + p1[0] + p2[0]  # 10 + 20 + 30 = 60


@compile
def test_array_of_pointers_modify() -> i32:
    """Modify values through array of pointers"""
    a: i32 = 1
    b: i32 = 2
    c: i32 = 3
    
    ptrs: array[ptr[i32], 3] = array[ptr[i32], 3]()
    ptrs[0] = ptr(a)
    ptrs[1] = ptr(b)
    ptrs[2] = ptr(c)
    
    p: ptr[i32] = ptrs[1]
    p[0] = 99
    
    return a + b + c  # 1 + 99 + 3 = 103


# ============================================================================
# Complex nesting: array of struct with array
# ============================================================================

@compile
def test_array_of_struct_with_array() -> i32:
    """Array of structs, where each struct contains an array"""
    data: array[struct[i32, array[i32, 2]], 2] = array[struct[i32, array[i32, 2]], 2]()
    
    # Initialize through direct indexing
    data[0][0] = 10
    data[0][1][0] = 1
    data[0][1][1] = 2
    
    data[1][0] = 20
    data[1][1][0] = 3
    data[1][1][1] = 4
    
    return data[0][0] + data[0][1][0] + data[0][1][1] + data[1][0] + data[1][1][0] + data[1][1][1]  # 10+1+2+20+3+4 = 40


@compile
def test_struct_with_array_of_pointers() -> i32:
    """Struct containing an array of pointers"""
    x: i32 = 5
    y: i32 = 10
    
    data: struct[i32, array[ptr[i32], 2]] = struct[i32, array[ptr[i32], 2]]()
    data[0] = 100
    data[1][0] = ptr(x)
    data[1][1] = ptr(y)
    
    p0: ptr[i32] = data[1][0]
    p1: ptr[i32] = data[1][1]
    
    return data[0] + p0[0] + p1[0]  # 100 + 5 + 10 = 115


# ============================================================================
# Address verification
# ============================================================================

@compile
def test_struct_field_addresses() -> i32:
    """Verify that we can get addresses of struct fields"""
    data: struct[i32, i32, i32] = (10, 20, 30)
    
    p0: ptr[i32] = ptr(data[0])
    p1: ptr[i32] = ptr(data[1])
    p2: ptr[i32] = ptr(data[2])
    
    if p0[0] == 10 and p1[0] == 20 and p2[0] == 30:
        return 1
    return 0


@compile
def test_array_element_addresses() -> i32:
    """Verify that we can get addresses of array elements"""
    arr: array[i32, 4] = [10, 20, 30, 40]
    
    p0: ptr[i32] = ptr(arr[0])
    p1: ptr[i32] = ptr(arr[1])
    p2: ptr[i32] = ptr(arr[2])
    p3: ptr[i32] = ptr(arr[3])
    
    if p0[0] == 10 and p1[0] == 20 and p2[0] == 30 and p3[0] == 40:
        return 1
    return 0


@compile
def test_nested_struct_addresses() -> i32:
    """Verify addresses in nested structs"""
    inner: struct[i32, i32] = (10, 20)
    outer: struct[i32, struct[i32, i32], i32] = struct[i32, struct[i32, i32], i32]()
    outer[0] = 5
    outer[1] = inner
    outer[2] = 30
    
    p_outer_0: ptr[i32] = ptr(outer[0])
    nested: struct[i32, i32] = outer[1]
    p_nested_0: ptr[i32] = ptr(nested[0])
    p_outer_2: ptr[i32] = ptr(outer[2])
    
    if p_outer_0[0] == 5 and p_nested_0[0] == 10 and p_outer_2[0] == 30:
        return 1
    return 0


# ============================================================================
# Mixed type sizes
# ============================================================================

@compile
def test_struct_different_sizes() -> i32:
    """Struct with fields of different sizes"""
    data: struct[i8, i32, i64] = struct[i8, i32, i64]()
    data[0] = 10
    data[1] = 20
    data[2] = 30
    
    v0: i8 = data[0]
    v1: i32 = data[1]
    v2: i64 = data[2]
    
    result: i32 = v0 + v1 + v2
    return result  # 10 + 20 + 30 = 60


@compile
def test_array_of_mixed_size_structs() -> i32:
    """Array of structs with different sized fields"""
    arr: array[struct[i8, i32], 2] = array[struct[i8, i32], 2]()
    arr[0][0] = 1
    arr[0][1] = 10
    arr[1][0] = 2
    arr[1][1] = 20
    
    return arr[0][0] + arr[0][1] + arr[1][0] + arr[1][1]  # 1 + 10 + 2 + 20 = 33


# ============================================================================
# Pointer chains
# ============================================================================

@compile
def test_pointer_to_pointer() -> i32:
    """Pointer to pointer (simulated with array)"""
    value: i32 = 42
    p1: ptr[i32] = ptr(value)
    
    ptr_storage: array[ptr[i32], 1] = array[ptr[i32], 1]()
    ptr_storage[0] = p1
    p2: ptr[ptr[i32]] = ptr(ptr_storage[0])
    
    intermediate: ptr[i32] = p2[0]
    result: i32 = intermediate[0]
    
    return result  # 42


@compile
def test_modify_through_pointer_chain() -> i32:
    """Modify value through pointer chain"""
    value: i32 = 10
    p1: ptr[i32] = ptr(value)
    
    ptr_storage: array[ptr[i32], 1] = array[ptr[i32], 1]()
    ptr_storage[0] = p1
    p2: ptr[ptr[i32]] = ptr(ptr_storage[0])
    
    intermediate: ptr[i32] = p2[0]
    intermediate[0] = 99
    
    return value  # 99


# ============================================================================
# Test Classes
# ============================================================================

class TestArrayOfStructs(unittest.TestCase):
    """Test arrays containing structs"""
    
    def test_basic(self):
        self.assertEqual(test_array_of_structs_basic(), 90)
    
    def test_modify(self):
        self.assertEqual(test_array_of_structs_modify(), 300)
    
    def test_addresses(self):
        self.assertEqual(test_array_of_structs_addresses(), 1)


class TestStructWithArray(unittest.TestCase):
    """Test structs containing arrays"""
    
    def test_basic(self):
        self.assertEqual(test_struct_with_array(), 160)
    
    def test_multiple_arrays(self):
        self.assertEqual(test_struct_with_multiple_arrays(), 150)


class TestStructWithPointer(unittest.TestCase):
    """Test structs containing pointers"""
    
    def test_basic(self):
        self.assertEqual(test_struct_with_pointer(), 52)
    
    def test_modify(self):
        self.assertEqual(test_modify_through_struct_pointer(), 99)
    
    def test_multiple_pointers(self):
        self.assertEqual(test_struct_with_multiple_pointers(), 30)


class TestNestedStructs(unittest.TestCase):
    """Test nested struct structures"""
    
    def test_two_levels(self):
        self.assertEqual(test_nested_struct_two_levels(), 35)
    
    def test_modify(self):
        self.assertEqual(test_nested_struct_modify(), 150)
    
    def test_three_levels(self):
        self.assertEqual(test_three_level_struct_nesting(), 10)


class TestPointerToArray(unittest.TestCase):
    """Test pointers to arrays"""
    
    def test_access(self):
        self.assertEqual(test_pointer_to_array_access(), 90)
    
    def test_modify(self):
        self.assertEqual(test_pointer_to_array_modify(), 103)


class TestPointerToStruct(unittest.TestCase):
    """Test pointers to structs"""
    
    def test_field(self):
        self.assertEqual(test_pointer_to_struct_field(), 139)
    
    def test_access(self):
        self.assertEqual(test_pointer_to_struct_access(), 30)


class TestArrayOfPointers(unittest.TestCase):
    """Test arrays of pointers"""
    
    def test_basic(self):
        self.assertEqual(test_array_of_pointers(), 60)
    
    def test_modify(self):
        self.assertEqual(test_array_of_pointers_modify(), 103)


class TestComplexNesting(unittest.TestCase):
    """Test complex nested type combinations"""
    
    def test_array_of_struct_with_array(self):
        self.assertEqual(test_array_of_struct_with_array(), 40)
    
    def test_struct_with_array_of_pointers(self):
        self.assertEqual(test_struct_with_array_of_pointers(), 115)


class TestAddressVerification(unittest.TestCase):
    """Test that addresses are correctly calculated"""
    
    def test_struct_fields(self):
        self.assertEqual(test_struct_field_addresses(), 1)
    
    def test_array_elements(self):
        self.assertEqual(test_array_element_addresses(), 1)
    
    def test_nested_structs(self):
        self.assertEqual(test_nested_struct_addresses(), 1)


class TestMixedTypeSizes(unittest.TestCase):
    """Test structs with different sized fields"""
    
    def test_basic(self):
        self.assertEqual(test_struct_different_sizes(), 60)
    
    def test_in_array(self):
        self.assertEqual(test_array_of_mixed_size_structs(), 33)


class TestPointerChains(unittest.TestCase):
    """Test pointer to pointer scenarios"""
    
    def test_basic(self):
        self.assertEqual(test_pointer_to_pointer(), 42)
    
    def test_modify(self):
        self.assertEqual(test_modify_through_pointer_chain(), 99)


if __name__ == '__main__':
    unittest.main()
