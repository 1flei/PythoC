#!/usr/bin/env python3
"""
Test ptr functionality in compiled PC functions
"""

from pythoc import i8, i16, i32, i64, f64, ptr, compile, ptr
from pythoc.libc.stdio import printf

@compile
class TestStruct:
    a: i32
    b: i64
    c: i8

@compile
def test_ptr_basic() -> i32:
    """Test ptr with basic variables"""
    x: i32 = 42
    y: i64 = 100
    
    # Get pointers to variables
    ptr_x = ptr(x)
    ptr_y = ptr(y)
    
    # Dereference pointers to verify they point to correct values
    val_x = ptr_x[0]  # Should be 42
    val_y = ptr_y[0]  # Should be 100
    
    return val_x + i32(val_y)  # Should return 142

@compile
def test_ptr_struct() -> i32:
    """Test ptr with struct fields"""
    s: TestStruct = TestStruct()
    s.a = 10
    s.b = 20
    s.c = 5
    
    # Get pointers to struct fields
    ptr_a = ptr(s.a)
    ptr_b = ptr(s.b)
    ptr_c = ptr(s.c)
    
    # Modify values through pointers
    ptr_a[0] = 30
    ptr_b[0] = 40
    ptr_c[0] = 7
    
    # Return sum to verify modifications
    return s.a + i32(s.b) + i32(s.c)  # Should return 77

@compile
def test_ptr_array() -> i32:
    """Test ptr with array elements"""
    # Create an array using multiple variables (simulating array)
    arr0: i32 = 1
    arr1: i32 = 2
    arr2: i32 = 3
    
    # Get pointers to array elements
    ptr0 = ptr(arr0)
    ptr1 = ptr(arr1)
    ptr2 = ptr(arr2)
    
    # Modify through pointers
    ptr0[0] = 10
    ptr1[0] = 20
    ptr2[0] = 30
    
    return arr0 + arr1 + arr2  # Should return 60

@compile
def test_ptr_pointer_arithmetic() -> i32:
    """Test basic pointer operations"""
    x: i32 = 123
    y: i32 = 456
    
    ptr_x = ptr(x)
    ptr_y = ptr(y)
    
    # Test dereferencing
    val1 = ptr_x[0]
    val2 = ptr_y[0]
    
    # Modify through pointer
    ptr_x[0] = 999
    
    return x + val2  # Should return 999 + 456 = 1455

@compile
def main() -> i32:
    """Main function that calls all tests"""
    printf("Starting ptr tests...\n")
    
    result1 = test_ptr_basic()
    printf("test_ptr_basic: %d\n", result1)
    
    result2 = test_ptr_struct()
    printf("test_ptr_struct: %d\n", result2)
    
    result3 = test_ptr_array()
    printf("test_ptr_array: %d\n", result3)
    
    result4 = test_ptr_pointer_arithmetic()
    printf("test_ptr_pointer_arithmetic: %d\n", result4)
    
    return 0

main()
