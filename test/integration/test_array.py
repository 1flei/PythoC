#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
array syntax design proposals for pc library
Multiple candidate syntaxes for array support
"""

from pythoc import i32, i64, f64, bool, ptr, compile, nullptr, array
from pythoc.libc.stdio import printf

@compile
def test_simple() -> i32:
    # Declaration and initialization
    arr: array[i32, 5] = [1, 2, 3, 4, 5]

@compile
def test_array() -> i32:
    # Declaration and initialization
    arr: array[i32, 5] = [1, 2, 3, 4, 5]

    # Partial initialization (use different variable name or remove annotation)
    arr2: array[i32, 5] = [1, 2, 3]  # Remaining elements are 0
    
    # Or zero-initialized
    zeros: array[f64, 10] = array[f64, 10]()

    # arr = [1, 2, 3, 4, 5]  # Type inference, arr of first element type (TODO: not yet supported)
    
    # Access
    x: i32 = arr[0]
    arr[1] = 42
    
    # Implicit decay to pointer
    p: ptr[i32] = arr

    # Multi-dimensional array
    matrix: array[i32, 2, 3] = [[1, 2, 3], [4, 5, 6]]

    # access multi-dimensional array
    x00: i32 = matrix[0][0]
    x12: i32 = matrix[1, 2]       # Tuple indexing supported!
    # x12_alt: i32 = matrix[1][2]  # Also works
    x1: ptr[i32] = matrix[1]    # Pointer to row
    x2 = matrix[1]              # Implicit decay to pointer (x2 == x1)

    pm: ptr[array[i32, 3]] = matrix

    printf("arr[0] = %d, arr[1] = %d, arr2[2] = %d\n", arr[0], arr[1], arr2[2])
    printf("matrix[0][0] = %d, matrix[1][2] = %d\n", x00, x12)
    printf("p = %p, x1 = %p, x2 = %p\n", p, x1, x2)
    return 0

@compile
def test_1d_array() -> i32:
    """Test 1D array declaration, initialization, and access"""
    # Declaration with full initialization
    arr: array[i32, 5] = [1, 2, 3, 4, 5]
    
    # Access elements
    x: i32 = arr[0]
    y: i32 = arr[4]
    
    # Modify element
    arr[2] = 99
    
    printf("1D array: arr[0]=%d, arr[2]=%d, arr[4]=%d\n", x, arr[2], y)
    return 0

@compile
def test_partial_init() -> i32:
    """Test partial initialization (remaining elements should be zero)"""
    arr: array[i32, 5] = [10, 20, 30]
    
    printf("Partial init: arr[0]=%d, arr[2]=%d, arr[3]=%d, arr[4]=%d\n", 
           arr[0], arr[2], arr[3], arr[4])
    return 0

@compile
def test_zero_init() -> i32:
    """Test zero initialization"""
    zeros: array[f64, 3] = array[f64, 3]()
    
    printf("Zero init: zeros[0]=%f, zeros[1]=%f, zeros[2]=%f\n",
           zeros[0], zeros[1], zeros[2])
    return 0

@compile
def test_array_to_pointer() -> i32:
    """Test implicit array to pointer decay"""
    arr: array[i32, 3] = [100, 200, 300]
    
    # Implicit decay to pointer
    p: ptr[i32] = arr
    
    # Access through pointer (pointer arithmetic)
    printf("Array to pointer: p[0]=%d, p[1]=%d, p[2]=%d\n",
           p[0], p[1], p[2])
    return 0

@compile
def test_2d_array() -> i32:
    """Test 2D array"""
    matrix: array[i32, 2, 3] = [[1, 2, 3], [4, 5, 6]]
    
    # Access elements
    a: i32 = matrix[0][0]
    b: i32 = matrix[0][2]
    c: i32 = matrix[1][0]
    d: i32 = matrix[1][2]
    
    # Modify element
    matrix[1][1] = 99
    
    printf("2D array: [0][0]=%d, [0][2]=%d, [1][0]=%d, [1][1]=%d, [1][2]=%d\n",
           a, b, c, matrix[1][1], d)
    
    # Get pointer to row
    row1: ptr[i32] = matrix[1]
    printf("Row pointer: row1[0]=%d, row1[1]=%d, row1[2]=%d\n",
           row1[0], row1[1], row1[2])
    
    return 0

@compile
def test_3d_array() -> i32:
    """Test 3D array"""
    # Create a 2x3x4 array
    cube: array[i32, 2, 3, 4] = [
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
        [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]
    ]
    
    # Access elements
    a: i32 = cube[0][0][0]
    b: i32 = cube[0][1][2]
    c: i32 = cube[1][2][3]
    
    # Modify element
    cube[1][1][1] = 99
    
    printf("3D array: [0][0][0]=%d, [0][1][2]=%d, [1][2][3]=%d, [1][1][1]=%d\n",
           a, b, c, cube[1][1][1])
    
    # Get pointer to 2D slice
    slice1: ptr[i32] = cube[1][1]
    printf("Slice pointer: slice1[0]=%d, slice1[1]=%d, slice1[2]=%d, slice1[3]=%d\n",
           slice1[0], slice1[1], slice1[2], slice1[3])
    
    return 0

@compile
def array_sum() -> i32:
    """Calculate sum of array elements"""
    arr: array[i32, 5] = [10, 20, 30, 40, 50]
    
    sum: i32 = 0
    i: i32 = 0
    while i < 5:
        sum = sum + arr[i]
        i = i + 1
    
    printf("Sum of array: %d\n", sum)
    return sum

@compile
def matrix_multiply() -> i32:
    """Simple 2x2 matrix multiplication"""
    # A = [[1, 2], [3, 4]]
    a: array[i32, 2, 2] = [[1, 2], [3, 4]]
    
    # B = [[5, 6], [7, 8]]
    b: array[i32, 2, 2] = [[5, 6], [7, 8]]
    
    # C = A * B
    c: array[i32, 2, 2] = array[i32, 2, 2]()
    
    # C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0]
    c[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0]
    # C[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1]
    c[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1]
    # C[1][0] = A[1][0]*B[0][0] + A[1][1]*B[1][0]
    c[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0]
    # C[1][1] = A[1][0]*B[0][1] + A[1][1]*B[1][1]
    c[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1]
    
    printf("Matrix multiplication result:\n")
    printf("  [%d, %d]\n", c[0][0], c[0][1])
    printf("  [%d, %d]\n", c[1][0], c[1][1])
    
    return 0

@compile
def array_pointer_demo() -> i32:
    """Demonstrate array to pointer decay"""
    arr: array[i32, 4] = [100, 200, 300, 400]
    
    # Array decays to pointer
    p: ptr[i32] = arr
    
    printf("Array elements via pointer:\n")
    printf("  p[0] = %d\n", p[0])
    printf("  p[1] = %d\n", p[1])
    printf("  p[2] = %d\n", p[2])
    printf("  p[3] = %d\n", p[3])
    
    return 0

@compile
def test_tuple_index_2d() -> i32:
    """Test tuple indexing for 2D arrays (both rvalue and lvalue)"""
    matrix: array[i32, 3, 4] = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ]
    
    # Test tuple index as rvalue (read)
    v1: i32 = matrix[0, 0]  # 1
    v2: i32 = matrix[1, 2]  # 7
    v3: i32 = matrix[2, 3]  # 12
    
    printf("Tuple index 2D rvalue: matrix[0,0]=%d, matrix[1,2]=%d, matrix[2,3]=%d\n",
           v1, v2, v3)
    
    # Test tuple index as lvalue (write)
    matrix[0, 1] = 99
    matrix[1, 3] = 88
    matrix[2, 0] = 77
    
    printf("Tuple index 2D lvalue: matrix[0,1]=%d, matrix[1,3]=%d, matrix[2,0]=%d\n",
           matrix[0, 1], matrix[1, 3], matrix[2, 0])
    
    # Compare with traditional indexing
    traditional: i32 = matrix[1][2]
    tuple_style: i32 = matrix[1, 2]
    printf("Traditional matrix[1][2]=%d, Tuple matrix[1,2]=%d\n",
           traditional, tuple_style)
    
    return 0

@compile
def test_tuple_index_3d() -> i32:
    """Test tuple indexing for 3D arrays"""
    cube: array[i32, 2, 3, 4] = [
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
        [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]
    ]
    
    # Test tuple index as rvalue
    a: i32 = cube[0, 0, 0]  # 1
    b: i32 = cube[0, 1, 2]  # 7
    c: i32 = cube[1, 2, 3]  # 24
    d: i32 = cube[1, 1, 1]  # 18
    
    printf("Tuple index 3D rvalue: [0,0,0]=%d, [0,1,2]=%d, [1,2,3]=%d, [1,1,1]=%d\n",
           a, b, c, d)
    
    # Test tuple index as lvalue
    cube[0, 0, 1] = 100
    cube[1, 1, 1] = 200
    cube[1, 2, 3] = 300
    
    printf("Tuple index 3D lvalue: [0,0,1]=%d, [1,1,1]=%d, [1,2,3]=%d\n",
           cube[0, 0, 1], cube[1, 1, 1], cube[1, 2, 3])
    
    # Verify equivalence
    trad1: i32 = cube[0][0][1]
    tupl1: i32 = cube[0, 0, 1]
    printf("Equivalence check: cube[0][0][1]=%d, cube[0,0,1]=%d\n",
           trad1, tupl1)
    
    return 0

@compile
def test_tuple_index_4d() -> i32:
    """Test tuple indexing for 4D arrays"""
    hyper: array[i32, 2, 2, 2, 2] = [
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]
    ]
    
    # Test tuple index as rvalue
    v1: i32 = hyper[0, 0, 0, 0]  # 1
    v2: i32 = hyper[0, 1, 1, 0]  # 7
    v3: i32 = hyper[1, 1, 1, 1]  # 16
    
    printf("Tuple index 4D rvalue: [0,0,0,0]=%d, [0,1,1,0]=%d, [1,1,1,1]=%d\n",
           v1, v2, v3)
    
    # Test tuple index as lvalue
    hyper[0, 0, 0, 1] = 99
    hyper[1, 0, 1, 0] = 88
    hyper[1, 1, 1, 1] = 77
    
    printf("Tuple index 4D lvalue: [0,0,0,1]=%d, [1,0,1,0]=%d, [1,1,1,1]=%d\n",
           hyper[0, 0, 0, 1], hyper[1, 0, 1, 0], hyper[1, 1, 1, 1])
    
    return 0

@compile
def test_tuple_index_mixed() -> i32:
    """Test mixed indexing styles"""
    matrix: array[i32, 3, 4] = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ]
    
    # Mix tuple index with traditional index
    v1: i32 = matrix[1, 2]      # Tuple style
    v2: i32 = matrix[1][2]      # Traditional style
    
    printf("Mixed indexing: tuple matrix[1,2]=%d, traditional matrix[1][2]=%d\n",
           v1, v2)
    
    # Modify using both styles
    matrix[0, 0] = 100
    matrix[2][3] = 200
    
    printf("After mixed modification: matrix[0,0]=%d, matrix[2,3]=%d\n",
           matrix[0, 0], matrix[2, 3])
    
    return 0

@compile
def test_tuple_index_arithmetic() -> i32:
    """Test tuple indexing in arithmetic expressions"""
    matrix: array[i32, 2, 3] = [[1, 2, 3], [4, 5, 6]]
    
    # Use tuple indexing in expressions
    sum: i32 = matrix[0, 0] + matrix[0, 1] + matrix[0, 2]
    prod: i32 = matrix[1, 0] * matrix[1, 1]
    
    printf("Tuple index arithmetic: sum=%d, prod=%d\n", sum, prod)
    
    # Complex expression
    result: i32 = (matrix[0, 0] + matrix[1, 1]) * matrix[1, 2]
    printf("Complex expression: (matrix[0,0] + matrix[1,1]) * matrix[1,2] = %d\n",
           result)
    
    return 0

@compile
def test_tuple_index_assignment() -> i32:
    """Test tuple indexing assignment with expressions"""
    matrix: array[i32, 3, 3] = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    
    # Initialize using tuple indexing
    for i in range(3):
        for j in range(3):
            matrix[i, j] = i * 3 + j + 1
    
    # Assignment with expression on right side
    matrix[0, 0] = matrix[0, 1] + matrix[0, 2]  # 2 + 3 = 5
    matrix[1, 1] = matrix[1, 0] * 2              # 4 * 2 = 8
    matrix[2, 2] = matrix[2, 0] + matrix[2, 1]  # 7 + 8 = 15
    
    printf("Tuple index assignment: [0,0]=%d, [1,1]=%d, [2,2]=%d\n",
           matrix[0, 0], matrix[1, 1], matrix[2, 2])
    
    return 0

test_array()
test_1d_array()
test_partial_init()
test_zero_init()
test_array_to_pointer()
test_2d_array()
test_3d_array()

# Run demos
printf("=== Array Sum Demo ===\n")
array_sum()

printf("\n=== Matrix Multiply Demo ===\n")
matrix_multiply()

printf("\n=== Array Pointer Demo ===\n")
array_pointer_demo()

# Test tuple indexing feature
printf("\n=== Tuple Index 2D Test ===\n")
test_tuple_index_2d()

printf("\n=== Tuple Index 3D Test ===\n")
test_tuple_index_3d()

printf("\n=== Tuple Index 4D Test ===\n")
test_tuple_index_4d()

printf("\n=== Tuple Index Mixed Test ===\n")
test_tuple_index_mixed()

printf("\n=== Tuple Index Arithmetic Test ===\n")
test_tuple_index_arithmetic()

printf("\n=== Tuple Index Assignment Test ===\n")
test_tuple_index_assignment()
