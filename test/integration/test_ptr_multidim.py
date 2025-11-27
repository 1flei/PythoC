#!/usr/bin/env python3
"""
Test ptr multi-dimensional subscript support.

Tests:
- ptr[T, dims...] type definition where first dim is ignored
- ptr multi-dimensional value subscript: p[i, j, k]
- Integration with array decay
"""

from pythoc import i32, compile, seq, static, array, ptr
from pythoc.libc.stdio import printf


@compile
def test_ptr_type_syntax():
    """Test that ptr[T, dims...] type syntax works"""
    xs: static[array[i32, 3, 4]]
    
    # ptr[i32, 3, 4] should be equivalent to ptr[array[i32, 4]]
    p: ptr[i32, 3, 4] = xs
    
    printf("ptr type syntax test passed\n")


@compile
def test_ptr_2d_subscript():
    """Test 2D pointer subscript"""
    xs: static[array[i32, 3, 4]]
    
    # Initialize array
    for i in seq(3):
        for j in seq(4):
            xs[i, j] = i32(i * 10 + j)
    
    # Access via pointer with multi-dim subscript
    p: ptr[i32, 3, 4] = xs
    
    printf("p[0, 0] = %d (expect 0)\n", p[0, 0])
    printf("p[1, 2] = %d (expect 12)\n", p[1, 2])
    printf("p[2, 3] = %d (expect 23)\n", p[2, 3])


@compile  
def test_ptr_3d_subscript():
    """Test 3D pointer subscript"""
    xs: static[array[i32, 2, 3, 4]]
    
    xs[0, 1, 2] = i32(123)
    xs[1, 2, 3] = i32(456)
    
    p: ptr[i32, 2, 3, 4] = xs
    
    printf("p[0, 1, 2] = %d (expect 123)\n", p[0, 1, 2])
    printf("p[1, 2, 3] = %d (expect 456)\n", p[1, 2, 3])


@compile
def test_ptr_decay_equivalence():
    """Test that ptr[i32, N, M] is equivalent to ptr[array[i32, M]]"""
    xs: static[array[i32, 3, 4]]
    
    xs[1, 2] = i32(42)
    
    # Both types should work the same way
    p1: ptr[i32, 3, 4] = xs
    p2: ptr[array[i32, 4]] = xs
    
    printf("p1[1, 2] = %d\n", p1[1, 2])
    printf("p2[1, 2] = %d\n", p2[1, 2])


if __name__ == "__main__":
    test_ptr_type_syntax()
    test_ptr_2d_subscript()
    test_ptr_3d_subscript()
    test_ptr_decay_equivalence()
    print("\n=== All ptr multidim tests passed ===")
