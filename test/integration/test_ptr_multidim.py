#!/usr/bin/env python3
"""
Test ptr multi-dimensional subscript support.

Tests:
- ptr[T, dims...] type definition where first dim is ignored
- ptr multi-dimensional value subscript: p[i, j, k]
- Integration with array decay
"""

import unittest
from pythoc import i32, compile, seq, static, array, ptr
from pythoc.libc.stdio import printf


@compile
def test_ptr_type_syntax() -> i32:
    """Test that ptr[T, dims...] type syntax works"""
    xs: static[array[i32, 3, 4]]
    
    # ptr[i32, 3, 4] should be equivalent to ptr[array[i32, 4]]
    p: ptr[i32, 3, 4] = xs
    
    return 0


@compile
def test_ptr_2d_subscript() -> i32:
    """Test 2D pointer subscript"""
    xs: static[array[i32, 3, 4]]
    
    # Initialize array
    for i in seq(3):
        for j in seq(4):
            xs[i, j] = i32(i * 10 + j)
    
    # Access via pointer with multi-dim subscript
    p: ptr[i32, 3, 4] = xs
    
    # p[0, 0] = 0, p[1, 2] = 12, p[2, 3] = 23
    return p[0, 0] + p[1, 2] + p[2, 3]  # 0 + 12 + 23 = 35


@compile  
def test_ptr_3d_subscript() -> i32:
    """Test 3D pointer subscript"""
    xs: static[array[i32, 2, 3, 4]]
    
    xs[0, 1, 2] = i32(123)
    xs[1, 2, 3] = i32(456)
    
    p: ptr[i32, 2, 3, 4] = xs
    
    return p[0, 1, 2] + p[1, 2, 3]  # 123 + 456 = 579


@compile
def test_ptr_decay_equivalence() -> i32:
    """Test that ptr[i32, N, M] is equivalent to ptr[array[i32, M]]"""
    xs: static[array[i32, 3, 4]]
    
    xs[1, 2] = i32(42)
    
    # Both types should work the same way
    p1: ptr[i32, 3, 4] = xs
    p2: ptr[array[i32, 4]] = xs
    
    return p1[1, 2] + p2[1, 2]  # 42 + 42 = 84


class TestPtrMultidim(unittest.TestCase):
    def test_type_syntax(self):
        self.assertEqual(test_ptr_type_syntax(), 0)
    
    def test_2d_subscript(self):
        self.assertEqual(test_ptr_2d_subscript(), 35)
    
    def test_3d_subscript(self):
        self.assertEqual(test_ptr_3d_subscript(), 579)
    
    def test_decay_equivalence(self):
        self.assertEqual(test_ptr_decay_equivalence(), 84)


if __name__ == "__main__":
    unittest.main()
