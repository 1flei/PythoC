"""
Test mutual recursion support in pythoc.

This tests the two-pass compilation that allows functions to call each other
regardless of definition order.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pythoc import compile, i32, bool
from pythoc.builtin_entities.types import void


# Mutual recursion: is_even calls is_odd, is_odd calls is_even
# is_odd is defined AFTER is_even, testing forward reference support

@compile
def is_even(n: i32) -> bool:
    """Check if n is even using mutual recursion"""
    if n == 0:
        return True
    return is_odd(n - 1)


@compile
def is_odd(n: i32) -> bool:
    """Check if n is odd using mutual recursion"""
    if n == 0:
        return False
    return is_even(n - 1)


class TestMutualRecursion(unittest.TestCase):
    """Test mutual recursion between @compile functions"""
    
    def test_is_even(self):
        """Test is_even function"""
        self.assertEqual(is_even(0), True)
        self.assertEqual(is_even(1), False)
        self.assertEqual(is_even(2), True)
        self.assertEqual(is_even(3), False)
        self.assertEqual(is_even(4), True)
        self.assertEqual(is_even(10), True)
    
    def test_is_odd(self):
        """Test is_odd function"""
        self.assertEqual(is_odd(0), False)
        self.assertEqual(is_odd(1), True)
        self.assertEqual(is_odd(2), False)
        self.assertEqual(is_odd(3), True)
        self.assertEqual(is_odd(4), False)
        self.assertEqual(is_odd(11), True)


# More complex mutual recursion: three functions

@compile
def func_a(n: i32) -> i32:
    """First function in three-way mutual recursion"""
    if n <= 0:
        return 0
    return func_b(n - 1) + 1


@compile
def func_b(n: i32) -> i32:
    """Second function in three-way mutual recursion"""
    if n <= 0:
        return 0
    return func_c(n - 1) + 2


@compile
def func_c(n: i32) -> i32:
    """Third function in three-way mutual recursion"""
    if n <= 0:
        return 0
    return func_a(n - 1) + 3


class TestThreeWayRecursion(unittest.TestCase):
    """Test three-way mutual recursion"""
    
    def test_func_a(self):
        """Test func_a"""
        self.assertEqual(func_a(0), 0)
        self.assertEqual(func_a(1), 1)  # func_b(0) + 1 = 0 + 1 = 1
        self.assertEqual(func_a(2), 3)  # func_b(1) + 1 = (func_c(0) + 2) + 1 = 3
    
    def test_func_b(self):
        """Test func_b"""
        self.assertEqual(func_b(0), 0)
        self.assertEqual(func_b(1), 2)  # func_c(0) + 2 = 0 + 2 = 2
    
    def test_func_c(self):
        """Test func_c"""
        self.assertEqual(func_c(0), 0)
        self.assertEqual(func_c(1), 3)  # func_a(0) + 3 = 0 + 3 = 3


if __name__ == '__main__':
    unittest.main()
