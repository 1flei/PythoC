#!/usr/bin/env python3
"""
Test yield function with tuple unpacking

This test verifies that tuple unpacking in for loops over yield functions works.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import unittest

from pythoc import compile, i32, struct
from pythoc.build.output_manager import clear_failed_group


# =============================================================================
# Test case: tuple unpacking
# =============================================================================

@compile
def yield_tuple_values(n: i32) -> struct[i32, i32]:
    """Yield tuple values"""
    i: i32 = 0
    while i < n:
        yield i, i * 2
        i = i + 1


@compile
def test_tuple_unpacking() -> i32:
    """Test tuple yield with tuple unpacking"""
    total: i32 = 0
    for a, b in yield_tuple_values(3):
        total = total + a + b
    return total


class TestYieldTupleUnpacking(unittest.TestCase):
    """Test that tuple unpacking works correctly"""
    
    def test_tuple_unpacking_works(self):
        """Tuple unpacking should work"""
        result = test_tuple_unpacking()
        # (0 + 0) + (1 + 2) + (2 + 4) = 9
        self.assertEqual(result, 9)


if __name__ == '__main__':
    unittest.main()
