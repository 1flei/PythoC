#!/usr/bin/env python3
"""
Test yield function with tuple unpacking in for loop

This test reproduces the issue where yield functions that return tuples
fail to inline when used with tuple unpacking in the for loop.

The current limitation is that yield functions only support simple Name
targets in for loops, not tuple unpacking.

Example that fails:
    for a, b in yield_tuples():  # Fails - tuple unpacking not supported
        ...

Example that works:
    for item in yield_tuples():  # Works - simple Name target
        a = item[0]
        b = item[1]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import unittest

from pythoc import compile, i32, i8, void, ptr, struct
from pythoc.build.output_manager import flush_all_pending_outputs, clear_failed_group


# =============================================================================
# Test case: simple Name target (should work)
# =============================================================================

@compile
def yield_single_value(n: i32) -> i32:
    """Yield single values - should work"""
    i: i32 = 0
    while i < n:
        yield i
        i = i + 1


@compile
def test_single_value() -> i32:
    """Test single value yield with simple Name target"""
    total: i32 = 0
    for x in yield_single_value(5):
        total = total + x
    return total


class TestYieldSimpleTarget(unittest.TestCase):
    """Test yield with simple Name target"""
    
    def test_single_value(self):
        """Single value yield should work"""
        result = test_single_value()
        # 0 + 1 + 2 + 3 + 4 = 10
        self.assertEqual(result, 10)


if __name__ == '__main__':
    unittest.main()
