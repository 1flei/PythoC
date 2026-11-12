#!/usr/bin/env python3
"""
Test yield function with tuple unpacking

This test verifies that tuple unpacking in for loops over yield functions works.
"""

import sys
import os
from types import SimpleNamespace
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc import compile, f64, i32, struct
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

    def test_tuple_members_use_the_resolved_alias(self):
        self.assertEqual(tuple_alias_fields(), 948)


MixedPair = struct[i32, f64]
YieldTypes = SimpleNamespace(Pair=MixedPair)

def pair_type():
    return MixedPair

@compile
def pair_values() -> pair_type():
    yield (7, 1.25)
    yield (9, 1.75)

@compile
def qualified_pair_values() -> YieldTypes.Pair:
    yield (7, 1.25)
    yield (9, 1.75)

@compile
def alias_pair_values() -> MixedPair:
    yield (7, 1.25)
    yield (9, 1.75)

@compile
def tuple_alias_fields() -> i32:
    total: i32 = 0
    for integer, real in pair_values():
        total += integer + i32(real * 100.0)
    for integer, real in qualified_pair_values():
        total += integer + i32(real * 100.0)
    for integer, real in alias_pair_values():
        total += integer + i32(real * 100.0)
    return total


if __name__ == '__main__':
    unittest.main()
