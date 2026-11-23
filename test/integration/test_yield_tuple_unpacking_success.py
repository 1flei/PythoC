#!/usr/bin/env python3
"""
Test yield function with tuple unpacking

This test verifies that tuple unpacking in for loops over yield functions works.
"""

import sys
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc import array, compile, f64, i32, struct
from pythoc.build.output_manager import clear_failed_group
from pythoc.logger import logger


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

    def test_nested_targets_survive_continue_and_break(self):
        self.assertEqual(nested_targets(), 511)

    def test_generator_expression_uses_nested_bindings(self):
        self.assertEqual(nested_genexpr(), 331)

    def test_nested_array_projection_is_shared(self):
        self.assertEqual(nested_array_unpacking(), 120)

    def test_bound_python_tuple_can_be_matched_and_unpacked(self):
        self.assertEqual(python_tuple_unpacking(), 41)

    def test_scalar_field_cannot_be_unpacked(self):
        @compile(suffix='yield_nested_scalar')
        def bad() -> i32:
            for ((a, b), c) in yield_tuple_values(1):
                return a + b + c
            return 0

        try:
            with patch.object(logger, 'raise_on_error', True):
                with self.assertRaisesRegex(TypeError, r'Cannot bind yield field 0\.0: i32'):
                    bad()
        finally:
            clear_failed_group(bad._binding.group_key)


MixedPair = struct[i32, f64]
NestedPair = struct[MixedPair, i32]
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


@compile
def nested_pairs() -> NestedPair:
    yield ((7, 1.25), 3)
    yield ((9, 1.75), 4)


@compile
def nested_targets() -> i32:
    total: i32 = 0
    a: i32 = 0
    for ((a, b), c) in nested_pairs():
        total += a + i32(b * 100.0) + c
        if c == 3:
            continue
        break
    return total + a + i32(b * 100.0) + c


@compile
def nested_genexpr() -> i32:
    total: i32 = 0
    outside: i32 = 4
    for value in (a + i32(b * 100.0) + c + outside for ((a, b), c) in nested_pairs()):
        total += value
    return total


ArrayPair = struct[array[i32, 2], i32]


@compile
def array_pair(start: i32) -> ArrayPair:
    return ((start, start + 1), start + 2)


@compile
def array_pairs(start: i32) -> ArrayPair:
    yield ((start, start + 1), start + 2)
    yield ((start + 3, start + 4), start + 5)


@compile
def nested_array_unpacking() -> i32:
    ((a, b), c) = array_pair(1)
    total: i32 = a + b + c
    match array_pair(4):
        case ((x, 5), z):
            total += x + 5 + z
        case _:
            return -1
    for ((u, v), w) in array_pairs(7):
        total += u + v + w
    packet: ArrayPair = array_pair(13)
    row, tail = packet
    match packet:
        case (matched_row, _):
            return total + row[0] + matched_row[1] + tail


@compile
def python_tuple_unpacking() -> i32:
    value = ((7, 8), 9)
    ((a, b), c) = value
    match value:
        case ((7, captured), tail) if captured > 0:
            return a + b + c + captured + tail
        case _:
            return 0


if __name__ == '__main__':
    unittest.main()
