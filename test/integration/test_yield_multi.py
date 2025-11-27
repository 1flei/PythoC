#!/usr/bin/env python3
"""
Test yield functions with multiple yield points

This tests the inlining optimization for generators with multiple yields.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc import compile, i32


# Multiple yields in sequence
@compile
def three_values() -> i32:
    """Yield three values: 1, 2, 3"""
    yield 1
    yield 2
    yield 3


@compile
def test_three_values() -> i32:
    sum: i32 = 0
    for x in three_values():
        sum = sum + x
    return sum


# Multiple yields with conditionals
@compile
def conditional_gen(n: i32) -> i32:
    """Generate values with conditional yields"""
    i: i32 = 0
    while i < n:
        if i % 2 == 0:
            yield i
        i = i + 1


@compile
def test_conditional_gen() -> i32:
    sum: i32 = 0
    for x in conditional_gen(10):
        sum = sum + x
    return sum


# Nested structure with multiple yields
@compile
def range_with_steps(start: i32, end: i32, step: i32) -> i32:
    """Generate range with custom step"""
    i: i32 = start
    while i < end:
        yield i
        i = i + step


@compile
def test_range_with_steps() -> i32:
    sum: i32 = 0
    for x in range_with_steps(0, 20, 3):
        sum = sum + x
    return sum


@compile
def test_multi_range() -> i32:
    sum: i32 = 0
    for i in range_with_steps(0, 20, 3):
        for j in range_with_steps(i, 2 * i, 1):
            sum = sum + i + j
    return sum


def main():
    """Run all tests"""
    print("Testing yield with multiple yield points...")
    print()
    
    # Test three_values
    result = test_three_values()
    expected = 1 + 2 + 3
    status = "PASS" if result == expected else "FAIL"
    print(f"test_three_values: {status} (result={result}, expected={expected})")
    assert result == expected, f"Expected {expected}, got {result}"
    
    # Test conditional_gen (0, 2, 4, 6, 8)
    result = test_conditional_gen()
    expected = 0 + 2 + 4 + 6 + 8
    status = "PASS" if result == expected else "FAIL"
    print(f"test_conditional_gen: {status} (result={result}, expected={expected})")
    assert result == expected, f"Expected {expected}, got {result}"
    
    # Test range_with_steps (0, 3, 6, 9, 12, 15, 18)
    result = test_range_with_steps()
    expected = 0 + 3 + 6 + 9 + 12 + 15 + 18
    status = "PASS" if result == expected else "FAIL"
    print(f"test_range_with_steps: {status} (result={result}, expected={expected})")
    assert result == expected, f"Expected {expected}, got {result}"

    result = test_multi_range()
    expected = 2016
    status = "PASS" if result == expected else "FAIL"
    print(f"test_multi_range: {status} (result={result}, expected={expected})")
    assert result == expected, f"Expected {expected}, got {result}"
    
    print()
    print("All multi-yield tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
