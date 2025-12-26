#!/usr/bin/env python3
"""
Test yield-based generator functions

Tests the simplified yield implementation that transforms yield functions
into vtable-based iterators automatically.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import unittest
from pythoc import compile, i32
from pythoc.builtin_entities.types import bool as bool_type


# Basic yield in while loop
@compile
def simple_seq(n: i32) -> i32:
    """Generate sequence 0, 1, 2, ..., n-1"""
    i: i32 = 0
    while i < n:
        yield i
        i = i + 1


# Test basic iteration
@compile
def test_simple_seq() -> i32:
    sum: i32 = 0
    for i in simple_seq(10):
        sum = sum + i
    return sum


# Yield with parameter
@compile
def countdown(start: i32) -> i32:
    """Generate countdown from start to 1"""
    n: i32 = start
    while n > 0:
        yield n
        n = n - 1


@compile
def test_countdown() -> i32:
    sum: i32 = 0
    for i in countdown(5):
        sum = sum + i
    return sum


# Yield with multiple local variables  
@compile
def fibonacci(limit: i32) -> i32:
    """Generate Fibonacci numbers less than limit"""
    a: i32 = 0
    b: i32 = 1
    while a < limit:
        yield a
        new_a: i32 = b
        new_b: i32 = a + b
        a = new_a
        b = new_b


@compile
def test_fibonacci() -> i32:
    sum: i32 = 0
    for n in fibonacci(100):
        sum = sum + n
    return sum


# Empty iterator
@compile
def empty_seq() -> i32:
    """Iterator that yields nothing"""
    i: i32 = 0
    while i < 0:
        yield i
        i = i + 1


@compile
def test_empty() -> i32:
    count: i32 = 0
    for i in empty_seq():
        count = count + 1
    return count


class TestYield(unittest.TestCase):
    def test_simple_seq(self):
        result = test_simple_seq()
        expected = sum(range(10))  # 45
        self.assertEqual(result, expected)
    
    def test_countdown(self):
        result = test_countdown()
        expected = 5 + 4 + 3 + 2 + 1  # 15
        self.assertEqual(result, expected)
    
    def test_fibonacci(self):
        result = test_fibonacci()
        # Fibonacci < 100: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89
        expected = 0 + 1 + 1 + 2 + 3 + 5 + 8 + 13 + 21 + 34 + 55 + 89  # 232
        self.assertEqual(result, expected)
    
    def test_empty(self):
        result = test_empty()
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
