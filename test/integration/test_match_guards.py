#!/usr/bin/env python3
"""
Match/case guard clauses and range tests
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc import i32, compile
import unittest


@compile
def test_guard_simple(x: i32) -> i32:
    """Simple guard clause with comparison"""
    match x:
        case n if n > 0:
            return 1
        case n if n < 0:
            return -1
        case _:
            return 0


@compile
def test_guard_with_arithmetic(x: i32) -> i32:
    """Guard with bound variable used in body"""
    match x:
        case n if n > 10:
            return n * 2
        case n if n > 5:
            return n + 10
        case n:
            return n


@compile
def test_guard_complex_condition(x: i32) -> i32:
    """Guard with complex boolean expression"""
    match x:
        case n if n > 0 and n < 100:
            return 1
        case n if n >= 100 and n <= 1000:
            return 2
        case _:
            return 0


@compile
def test_guard_mixed_with_literal(x: i32) -> i32:
    """Mix literal cases with guard cases"""
    match x:
        case 0:
            return 100
        case n if n > 0:
            return n
        case n if n < 0:
            return -n
        case _:
            return 0


@compile
def test_guard_multiple_bindings(x: i32, y: i32) -> i32:
    """Guard accessing multiple variables"""
    match x:
        case n if n == y:
            return 1
        case n if n > y:
            return 2
        case _:
            return 3


@compile
def test_range_basic(x: i32) -> i32:
    """Basic integer range matching using guards"""
    match x:
        case n if 1 <= n <= 10:
            return 1
        case n if 11 <= n <= 20:
            return 2
        case n if 21 <= n <= 30:
            return 3
        case _:
            return 0


@compile
def test_range_negative(x: i32) -> i32:
    """Ranges with negative numbers"""
    match x:
        case n if -10 <= n <= -1:
            return -1
        case 0:
            return 0
        case n if 1 <= n <= 10:
            return 1
        case _:
            return 99


@compile
def test_range_single_value(x: i32) -> i32:
    """Single value range (equivalent to literal)"""
    match x:
        case 5:
            return 555
        case n if 1 <= n <= 10:
            return 1
        case _:
            return 0


@compile
def test_range_large_span(x: i32) -> i32:
    """Large range span"""
    match x:
        case n if 1 <= n <= 1000:
            return 1
        case n if 1001 <= n <= 2000:
            return 2
        case _:
            return 0


@compile
def test_range_mixed_with_literal(x: i32) -> i32:
    """Mix range and literal cases"""
    match x:
        case 0:
            return 100
        case n if 1 <= n <= 10:
            return 1
        case 100:
            return 200
        case n if 101 <= n <= 200:
            return 2
        case _:
            return 0


class TestMatchGuardClauses(unittest.TestCase):
    """Test guard clause feature"""
    
    def test_guard_simple_positive(self):
        self.assertEqual(test_guard_simple(5), 1)
    
    def test_guard_simple_negative(self):
        self.assertEqual(test_guard_simple(-3), -1)
    
    def test_guard_simple_zero(self):
        self.assertEqual(test_guard_simple(0), 0)
    
    def test_guard_with_arithmetic_large(self):
        self.assertEqual(test_guard_with_arithmetic(20), 40)
    
    def test_guard_with_arithmetic_medium(self):
        self.assertEqual(test_guard_with_arithmetic(7), 17)
    
    def test_guard_with_arithmetic_small(self):
        self.assertEqual(test_guard_with_arithmetic(3), 3)
    
    def test_guard_complex_small(self):
        self.assertEqual(test_guard_complex_condition(50), 1)
    
    def test_guard_complex_large(self):
        self.assertEqual(test_guard_complex_condition(500), 2)
    
    def test_guard_complex_outlier(self):
        self.assertEqual(test_guard_complex_condition(2000), 0)
    
    def test_guard_mixed_literal(self):
        self.assertEqual(test_guard_mixed_with_literal(0), 100)
        self.assertEqual(test_guard_mixed_with_literal(5), 5)
        self.assertEqual(test_guard_mixed_with_literal(-5), 5)


class TestMatchRanges(unittest.TestCase):
    """Test integer range feature"""
    
    def test_range_basic_first(self):
        self.assertEqual(test_range_basic(5), 1)
    
    def test_range_basic_second(self):
        self.assertEqual(test_range_basic(15), 2)
    
    def test_range_basic_third(self):
        self.assertEqual(test_range_basic(25), 3)
    
    def test_range_basic_default(self):
        self.assertEqual(test_range_basic(100), 0)
    
    def test_range_boundary_low(self):
        self.assertEqual(test_range_basic(1), 1)
    
    def test_range_boundary_high(self):
        self.assertEqual(test_range_basic(30), 3)
    
    def test_range_negative_values(self):
        self.assertEqual(test_range_negative(-5), -1)
        self.assertEqual(test_range_negative(0), 0)
        self.assertEqual(test_range_negative(5), 1)
    
    def test_range_single_value(self):
        self.assertEqual(test_range_single_value(5), 555)
    
    def test_range_large_span(self):
        self.assertEqual(test_range_large_span(500), 1)
        self.assertEqual(test_range_large_span(1500), 2)
    
    def test_range_mixed_literal(self):
        self.assertEqual(test_range_mixed_with_literal(0), 100)
        self.assertEqual(test_range_mixed_with_literal(5), 1)
        self.assertEqual(test_range_mixed_with_literal(100), 200)
        self.assertEqual(test_range_mixed_with_literal(150), 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
