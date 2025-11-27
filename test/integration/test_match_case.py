#!/usr/bin/env python3
"""
Match/case statement tests (Python 3.10+ syntax)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc import i32, compile
from pythoc.decorators import clear_registry


@compile
def match_simple() -> i32:
    """Test basic match with integer literals"""
    x: i32 = 2
    result: i32 = 0
    
    match x:
        case 1:
            result = 10
        case 2:
            result = 20
        case 3:
            result = 30
        case _:
            result = 0
    
    return result


@compile
def match_no_default() -> i32:
    """Test match without wildcard default case"""
    x: i32 = 1
    result: i32 = 99
    
    match x:
        case 1:
            result = 10
        case 2:
            result = 20
    
    return result


@compile
def match_with_or_pattern() -> i32:
    """Test match with OR patterns (case 1 | 2 | 3)"""
    x: i32 = 2
    result: i32 = 0
    
    match x:
        case 1 | 2 | 3:
            result = 100
        case 4 | 5:
            result = 200
        case _:
            result = 0
    
    return result


@compile
def match_all_same_result() -> i32:
    """Test match where multiple cases have same result"""
    x: i32 = 5
    result: i32 = 0
    
    match x:
        case 1:
            result = 1
        case 2:
            result = 1
        case 3:
            result = 1
        case _:
            result = 0
    
    return result


@compile
def match_with_computation() -> i32:
    """Test match with computations in case bodies"""
    x: i32 = 2
    result: i32 = 0
    
    match x:
        case 1:
            result = x * 10
        case 2:
            result = x * 20
        case 3:
            result = x * 30
        case _:
            result = x
    
    return result


@compile
def match_nested() -> i32:
    """Test nested match statements"""
    x: i32 = 1
    y: i32 = 2
    result: i32 = 0
    
    match x:
        case 1:
            match y:
                case 1:
                    result = 11
                case 2:
                    result = 12
                case _:
                    result = 10
        case 2:
            result = 20
        case _:
            result = 0
    
    return result


@compile
def match_with_return() -> i32:
    """Test match with direct return statements"""
    x: i32 = 3
    
    match x:
        case 1:
            return 10
        case 2:
            return 20
        case 3:
            return 30
        case _:
            return 0


@compile
def match_in_loop() -> i32:
    """Test match statement inside a loop"""
    sum: i32 = 0
    i: i32 = 0
    
    while i < 5:
        match i:
            case 0:
                sum = sum + 1
            case 1:
                sum = sum + 10
            case 2:
                sum = sum + 100
            case _:
                sum = sum + 1000
        i = i + 1
    
    return sum


@compile
def match_large_values() -> i32:
    """Test match with larger integer values"""
    x: i32 = 1000
    result: i32 = 0
    
    match x:
        case 100:
            result = 1
        case 500:
            result = 2
        case 1000:
            result = 3
        case 5000:
            result = 4
        case _:
            result = 0
    
    return result


@compile
def match_negative_values() -> i32:
    """Test match with negative integer values"""
    x: i32 = -2
    result: i32 = 0
    
    match x:
        case -1:
            result = 10
        case -2:
            result = 20
        case -3:
            result = 30
        case _:
            result = 0
    
    return result


import unittest


class TestMatchCase(unittest.TestCase):
    def setUp(self):
        clear_registry()
    
    def tearDown(self):
        clear_registry()
    
    def test_simple_match(self):
        self.assertEqual(match_simple(), 20)
    
    def test_match_no_default(self):
        self.assertEqual(match_no_default(), 10)
    
    def test_match_with_or_pattern(self):
        self.assertEqual(match_with_or_pattern(), 100)
    
    def test_match_all_same_result(self):
        self.assertEqual(match_all_same_result(), 0)
    
    def test_match_with_computation(self):
        self.assertEqual(match_with_computation(), 40)
    
    def test_nested_match(self):
        self.assertEqual(match_nested(), 12)
    
    def test_match_with_return(self):
        self.assertEqual(match_with_return(), 30)
    
    def test_match_in_loop(self):
        # 1 + 10 + 100 + 1000 + 1000 = 2111
        self.assertEqual(match_in_loop(), 2111)
    
    def test_match_large_values(self):
        self.assertEqual(match_large_values(), 3)
    
    def test_match_negative_values(self):
        self.assertEqual(match_negative_values(), 20)


if __name__ == '__main__':
    unittest.main()
