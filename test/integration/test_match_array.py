#!/usr/bin/env python3
"""
Match/case array pattern tests
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc import i32, compile, array
import unittest


@compile
def test_array_literal_123() -> i32:
    """Match array [1,2,3]"""
    arr: array[i32, 3] = array[i32, 3]()
    arr[0] = 1
    arr[1] = 2
    arr[2] = 3
    match arr:
        case [1, 2, 3]:
            return 123
        case [0, 0, 0]:
            return 0
        case _:
            return 999


@compile
def test_array_literal_zeros() -> i32:
    """Match array [0,0,0]"""
    arr: array[i32, 3] = array[i32, 3]()
    match arr:
        case [1, 2, 3]:
            return 123
        case [0, 0, 0]:
            return 0
        case _:
            return 999


@compile
def test_array_literal_nomatch() -> i32:
    """Match array [9,8,7]"""
    arr: array[i32, 3] = array[i32, 3]()
    arr[0] = 9
    arr[1] = 8
    arr[2] = 7
    match arr:
        case [1, 2, 3]:
            return 123
        case [0, 0, 0]:
            return 0
        case _:
            return 999


@compile
def test_array_bind_123() -> i32:
    """Bind array elements [1,2,3]"""
    arr: array[i32, 3] = array[i32, 3]()
    arr[0] = 1
    arr[1] = 2
    arr[2] = 3
    match arr:
        case [0, 0, 0]:
            return 0
        case [x, y, z]:
            return x + y + z


@compile
def test_array_wildcard_first() -> i32:
    """Match first element = 1"""
    arr: array[i32, 3] = array[i32, 3]()
    arr[0] = 1
    arr[1] = 99
    arr[2] = 99
    match arr:
        case [1, _, _]:
            return 1
        case [_, 2, _]:
            return 2
        case [_, _, 3]:
            return 3
        case _:
            return 0


@compile
def test_array_wildcard_second() -> i32:
    """Match second element = 2"""
    arr: array[i32, 3] = array[i32, 3]()
    arr[0] = 99
    arr[1] = 2
    arr[2] = 99
    match arr:
        case [1, _, _]:
            return 1
        case [_, 2, _]:
            return 2
        case [_, _, 3]:
            return 3
        case _:
            return 0


@compile
def test_array_mixed_first() -> i32:
    """Mix pattern: first=1, bind others"""
    arr: array[i32, 3] = array[i32, 3]()
    arr[0] = 1
    arr[1] = 10
    arr[2] = 20
    match arr:
        case [0, 0, 0]:
            return 0
        case [1, x, y]:
            return x + y
        case [x, y, 3]:
            return x + y
        case [x, y, z]:
            return x * 100 + y * 10 + z


@compile
def test_array_mixed_general() -> i32:
    """Mix pattern: general case"""
    arr: array[i32, 3] = array[i32, 3]()
    arr[0] = 1
    arr[1] = 2
    arr[2] = 3
    match arr:
        case [0, 0, 0]:
            return 0
        case [1, x, y]:
            return x + y
        case [x, y, 3]:
            return x + y
        case [x, y, z]:
            return x * 100 + y * 10 + z


@compile
def test_array_guard_all_same() -> i32:
    """Guard: all elements same"""
    arr: array[i32, 3] = array[i32, 3]()
    arr[0] = 5
    arr[1] = 5
    arr[2] = 5
    match arr:
        case [x, y, z] if x == y and y == z:
            return 1
        case [x, y, z] if x < y and y < z:
            return 2
        case [x, y, z] if x > y and y > z:
            return 3
        case _:
            return 0


@compile
def test_array_guard_ascending() -> i32:
    """Guard: ascending order"""
    arr: array[i32, 3] = array[i32, 3]()
    arr[0] = 1
    arr[1] = 2
    arr[2] = 3
    match arr:
        case [x, y, z] if x == y and y == z:
            return 1
        case [x, y, z] if x < y and y < z:
            return 2
        case [x, y, z] if x > y and y > z:
            return 3
        case _:
            return 0


@compile
def test_array_guard_descending() -> i32:
    """Guard: descending order"""
    arr: array[i32, 3] = array[i32, 3]()
    arr[0] = 3
    arr[1] = 2
    arr[2] = 1
    match arr:
        case [x, y, z] if x == y and y == z:
            return 1
        case [x, y, z] if x < y and y < z:
            return 2
        case [x, y, z] if x > y and y > z:
            return 3
        case _:
            return 0


@compile
def test_array_larger_match() -> i32:
    """Match larger array [1,2,3,4,5]"""
    arr: array[i32, 5] = array[i32, 5]()
    arr[0] = 1
    arr[1] = 2
    arr[2] = 3
    arr[3] = 4
    arr[4] = 5
    match arr:
        case [1, 2, 3, 4, 5]:
            return 12345
        case [first, _, _, _, last] if first == last:
            return 1
        case [a, b, c, d, e]:
            return a + e


@compile
def test_array_larger_same() -> i32:
    """Match larger array: first==last"""
    arr: array[i32, 5] = array[i32, 5]()
    arr[0] = 7
    arr[1] = 2
    arr[2] = 3
    arr[3] = 4
    arr[4] = 7
    match arr:
        case [1, 2, 3, 4, 5]:
            return 12345
        case [first, _, _, _, last] if first == last:
            return 1
        case [a, b, c, d, e]:
            return a + e


@compile
def test_array_larger_general() -> i32:
    """Match larger array: general case"""
    arr: array[i32, 5] = array[i32, 5]()
    arr[0] = 10
    arr[1] = 20
    arr[2] = 30
    arr[3] = 40
    arr[4] = 50
    match arr:
        case [1, 2, 3, 4, 5]:
            return 12345
        case [first, _, _, _, last] if first == last:
            return 1
        case [a, b, c, d, e]:
            return a + e


class TestMatchArrayPatterns(unittest.TestCase):
    """Test array pattern feature"""
    
    def test_array_literal_match(self):
        self.assertEqual(test_array_literal_123(), 123)
    
    def test_array_literal_zeros(self):
        self.assertEqual(test_array_literal_zeros(), 0)
    
    def test_array_literal_no_match(self):
        self.assertEqual(test_array_literal_nomatch(), 999)
    
    def test_array_bind_values(self):
        self.assertEqual(test_array_bind_123(), 6)
    
    def test_array_wildcard_first(self):
        self.assertEqual(test_array_wildcard_first(), 1)
    
    def test_array_wildcard_second(self):
        self.assertEqual(test_array_wildcard_second(), 2)
    
    def test_array_mixed_first_pattern(self):
        self.assertEqual(test_array_mixed_first(), 30)
    
    def test_array_mixed_general(self):
        self.assertEqual(test_array_mixed_general(), 5)
    
    def test_array_with_guard_all_same(self):
        self.assertEqual(test_array_guard_all_same(), 1)
    
    def test_array_with_guard_ascending(self):
        self.assertEqual(test_array_guard_ascending(), 2)
    
    def test_array_with_guard_descending(self):
        self.assertEqual(test_array_guard_descending(), 3)
    
    def test_array_larger_exact_match(self):
        self.assertEqual(test_array_larger_match(), 12345)
    
    def test_array_larger_first_last_same(self):
        self.assertEqual(test_array_larger_same(), 1)
    
    def test_array_larger_general(self):
        self.assertEqual(test_array_larger_general(), 60)


if __name__ == '__main__':
    unittest.main(verbosity=2)
