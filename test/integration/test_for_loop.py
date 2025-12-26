"""
Test for-in loop with range iterator

This demonstrates the new iterator protocol for for-in loops.
"""

from pythoc import compile, i32, seq

@compile
def test_range_basic() -> i32:
    """Test basic range iteration"""
    sum: i32 = 0
    for i in seq(10):
        sum = sum + i
    return sum

@compile
def test_range_with_start() -> i32:
    """Test range with start and stop"""
    sum: i32 = 0
    for i in seq(5, 15):
        sum = sum + i
    return sum

@compile
def test_range_with_step() -> i32:
    """Test range with start, stop, and step"""
    sum: i32 = 0
    for i in seq(0, 20, 2):
        sum = sum + i
    return sum

@compile
def test_range_negative_step() -> i32:
    """Test range with negative step"""
    sum: i32 = 0
    for i in seq(10, 0, -1):
        sum = sum + i
    return sum

@compile
def test_range_nested() -> i32:
    """Test nested range loops"""
    sum: i32 = 0
    for i in seq(5):
        for j in seq(3):
            sum = sum + i * 10 + j
    return sum

@compile
def test_range_with_break() -> i32:
    """Test range with early exit"""
    sum: i32 = 0
    for i in seq(100):
        sum = sum + i
        if sum > 50:
            break
    return sum

@compile
def test_range_empty() -> i32:
    """Test empty range"""
    count: i32 = 0
    for i in seq(0):
        count = count + 1
    return count

@compile
def test_range_single() -> i32:
    """Test single iteration range"""
    result: i32 = 0
    for i in seq(1):
        result = i
    return result

@compile
def test_lowercase_range() -> i32:
    """Test lowercase range (Python-style)"""
    sum: i32 = 0
    for i in seq(5):
        sum = sum + i
    return sum

@compile
def test_mixed_case() -> i32:
    """Test mixing range and range"""
    sum: i32 = 0
    for i in seq(3):
        for j in seq(2):
            sum = sum + i + j
    return sum

@compile
def test_continue() -> i32:
    """Test continue statement"""
    sum: i32 = 0
    for i in seq(10):
        if i % 2 == 0:
            continue
        sum = sum + i
    return sum

@compile
def test_break_in_nested() -> i32:
    """Test break in nested loop"""
    sum: i32 = 0
    for i in seq(5):
        for j in seq(5):
            if j == 3:
                break
            sum = sum + i * 10 + j
    return sum

@compile
def test_continue_in_nested() -> i32:
    """Test continue in nested loop"""
    sum: i32 = 0
    for i in seq(5):
        for j in seq(5):
            if j == 2:
                continue
            sum = sum + i * 10 + j
    return sum

@compile
def test_while_with_break() -> i32:
    """Test while loop with break"""
    sum: i32 = 0
    i: i32 = 0
    while i < 100:
        sum = sum + i
        i = i + 1
        if sum > 50:
            break
    return sum

@compile
def test_while_with_continue() -> i32:
    """Test while loop with continue"""
    sum: i32 = 0
    i: i32 = 0
    while i < 10:
        i = i + 1
        if i % 2 == 0:
            continue
        sum = sum + i
    return sum

import unittest


class TestForLoop(unittest.TestCase):
    """Test for-in loop with range iterator"""

    def test_range_basic(self):
        """Test basic range iteration: sum of 0..9 = 45"""
        self.assertEqual(test_range_basic(), 45)

    def test_range_with_start(self):
        """Test range with start and stop: sum of 5..14 = 95"""
        self.assertEqual(test_range_with_start(), 95)

    def test_range_with_step(self):
        """Test range with step: sum of 0,2,4,6,8,10,12,14,16,18 = 90"""
        self.assertEqual(test_range_with_step(), 90)

    def test_range_negative_step(self):
        """Test range with negative step: sum of 10..1 = 55"""
        self.assertEqual(test_range_negative_step(), 55)

    def test_range_nested(self):
        """Test nested range loops"""
        self.assertEqual(test_range_nested(), 315)

    def test_range_with_break(self):
        """Test range with early exit"""
        self.assertEqual(test_range_with_break(), 55)

    def test_range_empty(self):
        """Test empty range"""
        self.assertEqual(test_range_empty(), 0)

    def test_range_single(self):
        """Test single iteration range"""
        self.assertEqual(test_range_single(), 0)

    def test_lowercase_range(self):
        """Test lowercase range: sum of 0..4 = 10"""
        self.assertEqual(test_lowercase_range(), 10)

    def test_mixed_case(self):
        """Test mixing range loops"""
        self.assertEqual(test_mixed_case(), 9)

    def test_continue(self):
        """Test continue statement: sum of odd numbers 1+3+5+7+9 = 25"""
        self.assertEqual(test_continue(), 25)

    def test_break_in_nested(self):
        """Test break in nested loop"""
        self.assertEqual(test_break_in_nested(), 315)

    def test_continue_in_nested(self):
        """Test continue in nested loop"""
        self.assertEqual(test_continue_in_nested(), 440)

    def test_while_with_break(self):
        """Test while loop with break"""
        self.assertEqual(test_while_with_break(), 55)

    def test_while_with_continue(self):
        """Test while loop with continue: sum of odd numbers = 25"""
        self.assertEqual(test_while_with_continue(), 25)


if __name__ == '__main__':
    unittest.main()
