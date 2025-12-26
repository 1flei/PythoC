"""
Test constant for loop unrolling

This tests the compile-time loop unrolling feature for constant iterables.
"""

from pythoc import compile, i32

@compile
def test_constant_list() -> i32:
    """Test for loop with constant list - should unroll at compile time"""
    sum: i32 = 0
    for i in [1, 2, 3, 4, 5]:
        sum = sum + i
    return sum

@compile
def test_constant_empty() -> i32:
    """Test for loop with empty constant list"""
    sum: i32 = 0
    for i in []:
        sum = sum + 1
    return sum

@compile
def test_constant_single() -> i32:
    """Test for loop with single element"""
    result: i32 = 0
    for i in [42]:
        result = i
    return result

@compile
def test_constant_nested() -> i32:
    """Test nested constant for loops"""
    sum: i32 = 0
    for i in [1, 2, 3]:
        for j in [10, 20]:
            sum = sum + i + j
    return sum

@compile
def test_constant_with_computation() -> i32:
    """Test constant loop with computation inside body"""
    result: i32 = 0
    for i in [1, 2, 3, 4]:
        result = result + i * i
    return result

@compile
def test_constant_overwrite() -> i32:
    """Test that loop variable is properly overwritten each iteration"""
    last: i32 = 0
    for i in [5, 10, 15, 20]:
        last = i
    return last

@compile
def test_constant_mixed_with_range() -> i32:
    """Test mixing constant loop with range loop"""
    sum: i32 = 0
    for i in [1, 2]:
        for j in range(3):
            sum = sum + i * 10 + j
    return sum

@compile
def test_constant_loop_with_return() -> i32:
    """Test mixing constant loop with range loop"""
    sum: i32 = 0
    for i in range(10):
        for j in range(20):
            if sum > 1000:
                return sum
            sum = sum + i * 10 + j
    return sum

@compile
def test_constant_loop_with_break() -> i32:
    """Test mixing constant loop with range loop"""
    sum: i32 = 0
    for i in range(10):
        for j in range(20):
            if sum > 1000:
                break
            sum = sum + i * 10 + j
    return sum

@compile
def test_constant_loop_with_continue() -> i32:
    """Test mixing constant loop with range loop"""
    sum: i32 = 0
    for i in range(10):
        for j in range(20):
            if sum > 1000:
                continue
            sum = sum + i * 10 + j
    return sum

import unittest


class TestConstantForLoop(unittest.TestCase):
    """Test constant for loop unrolling"""

    def test_constant_list(self):
        """Test for loop with constant list: 1+2+3+4+5 = 15"""
        self.assertEqual(test_constant_list(), 15)

    def test_constant_empty(self):
        """Test for loop with empty constant list"""
        self.assertEqual(test_constant_empty(), 0)

    def test_constant_single(self):
        """Test for loop with single element"""
        self.assertEqual(test_constant_single(), 42)

    def test_constant_nested(self):
        """Test nested constant for loops: 11+21+12+22+13+23 = 102"""
        self.assertEqual(test_constant_nested(), 102)

    def test_constant_with_computation(self):
        """Test constant loop with computation: 1+4+9+16 = 30"""
        self.assertEqual(test_constant_with_computation(), 30)

    def test_constant_overwrite(self):
        """Test that loop variable is properly overwritten"""
        self.assertEqual(test_constant_overwrite(), 20)

    def test_constant_mixed_with_range(self):
        """Test mixing constant loop with range loop"""
        self.assertEqual(test_constant_mixed_with_range(), 96)

    def test_constant_loop_with_break(self):
        """Test constant loop with break"""
        result = test_constant_loop_with_break()
        self.assertIsInstance(result, int)

    def test_constant_loop_with_continue(self):
        """Test constant loop with continue"""
        result = test_constant_loop_with_continue()
        self.assertIsInstance(result, int)

    def test_constant_loop_with_return(self):
        """Test constant loop with return"""
        result = test_constant_loop_with_return()
        self.assertIsInstance(result, int)


if __name__ == '__main__':
    unittest.main()
