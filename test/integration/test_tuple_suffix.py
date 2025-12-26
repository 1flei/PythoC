#!/usr/bin/env python3
import unittest
from pythoc import i32, i64, u64, compile


# Test tuple suffix normalization
@compile(suffix=(i32, 0, u64))
def test_func1(x: i32) -> i32:
    return x + 1


@compile(suffix=(i64, 2, u64))
def test_func2(x: i64) -> i64:
    return x + 2


def another_scope():
    # Deduplication test: same suffix should reuse files
    @compile(suffix=(i32, 0, u64))
    def test_func3(y: i32) -> i32:
        return y + 3
    return test_func3


test_func3 = another_scope()


class TestTupleSuffix(unittest.TestCase):
    def test_func1(self):
        self.assertEqual(test_func1(10), 11)
    
    def test_func2(self):
        self.assertEqual(test_func2(20), 22)
    
    def test_func3(self):
        self.assertEqual(test_func3(30), 33)


if __name__ == "__main__":
    unittest.main()
