#!/usr/bin/env python3
"""
Test yield functions calling @compile functions from other modules

This test reproduces the cross-module issue where yield functions 
cannot find functions imported from other modules.
"""

import unittest
from pythoc import compile, i32
from test.integration.test_yield_cross_module_helper import external_add


@compile
def generator_with_external_call(n: i32) -> i32:
    """Yield generator that calls function from another module"""
    i: i32 = 0
    while i < n:
        result: i32 = external_add(i, 10)
        yield result
        i = i + 1


@compile
def test_yield_with_external_call() -> i32:
    """Test using yield generator that calls external function"""
    sum: i32 = 0
    for val in generator_with_external_call(3):
        sum = sum + val
    return sum


class TestYieldCrossModule(unittest.TestCase):
    def test_external_call(self):
        result = test_yield_with_external_call()
        expected = 10 + 11 + 12  # 33
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
