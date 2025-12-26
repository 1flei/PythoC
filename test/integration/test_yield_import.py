#!/usr/bin/env python3
"""
Test importing and using yield functions from other modules

This test should reproduce the cross-module yield function issue.
"""

import unittest
from pythoc import compile, i32
from test.integration.test_yield_import_helper import generator_with_builtin


@compile
def test_imported_yield_with_builtin() -> i32:
    """Test using imported yield generator that calls builtin"""
    sum: i32 = 0
    for val in generator_with_builtin(5):
        sum = sum + val
    return sum


class TestYieldImport(unittest.TestCase):
    def test_imported_yield_with_builtin(self):
        result = test_imported_yield_with_builtin()
        expected = 0 + 1 + 2 + 3 + 4  # 10
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
