"""Python entry accepts an uncalled PythoC callable as a func argument."""

import unittest

from pythoc import compile, func, i64


Unary = func[i64, i64]


@compile
def increment(value: i64) -> i64:
    return value + 1


@compile
def apply_unary(callback: Unary, value: i64) -> i64:
    return callback(value)


class TestPythonFunctionPointerArgument(unittest.TestCase):
    def test_deferred_callable_materializes_as_function_pointer(self):
        self.assertFalse(increment.is_fast_bound())
        self.assertEqual(apply_unary(increment, 41), 42)
        self.assertTrue(increment.is_fast_bound())


if __name__ == '__main__':
    unittest.main()
