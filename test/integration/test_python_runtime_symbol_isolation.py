"""User kernels may have the same Python names as boundary helpers."""

import unittest

from pythoc import compile, i64


@compile
def release(value: i64) -> i64:
    return value + 1


@compile
def own(value: i64) -> i64:
    return value + 2


@compile
def none_ref() -> i64:
    return 43


@compile
def combine(value: i64) -> i64:
    return release(value) + own(value) + none_ref()


class PythonRuntimeSymbolIsolationTest(unittest.TestCase):
    def test_direct_entries_call_the_user_kernels(self):
        self.assertEqual(release(41), 42)
        self.assertEqual(own(40), 42)
        self.assertEqual(none_ref(), 43)

    def test_native_calls_and_python_entry_agree(self):
        self.assertEqual(combine(40), 126)


if __name__ == '__main__':
    unittest.main()
