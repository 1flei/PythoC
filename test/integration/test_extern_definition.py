"""Header declarations and unrelated captured extern aliases coexist."""

import unittest

from pythoc import compile, extern, i32


@extern(lib="c", name="local_factorial")
def header_factorial(value: i32) -> i32:
    pass


@compile
def local_factorial(value: i32) -> i32:
    if value <= 1:
        return 1
    return value * header_factorial(value - 1)


@extern(lib="c", name="abs")
def selected(value: i32) -> i32:
    pass


@compile
def captured_abs(value: i32) -> i32:
    return selected(value)


@compile
def selected(value: i32) -> i32:
    return value + 1000


@compile
def local_alias(value: i32) -> i32:
    return selected(value)


class TestExternDefinition(unittest.TestCase):
    def test_header_alias_reuses_local_c_symbol(self):
        self.assertEqual(local_factorial(5), 120)

    def test_unrelated_c_symbol_keeps_its_captured_binding(self):
        self.assertEqual(captured_abs(-3), 3)

    def test_local_name_still_resolves_normally(self):
        self.assertEqual(local_alias(-3), 997)


if __name__ == "__main__":
    unittest.main()
