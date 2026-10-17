"""Compile and execute nested zero and bool constant initializers."""

import unittest

from pythoc import (
    array, compile, f64, i32, nullptr, ptr, static, union,
)


@compile
class ZeroInner:
    pointer: ptr[i32]
    value: i32


@compile
class ZeroOuter:
    tag: i32
    inner: ZeroInner


@compile
def nested_struct_zero() -> i32:
    values: static[array[ZeroOuter, 2]] = (
        (1, (nullptr, 7)), (2, 0),
    )
    return (values[0].inner.value + values[1].tag
            + i32(values[1].inner.pointer == nullptr) + values[1].inner.value)


@compile
def zero_arrays() -> i32:
    pointers: static[array[ptr[i32], 2]] = 0
    matrix: static[array[array[i32, 2], 2]] = 0
    records: static[array[ZeroInner, 2]] = 0
    return (i32(pointers[0] == nullptr) + i32(pointers[1] == nullptr)
            + matrix[0][0] + matrix[1][1] + records[0].value
            + i32(records[1].pointer == nullptr))


@compile
def zero_union() -> i32:
    value: static[union[i32, f64]] = 0
    return value[0]


@compile
def bool_integer_constants() -> i32:
    values: static[array[i32, 2]] = (True, False)
    return values[0] * 10 + values[1]


class TestConstantInitialization(unittest.TestCase):
    def test_nested_struct_zero(self):
        self.assertEqual(nested_struct_zero(), 10)

    def test_zero_arrays(self):
        self.assertEqual(zero_arrays(), 3)

    def test_zero_union(self):
        self.assertEqual(zero_union(), 0)

    def test_bool_integer_constants(self):
        self.assertEqual(bool_integer_constants(), 10)


if __name__ == "__main__":
    unittest.main()
