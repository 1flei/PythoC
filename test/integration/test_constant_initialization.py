"""Compile and execute nested zero and bool constant initializers."""

import unittest

from pythoc import (
    array, compile, f64, i16, i32, nullptr, ptr, static, union,
)


@compile
class ZeroInner:
    pointer: ptr[i32]
    value: i32


@compile
class ZeroOuter:
    tag: i32
    inner: ZeroInner


@union
class StaticAlternative:
    storage: array[i32, 2]
    number: i32


@compile
class StaticSeeds:
    selected: static[StaticAlternative] = {'number': 17}


@compile
def named_union_constants() -> i32:
    selected: static[array[StaticAlternative, 2]] = (
        {'number': 3}, {'storage': (5, 7)},
    )
    return StaticSeeds.selected.number + selected[0].number + selected[1].storage[1]


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


@compile
def cast_and_negate_constants() -> i32:
    values: static[array[array[i16, 2], 2]] = (
        (i16((1.402 * (1 << 6)) + 0.5), -i32(i16((0.3441 * (1 << 6)) + 0.5))),
        (i16(6) * i16(7), i16(4) + i16(1)),
    )
    return i32(values[0][0]) + i32(values[0][1]) + i32(values[1][0]) + i32(values[1][1])


class TestConstantInitialization(unittest.TestCase):
    def test_nested_struct_zero(self):
        self.assertEqual(nested_struct_zero(), 10)

    def test_zero_arrays(self):
        self.assertEqual(zero_arrays(), 3)

    def test_zero_union(self):
        self.assertEqual(zero_union(), 0)

    def test_named_union_constants_in_static_storage(self):
        self.assertEqual(named_union_constants(), 27)

    def test_bool_integer_constants(self):
        self.assertEqual(bool_integer_constants(), 10)

    def test_native_integer_operations_remain_static_constants(self):
        self.assertEqual(cast_and_negate_constants(), 115)


if __name__ == "__main__":
    unittest.main()
