"""Addressable aggregate temporaries keep distinct storage across iterations."""

import unittest

from pythoc import array, compile, enum, i8, i32, ptr, seq, union


@enum(i8)
class Choice:
    Item: i32


Box = union[i32, i32]


@compile
def make_box(value: i32) -> Box:
    result: Box
    result[0] = value
    return result


@compile
def retained_enums(count: i32) -> i32:
    addresses: array[ptr[Choice], 2]
    for index in seq(count):
        addresses[index] = ptr(Choice(Choice.Item, index))
    first: i32 = 0
    second: i32 = 0
    match addresses[0][0]:
        case (Choice.Item, value):
            first = value
    match addresses[1][0]:
        case (Choice.Item, value):
            second = value
    return first * 10 + second


@compile
def retained_union_fields(count: i32) -> i32:
    addresses: array[ptr[i32], 2]
    for index in seq(count):
        addresses[index] = ptr(make_box(index)[0])
    return addresses[0][0] * 10 + addresses[1][0]


class TemporaryAddressLifetimeTest(unittest.TestCase):
    def test_enum_constructions_keep_separate_addressable_objects(self):
        self.assertEqual(retained_enums(2), 1)

    def test_union_rvalue_fields_keep_separate_addressable_objects(self):
        self.assertEqual(retained_union_fields(2), 1)


if __name__ == '__main__':
    unittest.main()
