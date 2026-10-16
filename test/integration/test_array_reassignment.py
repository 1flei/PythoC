"""Compile and execute array reassignment across local and member stores."""

import ctypes
import unittest

from pythoc import array, compile, i32, i64, nullptr, ptr


@compile
class ArrayHolder:
    before: i32
    values: array[i32, 3]
    after: i32


@compile
def reassign_full() -> i64:
    values: array[i64, 4]
    values = (10, 20, 30, 40)
    return values[0] + values[1] + values[2] + values[3]


@compile
def reassign_short() -> i32:
    values: array[i32, 3] = (1, 2, 3)
    values = (5,)
    values[2] = 9
    return values[0] * 100 + values[1] * 10 + values[2]


@compile
def reassign_empty() -> i32:
    values: array[i32, 3] = (1, 2, 3)
    values = ()
    return values[0] + values[1] + values[2]


@compile
def reassign_list() -> i32:
    values: array[i32, 3] = (1, 2, 3)
    values = [7, 8]
    return values[0] * 100 + values[1] * 10 + values[2]


@compile
def reassign_pointers(seed: ptr[i32]) -> i32:
    values: array[ptr[i32], 3]
    values = (seed, nullptr)
    if values[0] == seed and values[1] == nullptr and values[2] == nullptr:
        return values[0][0]
    return -1


@compile
def reassign_in_branch(flag: i32) -> i32:
    values: array[i32, 2] = (0, 0)
    if flag:
        values = (3, 4)
    return values[0] + values[1]


@compile
def reassign_member() -> i32:
    holder: ArrayHolder
    holder.before = 17
    holder.after = 19
    holder.values = (3, 4, 5)
    return (holder.before + holder.after + holder.values[0]
            + holder.values[1] + holder.values[2])


@compile
def reassign_row() -> i32:
    matrix: array[i32, 2, 3] = ((1, 2, 3), (4, 5, 6))
    matrix[1] = (7,)
    return (matrix[0][0] + matrix[0][1] + matrix[0][2]
            + matrix[1][0] + matrix[1][1] + matrix[1][2])


@compile
def reassign_array_pointer() -> i32:
    first: array[i32, 2] = (1, 2)
    second: array[i32, 2] = (3, 4)
    pointer: ptr[array[i32, 2]] = ptr(first)
    pointer = ptr(second)
    pointer[0][0] = 9
    return first[0] * 100 + second[0] * 10 + pointer[0][1]


class TestArrayReassignment(unittest.TestCase):
    def test_full_tuple(self):
        self.assertEqual(reassign_full(), 100)

    def test_short_tuple_overwrites_and_zero_fills(self):
        self.assertEqual(reassign_short(), 509)

    def test_empty_tuple_zero_fills(self):
        self.assertEqual(reassign_empty(), 0)

    def test_list_zero_fills(self):
        self.assertEqual(reassign_list(), 780)

    def test_pointer_elements(self):
        seed = ctypes.c_int32(42)
        self.assertEqual(reassign_pointers(ctypes.pointer(seed)), 42)

    def test_branch(self):
        self.assertEqual(reassign_in_branch(1), 7)
        self.assertEqual(reassign_in_branch(0), 0)

    def test_member_copy_preserves_adjacent_fields(self):
        self.assertEqual(reassign_member(), 48)

    def test_multidimensional_row(self):
        self.assertEqual(reassign_row(), 13)

    def test_pointer_assignment_still_aliases_the_destination(self):
        self.assertEqual(reassign_array_pointer(), 194)


if __name__ == "__main__":
    unittest.main()
