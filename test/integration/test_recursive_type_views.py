"""Recursive factories preserve PythoC identity and independent field types."""

import unittest

from pythoc import compile, i32, i64, nullptr, ptr


def node_type():
    @compile
    class RecursiveNode:
        next: ptr['RecursiveNode']
        value: i32

    return RecursiveNode


First = node_type()
Second = node_type()


@compile
def read_recursive_views() -> i32:
    first: First
    second: Second
    first.value = 11
    second.value = 31
    second.next = nullptr
    first.next = ptr(second)
    return first.value + first.next.value


def typed_node_type(value_type):
    @compile
    class TypedNode:
        next: ptr['TypedNode']
        value: value_type
        canary: i32

    return TypedNode


Small = typed_node_type(i32)
Wide = typed_node_type(i64)


@compile
def read_independent_recursive_types() -> i64:
    small: Small
    wide: Wide
    small.value = 7
    small.canary = 0x6badf00d
    wide.value = 4294967296
    wide.canary = 0x13579bdf
    small.next = ptr(small)
    wide.next = ptr(wide)
    return i64(small.next.value) + wide.next.value


class RecursiveTypeViewsTest(unittest.TestCase):
    def test_field_access_through_repeated_recursive_declarations(self):
        self.assertEqual(read_recursive_views(), 42)

    def test_same_name_recursive_factories_with_different_arguments(self):
        self.assertEqual(read_independent_recursive_types(), 4294967303)


if __name__ == '__main__':
    unittest.main()
