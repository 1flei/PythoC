"""Returned record fields preserve pointer, array, and callback operations."""

import unittest

from pythoc import array, bool, compile, func, i32, ptr, sizeof, void


@compile
class ReturnedFields:
    pointer: ptr[i32]
    pointers: array[ptr[i32], 2]
    callback: func[i32, i32]


@compile
def add_one(value: i32) -> i32:
    return value + 1


@compile
def returned_fields(value: ptr[i32]) -> ReturnedFields:
    fields: ReturnedFields
    fields.pointer = value
    fields.pointers[0] = value
    fields.pointers[1] = value
    fields.callback = add_one
    return fields


@compile
def read_attribute(initial: i32) -> i32:
    value: i32 = initial
    selected = returned_fields(ptr(value)).pointer
    selected[0] += 1
    return value


@compile
def read_index(initial: i32) -> i32:
    value: i32 = initial
    selected = returned_fields(ptr(value))[0]
    return selected[0]


@compile
def read_array(initial: i32) -> i32:
    value: i32 = initial
    selected: array[ptr[i32], 2] = returned_fields(ptr(value)).pointers
    return selected[0][0] + selected[1][0]


@compile
def call_field(initial: i32) -> i32:
    value: i32 = initial
    return returned_fields(ptr(value)).callback(value)


@compile
def fields_have_pointer_layout() -> bool:
    return sizeof(ReturnedFields) == 4 * sizeof(ptr[void])


class RvalueFieldTypesTest(unittest.TestCase):
    def test_attribute_writes_the_original_variable(self):
        self.assertEqual(read_attribute(41), 42)

    def test_index_retains_the_declared_pointer_type(self):
        self.assertEqual(read_index(37), 37)

    def test_nested_pointer_array_retains_both_aliases(self):
        self.assertEqual(read_array(23), 46)

    def test_returned_callback_is_callable(self):
        self.assertEqual(call_field(41), 42)

    def test_record_uses_pointer_field_storage(self):
        self.assertTrue(fields_have_pointer_layout())


if __name__ == '__main__':
    unittest.main()
