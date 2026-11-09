"""Static and runtime pointer casts preserve callbacks and shared storage."""

import unittest

from pythoc import (
    array, bool, compile, extern, func, i32, nullptr, ptr, static, void,
)


@extern(lib='c')
def abs(value: i32) -> i32:
    ...


@compile
class Entry:
    tag: i32
    callback: ptr[void]
    data: ptr[void]


@compile
class Table:
    entries: array[Entry, 3]


@compile
class Data:
    counter: static[i32] = 23


@compile
def plus(value: i32) -> i32:
    return value + 3


@compile
def minus(value: i32) -> i32:
    return value - 2


@compile
def get_table() -> ptr[Table]:
    table: static[Table] = (
        ((1, plus, ptr(Data.counter)),
         (2, ptr[void](minus), ptr[void](ptr(Data.counter))),
         (0, nullptr, nullptr)),
    )
    return ptr(table)


@compile
def call_table(index: i32, value: i32) -> i32:
    table = get_table()
    callback: func[i32, i32] = func[i32, i32](table.entries[index].callback)
    return callback(value)


@compile
def table_data_aliases() -> bool:
    table = get_table()
    first = ptr[i32](table.entries[0].data)
    second = ptr[i32](table.entries[1].data)
    return first == second and first == ptr(Data.counter)


@compile
def write_table_data(delta: i32) -> i32:
    table = get_table()
    Data.counter = 23
    ptr[i32](table.entries[0].data)[0] += delta
    return ptr[i32](table.entries[1].data)[0]


@compile
def table_null_entry() -> bool:
    entry = get_table().entries[2]
    return entry.callback == nullptr and entry.data == nullptr


@compile
def use_data(delta: i32) -> i32:
    alias: static[ptr[void]] = ptr[void](ptr(Data.counter))
    Data.counter = 23
    ptr[i32](alias)[0] += delta
    return Data.counter


@compile
def use_external(value: i32) -> i32:
    callback: static[ptr[void]] = ptr[void](abs)
    return func[i32, i32](callback)(value)


@compile
def use_runtime(initial: i32) -> i32:
    value: i32 = initial
    erased: ptr[void] = ptr[void](ptr(value))
    restored: ptr[i32] = ptr[i32](erased)
    restored[0] += 5
    return value


class ConstantPointerCastsTest(unittest.TestCase):
    def test_nested_table_retains_function_addresses(self):
        for value in (-7, 0, 7, 31):
            with self.subTest(value=value):
                self.assertEqual(call_table(0, value), value + 3)
                self.assertEqual(call_table(1, value), value - 2)

    def test_nested_table_retains_shared_data_addresses(self):
        self.assertTrue(table_data_aliases())
        for delta in (-5, 0, 5):
            with self.subTest(delta=delta):
                self.assertEqual(write_table_data(delta), 23 + delta)

    def test_nested_table_retains_null_addresses(self):
        self.assertTrue(table_null_entry())

    def test_static_data_alias_writes_the_original_variable(self):
        for delta in (-5, 0, 5):
            with self.subTest(delta=delta):
                self.assertEqual(use_data(delta), 23 + delta)

    def test_external_function_address_is_a_static_initializer(self):
        for value, expected in ((-9, 9), (0, 0), (9, 9)):
            with self.subTest(value=value):
                self.assertEqual(use_external(value), expected)

    def test_runtime_conversion_writes_the_original_variable(self):
        for value in (-7, 0, 23):
            with self.subTest(value=value):
                self.assertEqual(use_runtime(value), value + 5)


if __name__ == '__main__':
    unittest.main()
