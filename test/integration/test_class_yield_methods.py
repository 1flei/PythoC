"""Class lookup preserves the native continuation callable protocol."""

import unittest

from pythoc import (
    compile, consume, defer, enum, func, i32, inline, linear, ptr,
)


@compile
class Cursor:
    value: i32
    closed: i32

    def values(self: ptr['Cursor']) -> i32:
        def close():
            self.closed = self.closed + 1
        defer(close)
        yield self.value
        yield self.value + 1

    @compile(suffix='explicit_cursor_yield')
    def explicitly_decorated(value: i32) -> i32:
        yield value

    def transfer(token: linear) -> linear:
        yield token

    @inline
    def offset(value: i32) -> i32:
        return value + 13


@enum(i32)
class Counter:
    Value: i32

    def values(value: i32) -> i32:
        yield value
        yield value + 1


def make_items(T):
    @compile(suffix=T)
    class Items:
        value: T

        def values(self: ptr['Items']) -> T:
            yield self.value
    return Items


IntItems = make_items(i32)


@compile
def consume_values(cursor: ptr[Cursor], first_only: i32) -> i32:
    result: i32 = 0
    for value in Cursor.values(cursor):
        result = result + value
        if first_only != 0:
            break
    return result


@compile
def observe_cleanup(first_only: i32) -> i32:
    cursor: Cursor
    cursor.value = 4
    cursor.closed = 0
    result: i32 = consume_values(ptr(cursor), first_only)
    return result * 10 + cursor.closed


@compile
def other_class_paths() -> i32:
    total: i32 = 0
    for value in Counter.values(2):
        total = total + value
    for value in Cursor.explicitly_decorated(7):
        total = total + value
    items: IntItems
    items.value = 9
    for value in IntItems.values(ptr(items)):
        total = total + value
    for token in Cursor.transfer(linear()):
        consume(token)
    return total


@compile
def inline_class_method() -> i32:
    return Cursor.offset(4)


class ClassYieldMethodTest(unittest.TestCase):
    def test_all_values_and_scope_cleanup(self):
        self.assertEqual(observe_cleanup(0), 91)

    def test_early_break_still_closes_scope(self):
        self.assertEqual(observe_cleanup(1), 41)

    def test_enum_explicit_generic_and_linear_paths(self):
        self.assertEqual(other_class_paths(), 21)

    def test_explicit_inline_method_is_callable(self):
        self.assertEqual(inline_class_method(), 17)

    def test_z_continuation_cannot_become_a_runtime_function(self):
        from pythoc.logger import set_raise_on_error
        set_raise_on_error(True)

        @compile(suffix='invalid_class_continuation_escape')
        def invalid_escape() -> i32:
            callback: func[ptr[Cursor], i32] = Cursor.values
            return 0

        with self.assertRaisesRegex(TypeError, 'Compile-level callable'):
            invalid_escape()


if __name__ == '__main__':
    unittest.main()
