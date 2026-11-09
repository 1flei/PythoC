"""Class-body methods capture effects at definition, like @compile functions."""

from types import SimpleNamespace
import unittest

from pythoc import compile, effect, enum, union, i32


effect.default(class_method_value=7)


@compile
def existing_function() -> i32:
    return i32(effect.class_method_value)


def make_classes(label):
    @compile(suffix=label)
    class Record:
        value: i32

        def read() -> i32:
            return i32(effect.class_method_value)

        def lexical() -> i32:
            return existing_function()

    @union(suffix=label)
    class Overlay:
        value: i32

        def read() -> i32:
            return i32(effect.class_method_value)

    @enum(suffix=label)
    class Choice:
        Item: i32 = 0

        def read() -> i32:
            return i32(effect.class_method_value)

    @compile(suffix=label)
    def call_methods() -> i32:
        return Record.read() + Overlay.read() + Choice.read()

    return SimpleNamespace(
        record=Record, overlay=Overlay, choice=Choice, call_methods=call_methods,
    )


default = make_classes('default_class_methods')
with effect(class_method_value=11, suffix='class_methods_first'):
    first = make_classes('bound_class_methods')
with effect(class_method_value=22, suffix='class_methods_second'):
    second = make_classes('bound_class_methods')


class TestClassMethodEffect(unittest.TestCase):
    def test_all_class_decorators_capture_effect(self):
        for definitions, value in ((default, 7), (first, 11), (second, 22)):
            with self.subTest(value=value):
                self.assertEqual(definitions.record.read(), value)
                self.assertEqual(definitions.overlay.read(), value)
                self.assertEqual(definitions.choice.read(), value)
                self.assertEqual(definitions.call_methods(), value * 3)

    def test_ordinary_calls_keep_existing_binding(self):
        self.assertEqual(first.record.lexical(), 7)
        self.assertEqual(second.record.lexical(), 7)
        self.assertEqual(existing_function(), 7)


if __name__ == '__main__':
    unittest.main()
