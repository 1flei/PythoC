"""Ordinary loops compose continuations without special effect syntax."""

from types import SimpleNamespace
import unittest

from pythoc import compile, effect, i32, ptr
from pythoc.build.output_manager import clear_failed_group, flush_all_pending_outputs
from test.utils.test_utils import DeferredTestCase


@compile
def consecutive(value: i32) -> i32:
    yield value
    yield value + 1


effect.default(composition_stream=SimpleNamespace(values=consecutive))


@compile
def pairs(first: i32, second: i32) -> i32:
    for left in effect.composition_stream.values(first):
        for right in effect.composition_stream.values(second):
            yield left + right


@compile
def consume_pairs() -> i32:
    trace: i32 = 0
    for value in pairs(1, 10):
        trace = trace * 100 + value
    return trace


@compile
def maybe(value: i32) -> i32:
    if value >= 0:
        yield value


@compile
def mark(trace: ptr[i32], digit: i32, value: i32) -> i32:
    trace[0] = trace[0] * 10 + digit
    return value


@compile
def traced_pairs(trace: ptr[i32], first: i32) -> i32:
    for left in maybe(mark(trace, 1, first)):
        for right in maybe(mark(trace, 2, 4)):
            yield left + right


@compile
def consume_trace(first: i32) -> i32:
    trace: i32 = 0
    result: i32 = -1
    for value in traced_pairs(ptr(trace), first):
        result = value
    return result * 100 + trace


@compile
def increment(value: i32) -> i32:
    return value + 1


def make_namespace_producer(effect):
    @compile
    def producer(value: i32) -> i32:
        yield effect.perform(value)

    return producer


ordinary_producer = make_namespace_producer(SimpleNamespace(perform=increment))


@compile
def consume_ordinary_namespace() -> i32:
    total: i32 = 0
    for value in ordinary_producer(4):
        total += value
    return total


class TestYieldComposition(DeferredTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        flush_all_pending_outputs()

    def test_nested_effect_sources_preserve_all_yields_in_order(self):
        self.assertEqual(consume_pairs(), 11121213)

    def test_arguments_are_evaluated_once_in_order(self):
        self.assertEqual(consume_trace(3), 712)

    def test_short_circuit_skips_the_later_call(self):
        self.assertEqual(consume_trace(-1), -99)

    def test_perform_is_an_ordinary_attribute_of_user_namespaces(self):
        self.assertEqual(consume_ordinary_namespace(), 5)

    def assert_compile_error(self, producer, exception, diagnostic):
        @compile(suffix=producer.__name__)
        def consumer() -> i32:
            total: i32 = 0
            for item in producer(3):
                total += item
            return total

        try:
            with self.assertRaisesRegex(exception, diagnostic):
                flush_all_pending_outputs()
        finally:
            clear_failed_group(consumer._binding.group_key)

    def test_retired_assignment_fails_during_compilation(self):
        @compile
        def assignment_producer(value: i32) -> i32:
            item = yield effect.perform(maybe, value)
            yield item + 1

        self.assert_compile_error(
            assignment_producer, SyntaxError,
            'Unsupported Python syntax: Yield',
        )

    def test_retired_statement_fails_during_compilation(self):
        @compile
        def statement_producer(value: i32) -> i32:
            yield effect.perform(maybe, value)

        self.assert_compile_error(
            statement_producer, NotImplementedError,
            "Cannot call Python object 'EffectNamespace' in compiled code",
        )


if __name__ == '__main__':
    unittest.main()
