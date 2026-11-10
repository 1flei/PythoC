"""Instantiated captures preserve Python constants and move linear state."""

import unittest

from pythoc import compile, consume, i32, linear, ptr, struct, void
from pythoc import instantiate
from test.utils.test_utils import DeferredTestCase, expect_error


@compile
def type_capture() -> i32:
    Record = struct[value: i32]

    def work(x: i32) -> i32:
        value: Record
        value.value = x
        return value.value + 1

    closure = instantiate(work)
    state = closure.init()
    return closure.call(ptr(state), 41)


@compile
def none_capture() -> i32:
    captured = None

    def work() -> i32:
        if captured is None:
            return 42
        return 0

    closure = instantiate(work)
    state = closure.init()
    return closure.call(ptr(state))


@compile
def unused_parameter_capture() -> i32:
    captured = 10

    def work(__pc_capture_captured: i32) -> i32:
        return captured

    closure = instantiate(work)
    state = closure.init()
    return closure.call(ptr(state), 99)


@compile
def transfer_capture() -> i32:
    token = linear()

    def work() -> i32:
        consume(token)
        return 42

    closure = instantiate(work)
    state = closure.init()
    consume(state.token)
    return 42


@expect_error(['consumed'], suffix='capture_source_reuse')
def error_source_reuse():
    @compile(suffix='capture_source_reuse')
    def invalid() -> void:
        token = linear()

        def work() -> void:
            consume(token)

        closure = instantiate(work)
        state = closure.init()
        consume(state.token)
        consume(token)


@expect_error(['consumed'], suffix='capture_init_twice')
def error_init_twice():
    @compile(suffix='capture_init_twice')
    def invalid() -> void:
        token = linear()

        def work() -> void:
            consume(token)

        closure = instantiate(work)
        first = closure.init()
        second = closure.init()
        consume(first.token)
        consume(second.token)


class InstantiateCaptureTest(DeferredTestCase):
    def test_none_capture(self):
        self.assertEqual(none_capture(), 42)

    def test_unused_parameter_cannot_shadow_capture(self):
        self.assertEqual(unused_parameter_capture(), 10)

    def test_local_type_capture(self):
        self.assertEqual(type_capture(), 42)

    def test_init_moves_linear_capture_into_state(self):
        self.assertEqual(transfer_capture(), 42)

    def test_source_cannot_be_reused_after_init(self):
        passed, message = error_source_reuse()
        self.assertTrue(passed, message)

    def test_linear_capture_cannot_be_initialized_twice(self):
        passed, message = error_init_twice()
        self.assertTrue(passed, message)


if __name__ == '__main__':
    unittest.main()
