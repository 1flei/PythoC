"""Expansion aliases retain the ownership path of projected values."""

import unittest

from pythoc import compile, consume, i32, inline, linear, struct
from pythoc.build.output_manager import flush_all_pending_outputs
from pythoc.session import CompileSession
from test.utils.test_utils import DeferredTestCase


@compile
class Pair:
    first: linear
    second: linear


@compile
class Nested:
    pair: Pair
    tail: linear


@compile
def take_native(token: linear) -> i32:
    consume(token)
    return 7


@inline
def take_inline(token: linear) -> i32:
    consume(token)
    return 7


@inline
def return_inline(token: linear) -> linear:
    return token


@compile
def take_yield(token: linear) -> i32:
    consume(token)
    yield 7


@inline
def take_collector(*tokens: struct[linear]) -> i32:
    consume(tokens[0])
    return 7


@compile
def native_field() -> i32:
    owner: Pair = (linear(), linear())
    result = take_native(owner.first)
    consume(owner.second)
    return result


@compile
def inline_field() -> i32:
    owner: Pair = (linear(), linear())
    result = take_inline(owner.first)
    consume(owner.second)
    return result


@compile
def inline_nested_field() -> i32:
    owner: Nested = ((linear(), linear()), linear())
    result = take_inline(owner.pair.second)
    consume(owner.pair.first)
    consume(owner.tail)
    return result


@compile
def closure_field() -> i32:
    def take(token: linear) -> i32:
        consume(token)
        return 7

    owner: Pair = (linear(), linear())
    result = take(owner.first)
    consume(owner.second)
    return result


@compile
def yield_field() -> i32:
    owner: Pair = (linear(), linear())
    result: i32 = 0
    for value in take_yield(owner.first):
        result = value
    consume(owner.second)
    return result


@compile
def returned_field() -> i32:
    owner: Pair = (linear(), linear())
    token = return_inline(owner.first)
    consume(token)
    consume(owner.second)
    return 7


@compile
def collector_field() -> i32:
    owner: Pair = (linear(), linear())
    result = take_collector(owner.first)
    consume(owner.second)
    return result


@compile
def iteration_fields() -> i32:
    owner: Pair = (linear(), linear())
    result: i32 = 0
    for token in (owner.first, owner.second):
        consume(token)
        result += 1
    return result


class InlineOwnershipTest(DeferredTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        flush_all_pending_outputs()

    def test_inline_field_matches_native_call(self):
        self.assertEqual(native_field(), 7)
        self.assertEqual(inline_field(), 7)

    def test_nested_field_keeps_its_entire_path(self):
        self.assertEqual(inline_nested_field(), 7)

    def test_closure_argument_keeps_its_field_path(self):
        self.assertEqual(closure_field(), 7)

    def test_yield_argument_keeps_its_field_path(self):
        self.assertEqual(yield_field(), 7)

    def test_returned_field_acquires_its_own_owner(self):
        self.assertEqual(returned_field(), 7)

    def test_typed_collector_keeps_the_element_path(self):
        self.assertEqual(collector_field(), 7)

    def test_constant_iteration_transfers_each_field_once(self):
        self.assertEqual(iteration_fields(), 2)

    def test_inline_does_not_consume_the_sibling_field(self):
        with CompileSession(), self.assertRaisesRegex(RuntimeError, 'not consumed'):
            @compile(suffix='inline_owned_sibling_leak')
            def missing_sibling() -> i32:
                owner: Pair = (linear(), linear())
                return take_inline(owner.first)

            flush_all_pending_outputs()

    def test_consumed_source_field_cannot_be_reused(self):
        with CompileSession(), self.assertRaisesRegex(RuntimeError, 'already consumed'):
            @compile(suffix='inline_owned_field_reuse')
            def reused_field() -> i32:
                owner: Pair = (linear(), linear())
                result = take_inline(owner.first)
                consume(owner.first)
                consume(owner.second)
                return result

            flush_all_pending_outputs()

    def test_returned_field_cannot_be_consumed_twice(self):
        with CompileSession(), self.assertRaisesRegex(RuntimeError, 'already consumed'):
            @compile(suffix='inline_returned_field_reuse')
            def reused_result() -> i32:
                owner: Pair = (linear(), linear())
                token = return_inline(owner.first)
                consume(token)
                consume(token)
                consume(owner.second)
                return 7

            flush_all_pending_outputs()

    def test_iteration_cannot_transfer_the_same_field_twice(self):
        with CompileSession(), self.assertRaisesRegex(RuntimeError, 'already consumed'):
            @compile(suffix='iteration_owned_field_reuse')
            def repeated_field() -> i32:
                owner: Pair = (linear(), linear())
                for token in (owner.first, owner.first):
                    consume(token)
                consume(owner.second)
                return 7

            flush_all_pending_outputs()


if __name__ == '__main__':
    unittest.main()
