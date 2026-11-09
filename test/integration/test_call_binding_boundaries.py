"""Call binding preserves parameter kinds and rejects owned defaults."""

import unittest

from pythoc import compile, consume, i32, linear, ptr, struct
from pythoc.build.output_manager import flush_all_pending_outputs
from pythoc.session import CompileSession
from test.utils.test_utils import DeferredTestCase


@compile
def keyword_values(a: i32, *, b: i32 = 8) -> i32:
    yield a * 10 + b


@compile
def collector_values(a: i32, /, *items: struct[i32], b: i32 = 8,
                     **named: struct['a': i32]) -> i32:
    yield a * 1000 + items[0] * 100 + b * 10 + named.a


@compile
def keyword_binding() -> i32:
    total: i32 = 0
    for value in keyword_values(1):
        total += value
    for value in keyword_values(2, b=3):
        total += value
    return total


@compile
def collector_binding() -> i32:
    total: i32 = 0
    for value in collector_values(1, 2, a=4, b=3):
        total += value
    return total


@compile
def closure_binding() -> i32:
    count: i32 = 0

    def step(value: ptr[i32]) -> i32:
        value[0] += 1
        return value[0]

    def plain(a: i32, /, *, b: i32 = step(ptr(count))) -> i32:
        return a * 10 + b

    def producer(a: i32, /, *, b: i32 = step(ptr(count))) -> i32:
        yield a * 10 + b

    total = plain(1) + plain(2, b=3)
    for value in producer(4):
        total += value
    return total * 10 + count


@compile
def explicit_owned_arguments() -> i32:
    def take(token: linear) -> i32:
        consume(token)
        return 1

    def take_pair(*tokens: struct[linear, linear]) -> i32:
        consume(tokens[0])
        consume(tokens[1])
        return 2

    return take(linear()) + take(linear()) + take_pair(linear(), linear())


@compile
def copyable_tuple_defaults() -> i32:
    value: i32 = 7

    def read(item: struct[i32] = (value,)) -> i32:
        return item[0]

    value = 9
    return read() + read()


class CallBindingBoundariesTest(DeferredTestCase):
    def test_yield_keyword_only_defaults_and_overrides(self):
        self.assertEqual(keyword_binding(), 41)

    def test_collectors_preserve_positional_and_keyword_only_parameters(self):
        self.assertEqual(collector_binding(), 1234)

    def test_plain_and_yield_closures_share_binding_and_eager_defaults(self):
        self.assertEqual(closure_binding(), 762)

    def test_owned_values_can_be_passed_explicitly(self):
        self.assertEqual(explicit_owned_arguments(), 4)

    def test_copyable_aggregate_defaults_remain_reusable_snapshots(self):
        self.assertEqual(copyable_tuple_defaults(), 14)

    def test_rejects_closure_owned_default(self):
        with CompileSession(), self.assertRaisesRegex(
                TypeError, 'linear default arguments'):
            @compile
            def main() -> i32:
                token = linear()

                def take(value: linear = token) -> i32:
                    consume(value)
                    return 1

                return take() + take()

            flush_all_pending_outputs()

    def test_rejects_lambda_owned_default(self):
        with CompileSession(), self.assertRaisesRegex(
                TypeError, 'linear default arguments'):
            @compile
            def main() -> i32:
                token = linear()
                take = lambda value=token: consume(value)
                take()
                take()
                return 2

            flush_all_pending_outputs()

    def test_rejects_yield_owned_default(self):
        with CompileSession(), self.assertRaisesRegex(
                TypeError, 'linear default arguments'):
            @compile
            def main() -> i32:
                token = linear()

                def take(value: linear = token) -> i32:
                    consume(value)
                    yield 1

                total: i32 = 0
                for value in take():
                    total += value
                for value in take():
                    total += value
                return total

            flush_all_pending_outputs()

    def test_rejects_closure_owned_tuple_default(self):
        with CompileSession(), self.assertRaisesRegex(
                TypeError, 'linear default arguments'):
            @compile
            def main() -> i32:
                token = linear()

                def take(value: struct[linear] = (token,)) -> i32:
                    consume(value[0])
                    return 1

                return take() + take()

            flush_all_pending_outputs()

    def test_rejects_closure_owned_record_default(self):
        with CompileSession(), self.assertRaisesRegex(
                TypeError, 'linear default arguments'):
            @compile
            class Owner:
                value: i32
                token: linear

            @compile
            def main() -> i32:
                owner: Owner = (7, linear())

                def take(value: Owner = owner) -> i32:
                    consume(value.token)
                    return value.value

                return take() + take()

            flush_all_pending_outputs()

    def test_rejects_yield_keyword_only_positional(self):
        with CompileSession(), self.assertRaisesRegex(
                TypeError, 'too many positional'):
            @compile
            def values(a: i32, *, b: i32 = 8) -> i32:
                yield a * 10 + b

            @compile
            def main() -> i32:
                total: i32 = 0
                for value in values(1, 2):
                    total += value
                return total

            flush_all_pending_outputs()

    def test_rejects_yield_positional_only_keyword(self):
        with CompileSession(), self.assertRaisesRegex(
                TypeError, 'positional-only'):
            @compile
            def values(a: i32, /) -> i32:
                yield a

            @compile
            def main() -> i32:
                total: i32 = 0
                for value in values(a=1):
                    total += value
                return total

            flush_all_pending_outputs()

    def test_rejects_yield_missing_keyword_only(self):
        with CompileSession(), self.assertRaisesRegex(
                TypeError, 'missing required'):
            @compile
            def values(a: i32, *, b: i32) -> i32:
                yield a + b

            @compile
            def main() -> i32:
                total: i32 = 0
                for value in values(1):
                    total += value
                return total

            flush_all_pending_outputs()


if __name__ == '__main__':
    unittest.main()
