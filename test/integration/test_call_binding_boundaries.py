"""Call binding preserves parameter kinds and rejects owned defaults."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pythoc
from pythoc import compile, consume, i32, linear, ptr, struct


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


class CallBindingBoundariesTest(unittest.TestCase):
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

    def test_invalid_calls_and_owned_defaults_are_rejected(self):
        fixtures = [
            ('closure_owned_default.py', 'linear default arguments'),
            ('lambda_owned_default.py', 'linear default arguments'),
            ('yield_owned_default.py', 'linear default arguments'),
            ('closure_owned_tuple_default.py', 'linear default arguments'),
            ('closure_owned_record_default.py', 'linear default arguments'),
            ('yield_keyword_only_positional.py', 'too many positional'),
            ('yield_positional_only_keyword.py', 'positional-only'),
            ('yield_missing_keyword_only.py', 'missing required'),
        ]
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join([
            str(Path(pythoc.__file__).resolve().parent.parent),
            env.get('PYTHONPATH', ''),
        ])
        env['PC_RAISE_ON_ERROR'] = '1'
        directory = Path(__file__).resolve().parent / 'fixtures'
        for fixture, expected_error in fixtures:
            with self.subTest(fixture=fixture):
                with tempfile.TemporaryDirectory(prefix='binding-') as temp:
                    result = subprocess.run(
                        [sys.executable, str(directory / fixture)], cwd=temp,
                        env=env, capture_output=True, text=True, timeout=120)
                output = result.stdout + result.stderr
                self.assertNotEqual(result.returncode, 0, output)
                self.assertIn(expected_error, output)


if __name__ == '__main__':
    unittest.main()
