"""Yield calls preserve argument binding, evaluation order and diagnostics."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pythoc
from pythoc import compile, consume, i32, linear, param, ptr, struct


@compile
def step(count: ptr[i32]) -> i32:
    count[0] += 1
    return count[0]


@compile
def digits(a: i32, b: i32 = 8) -> i32:
    yield a * 10 + b


@compile
def pair(count: ptr[i32]) -> struct[i32, i32]:
    return step(count), step(count)


@compile
def reordered_keywords() -> i32:
    count: i32 = 0
    total: i32 = 0
    for value in digits(b=step(ptr(count)), a=step(ptr(count))):
        total += value
    return total * 10 + count


@compile
def default_argument() -> i32:
    count: i32 = 0
    total: i32 = 0
    for value in digits(step(ptr(count))):
        total += value
    return total * 10 + count


@compile
def starred_arguments() -> i32:
    count: i32 = 0
    total: i32 = 0
    for value in digits(*pair(ptr(count))):
        total += value
    return total * 10 + count


@compile
def positional(*values: struct[i32, i32]) -> i32:
    yield values[0] * 10 + values[1]


@compile
def named(**values: struct['left': i32, 'right': i32]) -> i32:
    yield values.left * 10 + values.right


@compile
def typed_collectors() -> i32:
    count: i32 = 0
    total: i32 = 0
    for value in positional(step(ptr(count)), step(ptr(count))):
        total += value
    for value in named(right=step(ptr(count)), left=step(ptr(count))):
        total += value
    return total * 10 + count


@compile
def yield_closure_default() -> i32:
    count: i32 = 0

    def values(value: i32 = step(ptr(count))) -> i32:
        yield value

    total: i32 = 0
    for value in values():
        total += value
    for value in values():
        total += value
    return total * 10 + count


default_evaluations = []


def default():
    default_evaluations.append(1)
    return 8


@compile
def generic(T: param, value: 'T' = default()) -> 'T':
    yield value


generic_values = generic(i32)


@compile
def parametric_defaults() -> i32:
    total: i32 = 0
    for value in generic_values():
        total += value
    for value in generic_values(value=3):
        total += value
    return total


@compile
def plain_closure_default() -> i32:
    count: i32 = 0

    def value(item: i32 = step(ptr(count))) -> i32:
        return item

    return (value() + value()) * 10 + count


@compile
def linear_values(*tokens: struct[linear]) -> i32:
    consume(tokens[0])
    yield 7


@compile
def linear_collector() -> i32:
    token = linear()
    total: i32 = 0
    for value in linear_values(token):
        total += value
    return total


class YieldCallArgumentsTest(unittest.TestCase):
    def test_reordered_keywords_evaluate_once_in_source_order(self):
        self.assertEqual(reordered_keywords(), 212)

    def test_default_argument_is_bound_without_extra_evaluation(self):
        self.assertEqual(default_argument(), 181)

    def test_starred_expression_is_evaluated_once(self):
        self.assertEqual(starred_arguments(), 122)

    def test_typed_varargs_and_kwargs_keep_argument_order(self):
        self.assertEqual(typed_collectors(), 554)

    def test_yield_closure_default_is_eager_at_definition(self):
        self.assertEqual(yield_closure_default(), 21)

    def test_parametric_yield_keeps_definition_time_defaults(self):
        self.assertEqual(default_evaluations, [1])
        self.assertEqual(parametric_defaults(), 11)
        self.assertEqual(default_evaluations, [1])

    def test_plain_closure_default_is_also_eager_at_definition(self):
        self.assertEqual(plain_closure_default(), 21)

    def test_linear_typed_collector_transfers_ownership_once(self):
        self.assertEqual(linear_collector(), 7)

    def assert_compile_failure(self, fixture, expected_error):
        script = Path(__file__).resolve().parent / 'fixtures' / fixture
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join([
            str(Path(pythoc.__file__).resolve().parent.parent),
            env.get('PYTHONPATH', ''),
        ])
        env['PC_RAISE_ON_ERROR'] = '1'
        with tempfile.TemporaryDirectory(prefix='pythoc-yield-failure-') as temp:
            result = subprocess.run(
                [sys.executable, str(script)], cwd=temp, env=env,
                capture_output=True, text=True, timeout=120)
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, output)
        self.assertIn(expected_error, output)

    def test_linear_typed_collector_rejects_use_after_transfer(self):
        self.assert_compile_failure(
            'yield_linear_use_after_transfer.py', 'already consumed')

    def test_kwargs_splat_is_rejected_at_the_shared_call_boundary(self):
        self.assert_compile_failure(
            'yield_kwargs_splat.py',
            '**kwargs splat in calls is not supported')


if __name__ == '__main__':
    unittest.main()
