"""Native and compile-time arguments are evaluated once in source order."""

import unittest
from unittest.mock import patch

from pythoc import (
    compile, i32, inline, label, llvm_asm, offsetof, sizeof, static, struct,
    typeof, u64, va_arg, va_end, va_start, void,
)
from pythoc.build.output_manager import flush_all_pending_outputs
from test.utils.test_utils import DeferredTestCase


host_calls = []
host_keyword_calls = []
Record = struct['first': i32, 'second': i32]


def make_type(label):
    host_calls.append(label)
    return i32


def make_record(label):
    host_calls.append(label)
    return Record


def make_value():
    host_calls.append('value')
    return 4


def get_label():
    host_calls.append('label')
    return label


def host_mark(value):
    host_keyword_calls.append(value)
    return value


def host_combine(a, b):
    return a * 10 + b


def host_bool():
    host_calls.append('bool')
    return True


def host_void():
    host_calls.append('void')
    return void


@compile
class EvaluationState:
    calls: static[i32]


@compile
def bump() -> i32:
    EvaluationState.calls += 1
    return EvaluationState.calls


@compile
def count() -> i32:
    return EvaluationState.calls


@compile
def typeof_native() -> i32:
    T = typeof(bump())
    return EvaluationState.calls


@compile
def typeof_type() -> u64:
    T = typeof(make_type('typeof_type'))
    return sizeof(T)


@compile
def typeof_host_value() -> i32:
    T = typeof(make_value())
    return 4


@compile
def sizeof_type() -> u64:
    return sizeof(make_type('sizeof_type'))


@compile
def offsetof_type() -> u64:
    return offsetof(make_record('offsetof_type'), 'second')


@compile
def va_arg_type_variadic(n: i32, *args) -> i32:
    ap = va_start()
    value = va_arg(ap, make_type('va_arg_type'))
    va_end(ap)
    return value


@compile
def va_arg_type() -> i32:
    return va_arg_type_variadic(1, i32(42))


@compile
def with_callee() -> i32:
    with get_label()('scope'):
        value: i32 = 4
    return 4


@compile
def cast_control() -> i32:
    return i32(bump())


@inline
def inline_control_repeat(value: i32) -> i32:
    return value + value


@compile
def inline_control() -> i32:
    return inline_control_repeat(bump())


@compile
def yield_control_repeat(value: i32) -> i32:
    yield value + value


@compile
def yield_control() -> i32:
    for value in yield_control_repeat(bump()):
        return value


@compile
def native_keywords_combine(a: i32, b: i32) -> i32:
    return a * 10 + b


@compile
def native_keywords() -> i32:
    return native_keywords_combine(b=bump(), a=bump())


@inline
def inline_keywords_combine(a, b):
    return a * 10 + b


@compile
def inline_keywords() -> i32:
    return inline_keywords_combine(b=bump(), a=bump())


@compile
def python_keywords() -> i32:
    return host_combine(host_mark(1), b=host_mark(2))


@compile
def asm_keywords() -> i32:
    llvm_asm('', 'r,r', side_effect=host_bool(),
             inputs=[bump(), bump()], ret_type=host_void())
    return EvaluationState.calls


@compile
def nested_type_queries() -> u64:
    return sizeof(typeof(bump()))


class Affine:
    def __init__(self, bias):
        self.bias = bias

    @inline
    def __call__(self, value, *, scale=2):
        return self.bias + value * scale

    @inline
    def apply(self, value):
        return self(value, scale=3)

    @inline(method=True)
    def explicit_apply(self, value):
        return self.bias + value * 3


affine = Affine(1)


@compile
def inline_callable() -> i32:
    return affine(bump(), scale=bump()) + affine.apply(bump())


@compile
def explicit_inline_method() -> i32:
    return affine.explicit_apply(bump())


@compile
def reset_calls() -> None:
    EvaluationState.calls = 0


class CallEvaluationTest(DeferredTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Host-side factory counts require code generation even on warm runs.
        with patch('pythoc.build.cache.BuildCache.check_obj_uptodate',
                   return_value=False):
            flush_all_pending_outputs()

    def setUp(self):
        reset_calls()

    def observe(self, function, value, calls):
        self.assertEqual(function(), value)
        self.assertEqual(count(), calls)

    def test_typeof_evaluates_native_operand_once(self):
        self.observe(typeof_native, 1, 1)

    def test_typeof_evaluates_type_factory_once(self):
        self.observe(typeof_type, 4, 0)
        self.assertEqual(host_calls.count('typeof_type'), 1)

    def test_typeof_evaluates_python_value_once(self):
        self.observe(typeof_host_value, 4, 0)
        self.assertEqual(host_calls.count('value'), 1)

    def test_sizeof_evaluates_type_factory_once(self):
        self.observe(sizeof_type, 4, 0)
        self.assertEqual(host_calls.count('sizeof_type'), 1)

    def test_offsetof_evaluates_type_factory_once(self):
        self.observe(offsetof_type, 4, 0)
        self.assertEqual(host_calls.count('offsetof_type'), 1)

    def test_va_arg_evaluates_type_factory_once(self):
        self.observe(va_arg_type, 42, 0)
        self.assertEqual(host_calls.count('va_arg_type'), 1)

    def test_with_resolves_its_callee_once(self):
        self.observe(with_callee, 4, 0)
        self.assertEqual(host_calls.count('label'), 1)

    def test_cast_evaluates_its_argument_once(self):
        self.observe(cast_control, 1, 1)

    def test_repeated_inline_parameter_does_not_repeat_the_argument(self):
        self.observe(inline_control, 2, 1)

    def test_repeated_yield_parameter_does_not_repeat_the_argument(self):
        self.observe(yield_control, 2, 1)

    def test_native_keywords_preserve_source_evaluation_order(self):
        self.observe(native_keywords, 21, 2)

    def test_inline_keywords_preserve_source_evaluation_order(self):
        self.observe(inline_keywords, 21, 2)

    def test_python_keywords_preserve_source_evaluation_order(self):
        self.observe(python_keywords, 12, 0)
        self.assertEqual(host_keyword_calls, [1, 2])

    def test_asm_keywords_evaluate_each_operand_and_factory_once(self):
        self.observe(asm_keywords, 2, 2)
        self.assertEqual(host_calls.count('bool'), 1)
        self.assertEqual(host_calls.count('void'), 1)

    def test_nested_type_queries_do_not_repeat_the_operand(self):
        self.observe(nested_type_queries, 4, 1)

    def test_inline_callable_and_bound_method_keep_argument_values(self):
        self.observe(inline_callable, 13, 3)

    def test_explicit_inline_method_keeps_self_as_a_python_value(self):
        self.observe(explicit_inline_method, 4, 1)


if __name__ == '__main__':
    unittest.main()
