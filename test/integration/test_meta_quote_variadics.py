"""Variadic quote templates compile through the ordinary native call path."""

import ast
import unittest

from pythoc import compile, func, goto, i32, inline, label, meta, ptr, struct
from pythoc.session import CompileSession
from test.utils.test_utils import DeferredTestCase


@compile
def calculate(a: i32, b: i32, scale: i32) -> i32:
    return (a * 10 + b) * scale


@compile
def constant() -> i32:
    return 7


@compile
def weighted6(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32) -> i32:
    return a + b * 10 + c * 100 + d * 1000 + e * 10000 + f * 100000


@compile
def record(trace: ptr[i32], digit: i32) -> i32:
    trace[0] = trace[0] * 10 + digit
    return digit


@inline
def read_options(**options) -> i32:
    return i32(options[''] * 100 + options['class'] * 10 + options['not-valid'])


@meta.quote
def invoke(callee, /, *args, **kwargs):
    return callee(*args, **kwargs)


@meta.quote
def scaled(callee, /, *args, scale=3, **kwargs):
    return callee(*args, scale=scale, **kwargs)


@meta.quote
def recorded(digit, trace):
    return record(ptr(trace), i32(digit))


@meta.quote
def with_trace(body):
    trace: i32 = i32(0)
    result: i32 = body
    return result * 1000 + trace


@meta.quote
def spread(callee, arguments):
    return callee(*arguments)


native_call = meta.compile_generated(meta.func(
    'quoted_native_call', [('x', i32), ('y', i32)], i32,
    invoke(calculate, 'x', 'y', scale=3),
), user_globals={'calculate': calculate})
default_call = meta.compile_generated(meta.func(
    'quoted_default_call', [('x', i32), ('y', i32)], i32,
    scaled(calculate, 'x', 'y'),
), user_globals={'calculate': calculate})
empty_call = meta.compile_generated(meta.func(
    'quoted_empty_call', [], i32, invoke(constant),
), user_globals={'constant': constant})
many_args = meta.compile_generated(meta.func(
    'quoted_many_arguments', [], i32, invoke(weighted6, 1, 2, 3, 4, 5, 6),
), user_globals={'weighted6': weighted6})
ordered_call = meta.compile_generated(meta.func(
    'quoted_ordered_call', [], i32,
    with_trace(invoke(calculate, recorded(1, 'trace'), recorded(2, 'trace'),
                      scale=recorded(3, 'trace'))),
), user_globals={'calculate': calculate, 'record': record, 'ptr': ptr, 'i32': i32})
keyword_order = meta.compile_generated(meta.func(
    'quoted_keyword_order', [], i32,
    with_trace(invoke(calculate, **{
        'b': recorded(2, 'trace'), 'a': recorded(1, 'trace'),
        'scale': recorded(3, 'trace'),
    })),
), user_globals={'calculate': calculate, 'record': record, 'ptr': ptr, 'i32': i32})
indirect_call = meta.compile_generated(meta.func(
    'quoted_indirect_call',
    [('callback', func[i32, i32, i32, i32]), ('x', i32), ('y', i32)],
    i32, invoke('callback', 'x', 'y', 3),
))

ArgumentPack = struct[i32, i32, i32]
runtime_star = meta.compile_generated(meta.func(
    'quoted_runtime_star', [('arguments', ArgumentPack)], i32,
    spread(calculate, 'arguments'),
), user_globals={'calculate': calculate})

string_keys = meta.compile_generated(meta.func(
    'quoted_string_keyword_keys', [], i32,
    invoke(read_options, **{'': 1, 'class': 2, 'not-valid': 3}),
), user_globals={'read_options': read_options})


def make_symbolic_body(definition_value):
    @meta.quote
    def read():
        return definition_value

    return read()


symbolic_parameter = meta.compile_generated(meta.func(
    'quoted_symbolic_parameter', [('definition_value', i32)], i32,
    make_symbolic_body(7),
))
symbolic_global = meta.compile_generated(meta.func(
    'quoted_symbolic_global', [], i32, make_symbolic_body(7),
), user_globals={'definition_value': 41})


@meta.quote
def explicit_value(value):
    return value


explicit_parameter = meta.compile_generated(meta.func(
    'quoted_explicit_parameter', [('definition_value', i32)], i32,
    explicit_value('definition_value'),
))


def make_piece(value):
    def helper():
        return value

    @meta.quote
    def read():
        return helper()

    return read()


@meta.quote
def combine(left, right):
    return left * 10 + right


combined_bindings = meta.compile_generated(meta.func(
    'quoted_combined_bindings', [], i32,
    combine(make_piece(2), make_piece(3)).stmts,
))
extracted_expression = meta.compile_generated(meta.func(
    'quoted_extracted_expression', [], i32,
    [ast.Return(value=make_piece(7).expr)],
))


@meta.quote
def unresolved_value():
    return missing_definition_binding


@meta.quote
def forward_label_body():
    goto('done')
    with label('done'):
        return 42


forward_label = meta.compile_generated(meta.func(
    'quoted_forward_label', [], i32, forward_label_body(),
))


def make_none_body():
    captured = None

    @meta.quote
    def body():
        if captured is None:
            return 42
        return 0

    return body()


none_binding = meta.compile_generated(meta.func(
    'quoted_none_binding', [], i32, make_none_body(),
))


@compile
def run_indirect() -> i32:
    return indirect_call(calculate, 2, 4)


@compile
def run_runtime_star() -> i32:
    arguments: ArgumentPack = (1, 2, 3)
    return runtime_star(arguments)


@compile
def direct_order() -> i32:
    trace: i32 = 0
    result = calculate(record(ptr(trace), 1), record(ptr(trace), 2),
                       scale=record(ptr(trace), 3))
    return result * 1000 + trace


@compile
def direct_keyword_order() -> i32:
    trace: i32 = 0
    result = calculate(b=record(ptr(trace), 2), a=record(ptr(trace), 1),
                       scale=record(ptr(trace), 3))
    return result * 1000 + trace


class TestMetaQuoteVariadics(DeferredTestCase):
    def test_native_call_uses_the_bound_callee(self):
        for x, y in [(2, 4), (0, 7), (-3, 5)]:
            with self.subTest(x=x, y=y):
                self.assertEqual(native_call(x, y), (x * 10 + y) * 3)

    def test_template_default(self):
        self.assertEqual(default_call(2, 4), 72)

    def test_empty_arguments(self):
        self.assertEqual(empty_call(), 7)

    def test_many_arguments(self):
        self.assertEqual(many_args(), 654321)

    def test_evaluation_order_and_count(self):
        self.assertEqual(direct_order(), 36123)
        self.assertEqual(ordered_call(), 36123)

    def test_keyword_evaluation_precedes_parameter_binding(self):
        self.assertEqual(direct_keyword_order(), 36213)
        self.assertEqual(keyword_order(), 36213)

    def test_indirect_call(self):
        self.assertEqual(run_indirect(), 72)

    def test_runtime_unpacking_keeps_existing_call_lowering(self):
        self.assertEqual(run_runtime_star(), 36)

    def test_arbitrary_string_keyword_keys(self):
        self.assertEqual(string_keys(), 123)

    def test_definition_binding_survives_a_same_named_generated_parameter(self):
        self.assertEqual(symbolic_parameter(41), 7)

    def test_definition_binding_survives_same_named_compilation_globals(self):
        self.assertEqual(symbolic_global(), 7)

    def test_explicit_reference_uses_the_generated_parameter(self):
        self.assertEqual(explicit_parameter(41), 41)

    def test_same_named_helpers_keep_distinct_bindings_after_statement_cast(self):
        self.assertEqual(combined_bindings(), 23)

    def test_expression_cast_keeps_its_binding(self):
        self.assertEqual(extracted_expression(), 7)

    def test_bound_label_alias_is_reachable_after_a_forward_goto(self):
        self.assertEqual(forward_label(), 42)

    def test_none_is_a_bound_value(self):
        self.assertEqual(none_binding(), 42)

    def test_one_fragment_can_be_compiled_more_than_once(self):
        with CompileSession():
            fragment = make_piece(7)
            before = ast.dump(fragment.expr)
            first = meta.compile_generated(meta.func(
                'quoted_reused_first', [], i32, fragment,
            ), suffix='quote_reuse')
            self.assertEqual(ast.dump(fragment.expr), before)
            second = meta.compile_generated(meta.func(
                'quoted_reused_second', [], i32, fragment,
            ), suffix='quote_reuse')
            self.assertEqual(first(), 7)
            self.assertEqual(second(), 7)

    def test_unknown_definition_name_cannot_use_compilation_globals(self):
        with CompileSession(), self.assertRaisesRegex(NameError, 'missing_definition_binding'):
            missing = meta.compile_generated(meta.func(
                'quoted_missing_binding', [], i32, unresolved_value(),
            ), suffix='quote_missing_binding',
                user_globals={'missing_definition_binding': 41})
            missing()


if __name__ == '__main__':
    unittest.main()
