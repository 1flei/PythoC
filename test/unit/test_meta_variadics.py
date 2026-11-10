"""Quote preserves definition bindings and substitutes explicit syntax holes."""

import ast
import copy
import unittest

from pythoc import i32, meta


definition_value = 7


@meta.quote
def invoke(callee, /, *args, **kwargs):
    return callee(*args, **kwargs)


@meta.quote
def expression(value):
    return value


class TestQuoteVariadics(unittest.TestCase):
    def source(self, fragment):
        return ast.unparse(fragment.stmt)

    def test_collects_and_expands_arguments(self):
        fragment = invoke('f', 'x', expression(2), scale=3)
        self.assertEqual(self.source(fragment), 'return f(x, 2, scale=3)')

    def test_empty_collectors(self):
        self.assertEqual(self.source(invoke('f')), 'return f()')

    def test_keyword_names_do_not_bind_template_receiver(self):
        fragment = invoke('f', self='x', callee='y')
        self.assertEqual(self.source(fragment), 'return f(self=x, callee=y)')

    def test_keyword_order_is_preserved(self):
        fragment = invoke('f', **{'second': 'b', 'first': 'a'})
        self.assertEqual([kw.arg for kw in fragment.expr.keywords],
                         ['second', 'first'])

    def test_signature_defaults_and_keyword_only_parameters(self):
        @meta.quote
        def scaled(callee, /, *args, scale=2, **kwargs):
            return callee(*args, scale=scale, **kwargs)

        self.assertEqual(self.source(scaled('f', 'x', label=meta.const('ok'))),
                         "return f(x, scale=2, label='ok')")
        self.assertEqual(self.source(scaled('f', scale=4)), 'return f(scale=4)')

    def test_required_holes_must_be_supplied(self):
        @meta.quote
        def add(left, right):
            return left + right

        with self.assertRaisesRegex(TypeError, 'right'):
            add(1)

    def test_signature_errors_follow_python(self):
        @meta.quote
        def fixed(value, /, *, scale):
            return value * scale

        for args, kwargs in [((1, 2), {}), ((), {'value': 1}),
                             ((1,), {'unknown': 2})]:
            with self.subTest(args=args, kwargs=kwargs):
                with self.assertRaises(TypeError):
                    fixed(*args, **kwargs)

    def test_explicit_container_bindings(self):
        @meta.quote
        def spread(callee, args, kwargs):
            return callee(*args, **kwargs)

        for args in (['x', 2], ('x', 2)):
            self.assertEqual(self.source(spread('f', args, {'size': 3})),
                             'return f(x, 2, size=3)')

    def test_mixed_splices(self):
        @meta.quote
        def mixed(callee, first, second, options):
            return callee(0, *first, 3, *second, fixed=4, **options)

        fragment = mixed('f', [1, 2], (5,), {'extra': 6})
        self.assertEqual(self.source(fragment),
                         'return f(0, 1, 2, 3, 5, fixed=4, extra=6)')

    def test_nested_calls(self):
        @meta.quote
        def nested(outer, inner, /, *args, **kwargs):
            return outer(inner(*args, **kwargs))

        self.assertEqual(self.source(nested('outer', 'inner', 'x', scale=2)),
                         'return outer(inner(x, scale=2))')

    def test_replacement_names_are_not_substituted_again(self):
        self.assertEqual(self.source(invoke('f', 'callee')),
                         'return f(callee)')

    def test_fragments_are_not_mutated(self):
        argument = expression('value')
        before = ast.dump(argument.expr, include_attributes=False)
        invoke('first', argument)
        invoke('second', argument)
        self.assertEqual(ast.dump(argument.expr, include_attributes=False), before)

    def test_duplicate_keywords_are_rejected(self):
        @meta.quote
        def duplicate(options):
            return f(value=1, **options)

        with self.assertRaisesRegex(TypeError, 'value'):
            duplicate({'value': 2})

    def test_duplicates_between_mapping_splices_are_rejected(self):
        @meta.quote
        def duplicate(first, second):
            return f(**first, **second)

        with self.assertRaisesRegex(TypeError, 'value'):
            duplicate({'value': 1}, {'value': 2})

    def test_non_string_keyword_keys_are_rejected(self):
        @meta.quote
        def spread(options):
            return f(**options)

        for key in (1, None):
            with self.subTest(key=key):
                with self.assertRaises(TypeError):
                    spread({key: 1})

    def test_arbitrary_string_keyword_keys_are_preserved(self):
        fragment = invoke('f', **{'': 1, 'not-valid': 2, 'class': 3})
        self.assertEqual([entry.arg for entry in fragment.expr.keywords],
                         ['', 'not-valid', 'class'])

    def test_non_expression_fragment_is_rejected(self):
        @meta.quote
        def statements(value):
            temporary = value
            return temporary

        with self.assertRaises(TypeError):
            invoke('f', statements(1))

    def test_nested_sequence_is_not_flattened(self):
        with self.assertRaises(TypeError):
            invoke('f', [1, 2])

    def test_ordinary_expression_position_does_not_splice(self):
        @meta.quote
        def ordinary(args):
            return f(args)

        with self.assertRaises(TypeError):
            ordinary([1, 2])

    def test_explicit_expression_unpacking_is_preserved(self):
        @meta.quote
        def spread(callee, args, kwargs):
            return callee(*args, **kwargs)

        self.assertEqual(self.source(spread('f', 'runtime_args',
                                            expression('runtime_kwargs'))),
                         'return f(*runtime_args, **runtime_kwargs)')

    def test_string_literals_keep_explicit_const_rule(self):
        fragment = invoke('f', 'variable', label=meta.const('literal'))
        self.assertEqual(self.source(fragment),
                         "return f(variable, label='literal')")


class TestQuoteBindings(unittest.TestCase):
    def make_piece(self, value):
        def helper():
            return value

        @meta.quote
        def piece():
            return helper()

        return piece()

    def test_definition_local_symbol_is_bound(self):
        fragment = self.make_piece(1)
        self.assertEqual(fragment.expr.func._definition_symbol.value(), 1)

    def test_definition_global_symbol_is_bound(self):
        @meta.quote
        def piece():
            return definition_value

        fragment = piece()
        self.assertEqual(fragment.expr._definition_symbol.value, 7)

    def test_explicit_object_keeps_its_identity(self):
        fragment = expression(i32)
        self.assertIs(fragment.expr._definition_symbol.value, i32)

    def test_fragments_keep_distinct_bindings_for_the_same_name(self):
        fragment = invoke('f', self.make_piece(1), value=self.make_piece(2))
        first = fragment.expr.args[0].func._definition_symbol.value
        second = fragment.expr.keywords[0].value.func._definition_symbol.value
        self.assertIsNot(first, second)
        self.assertEqual((first(), second()), (1, 2))

    def test_nested_defaults_and_annotations_keep_definition_bindings(self):
        SomeType = i32
        ResultType = i32

        def helper():
            return 41

        @meta.quote
        def piece():
            def inner(*, value: SomeType = helper()) -> ResultType:
                return value
            return inner()

        fragment = piece()
        inner = fragment.stmts[0]
        self.assertIs(inner.args.kwonlyargs[0].annotation._definition_symbol.value,
                      SomeType)
        self.assertIs(inner.args.kw_defaults[0].func._definition_symbol.value, helper)
        self.assertIs(inner.returns._definition_symbol.value, ResultType)

    def test_ast_copy_preserves_binding_identity(self):
        fragment = self.make_piece(1)
        original = fragment.expr.func._definition_symbol
        self.assertIs(copy.deepcopy(fragment.expr).func._definition_symbol, original)

    def test_nested_parameter_does_not_become_a_template_hole(self):
        @meta.quote
        def piece(value):
            return (lambda value: value + 1)(value)

        call = piece(41).expr
        self.assertEqual(call.func.body.left.id, 'value')
        self.assertEqual(call.args[0].value, 41)

    def test_dead_branch_has_no_binding_conflict(self):
        def helper():
            return 1

        @meta.quote
        def choose(flag, body):
            if flag:
                return helper()
            else:
                return body

        fragment = choose(False, self.make_piece(2))
        self.assertEqual(fragment.expr.func._definition_symbol.value(), 2)

    def test_statement_splices_remain_syntax(self):
        @meta.quote
        def block(body):
            body

        original = self.make_piece(1)
        before = ast.dump(original.stmt)
        fragment = block([original])
        self.assertIs(fragment.expr.func._definition_symbol,
                      original.expr.func._definition_symbol)
        self.assertEqual(ast.dump(original.stmt), before)


if __name__ == '__main__':
    unittest.main()
