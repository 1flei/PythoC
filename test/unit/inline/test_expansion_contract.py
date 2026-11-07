"""Expansion boundaries keep values, identities, and names distinct."""

import ast
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from llvmlite import ir

from pythoc import linear
from pythoc._inline.arguments import bind_argument_values, bind_evaluated_call
from pythoc._inline.constant_loop_adapter import ConstantLoopAdapter
from pythoc._inline.exit_rules import ReturnExitRule
from pythoc._inline.kernel import MetaInlineRequest, expand_inline
from pythoc._inline.names import ExpansionNames
from pythoc._inline.perform_desugar import desugar_effect_performs
from pythoc._inline.renaming import ScopedVariableRenamer
from pythoc._inline.scope_analyzer import ScopeContext
from pythoc._inline.yield_state_machine import inline_yield_function_iterators
from pythoc.ast_visitor.base import LLVMIRVisitor
from pythoc.context import VariableInfo
from pythoc.valueref import wrap_python_constant, wrap_value


class EvaluatedArgumentsTest(unittest.TestCase):
    def setUp(self):
        self.visitor = LLVMIRVisitor()

    def test_bare_syntax_is_rejected_before_creating_aliases(self):
        before = set(self.visitor.scope_manager.get_all_visible())
        with self.assertRaisesRegex(TypeError, 'evaluated call arguments'):
            bind_argument_values(self.visitor, [wrap_python_constant(1),
                                                ast.Constant(value=2)])
        self.assertEqual(set(self.visitor.scope_manager.get_all_visible()), before)

    def test_explicit_syntax_carrier_remains_a_compile_time_value(self):
        syntax = ast.parse('effectful()', mode='eval').body
        name, = bind_argument_values(self.visitor, [wrap_python_constant(syntax)])
        actual = self.visitor.lookup_variable(name).value_ref.get_python_value()
        self.assertIs(actual, syntax)

    def test_linear_alias_preserves_the_original_cfg_owner(self):
        value = wrap_value(None, kind='value', type_hint=linear,
                           var_name='owner', linear_path=(), linear_owner_id=41)
        name, = bind_argument_values(self.visitor, [value])
        actual = self.visitor.lookup_variable(name).value_ref
        self.assertIs(actual, value)
        self.assertEqual(actual.linear_owner_id, 41)

    def test_alias_preserves_the_complete_field_origin(self):
        value = wrap_value(None, kind='value', type_hint=linear,
                           var_name='owner', linear_path=(1, 0),
                           linear_owner_id=41)
        name, = bind_argument_values(self.visitor, [value])
        actual = self.visitor.lookup_variable(name).value_ref
        self.assertEqual((actual.var_name, actual.linear_path,
                          actual.linear_owner_id), ('owner', (1, 0), 41))
        self.assertIs(actual, value)

    def test_bare_python_values_are_rejected_before_creating_aliases(self):
        before = set(self.visitor.scope_manager.get_all_visible())
        for value in (4, object(), None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, 'ValueRef'):
                    bind_argument_values(
                        self.visitor, [wrap_python_constant(1), value])
                self.assertEqual(
                    set(self.visitor.scope_manager.get_all_visible()), before)

    def test_reading_a_place_alias_preserves_the_source_owner_and_path(self):
        place = ir.Constant(linear.get_llvm_type().as_pointer(), None)
        value = wrap_value(place, kind='address', type_hint=linear,
                           address=place, var_name='owner',
                           linear_path=(1, 0), linear_owner_id=41)
        name, = bind_argument_values(self.visitor, [value])
        actual = self.visitor._bind_name_reference(value, name)
        self.assertEqual(actual.linear_owner_id, 41)
        self.assertEqual(actual.linear_path, (1, 0))
        self.assertIs(actual.require_place(), place)

    def test_owned_storage_gets_its_own_binding_identity(self):
        place = ir.Constant(linear.get_llvm_type().as_pointer(), None)
        value = wrap_value(place, kind='address', type_hint=linear,
                           address=place)
        variable = VariableInfo(name='local', value_ref=value, alloca=place)
        self.visitor.scope_manager.declare_variable(variable)
        actual = self.visitor._bind_name_reference(value, 'local')
        self.assertEqual(actual.linear_owner_id, variable.var_id)
        self.assertEqual(actual.linear_path, ())
        self.assertEqual(actual.var_name, 'local')

    def test_iteration_binding_retains_the_field_path(self):
        value = wrap_value(None, kind='value', type_hint=linear,
                           var_name='owner', linear_path=(1, 0),
                           linear_owner_id=41)
        actual = ConstantLoopAdapter(self.visitor)._element_to_valueref(value)
        self.assertEqual(actual.linear_owner_id, 41)
        self.assertEqual(actual.linear_path, (1, 0))

    def test_alias_avoids_identifiers_used_later_in_the_function(self):
        self.visitor.expansion_names.reserve_ast(ast.parse('_arg_inline = 99'))
        name, = bind_argument_values(self.visitor, [wrap_python_constant(4)])
        self.assertNotEqual(name, '_arg_inline')
        self.assertEqual(self.visitor.lookup_variable(name).value_ref.get_python_value(), 4)

    def test_alias_names_are_local_to_the_visitor(self):
        other = LLVMIRVisitor()
        with patch('pythoc.utils.get_next_id', side_effect=AssertionError):
            first = bind_argument_values(self.visitor, [wrap_python_constant(4)])
            second = bind_argument_values(other, [wrap_python_constant(5)])
        self.assertEqual(first, second)

    def test_fresh_aliases_skip_reserved_suffixes(self):
        names = ExpansionNames()
        names.reserve(['_arg_inline', '_arg_inline_2'])
        self.assertEqual(names.fresh('_arg_inline'), '_arg_inline_1')
        self.assertEqual(names.fresh('_arg_inline'), '_arg_inline_3')
        self.assertEqual(names.fresh('_arg_inline'), '_arg_inline_4')

    def test_binding_never_reuses_source_argument_or_keyword_syntax(self):
        source = ast.parse('f(effectful(), value=other())', mode='eval').body
        before = ast.dump(source)
        bound = bind_evaluated_call(self.visitor, [wrap_python_constant(4)], source)
        self.assertEqual(ast.dump(source), before)
        self.assertIsNot(bound, source)
        self.assertEqual(bound.keywords, [])
        self.assertTrue(all(isinstance(node, ast.Name) for node in bound.args))


class ExpansionKernelContractTest(unittest.TestCase):
    def request(self, args, available=()):
        source = ast.parse('def f(value):\n    return value + 1').body[0]
        return MetaInlineRequest(
            callee_ast=source, callee_globals={}, call_args=args,
            call_site=ast.parse('f(4)', mode='eval').body,
            caller_context=ScopeContext(set(available)),
            exit_rule=ReturnExitRule(result_var='answer'),
        )

    def test_missing_and_extra_arguments_are_both_rejected(self):
        for arguments in ([], [ast.Constant(value=1), ast.Constant(value=2)]):
            with self.subTest(count=len(arguments)):
                with self.assertRaisesRegex(TypeError, 'argument count mismatch'):
                    expand_inline(self.request(arguments))

    def test_generated_parameter_cannot_clobber_a_caller_binding(self):
        request = self.request([ast.Constant(value=4)], ['value_inline_7'])
        original = ast.dump(request.callee_ast)
        with patch('pythoc.utils.get_next_id', return_value=7):
            result = expand_inline(request)
        namespace = {
            '__pc_intrinsics': SimpleNamespace(move=lambda value: value),
            'value_inline_7': 99,
        }
        program = ast.fix_missing_locations(ast.Module(body=result.stmts, type_ignores=[]))
        exec(compile(program, '<expanded>', 'exec'), namespace)
        self.assertEqual(namespace['answer'], 5)
        self.assertEqual(namespace['value_inline_7'], 99)
        self.assertEqual(ast.dump(request.callee_ast), original)

    def test_reserved_callee_names_are_also_avoided(self):
        names = ExpansionNames()
        names.reserve_ast(ast.parse('value_inline_7 = 99'))
        request = self.request([ast.Constant(value=4)])
        request.names = names
        with patch('pythoc.utils.get_next_id', return_value=7):
            result = expand_inline(request)
        targets = [node.id for stmt in result.stmts for node in ast.walk(stmt)
                   if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)]
        self.assertNotIn('value_inline_7', targets)


class ScopedRenamingTest(unittest.TestCase):
    def test_parameters_shadow_while_defaults_use_the_outer_scope(self):
        source = ast.parse('def inner(value=value + 1):\n    return value').body[0]
        result = ScopedVariableRenamer({'value': 'outer_value'}).visit(source)
        self.assertEqual(result.args.defaults[0].left.id, 'outer_value')
        self.assertEqual(result.body[0].value.id, 'value')

    def test_annotated_locals_shadow_but_unannotated_assignments_capture(self):
        source = ast.parse('''
def inner():
    local: i32 = value
    value = 9
    return local + value
''').body[0]
        result = ScopedVariableRenamer({'local': 'outer_local',
                                        'value': 'outer_value'}).visit(source)
        self.assertEqual(result.body[0].target.id, 'local')
        self.assertEqual(result.body[0].value.id, 'outer_value')
        self.assertEqual(result.body[1].targets[0].id, 'outer_value')
        self.assertEqual(result.body[2].value.left.id, 'local')

    def test_lambda_masks_its_parameters_and_retains_outer_captures(self):
        source = ast.parse('lambda value: value + captured', mode='eval').body
        result = ScopedVariableRenamer({'value': 'outer_value',
                                        'captured': 'outer_captured'}).visit(source)
        self.assertEqual(result.body.left.id, 'value')
        self.assertEqual(result.body.right.id, 'outer_captured')


class SyntacticExpansionContractTest(unittest.TestCase):
    def test_constant_loop_alias_avoids_consumer_identifiers(self):
        visitor = LLVMIRVisitor()
        source = ast.parse('for value in [1]:\n    _iter_val_7_0 = value').body[0]
        with patch('pythoc._inline.constant_loop_adapter.get_next_id', return_value=7):
            result = ConstantLoopAdapter(visitor).transform_constant_loop(source, [1])
        self.assertNotIn('_iter_val_7_0', result.iter_var_names)
        self.assertEqual(len(result.iter_var_names), 1)
        self.assertIsNotNone(visitor.lookup_variable(result.iter_var_names[0]))

    def test_perform_temporary_does_not_capture_a_source_variable(self):
        source = ast.parse('''
def producer():
    _perform_val_7 = 99
    yield effect.perform(handler, input())
    yield _perform_val_7
''').body[0]
        original = ast.dump(source)
        with patch('pythoc._inline.perform_desugar.get_next_id', return_value=7):
            result = desugar_effect_performs(source)
        loop = result.body[1]
        self.assertNotEqual(loop.target.id, '_perform_val_7')
        self.assertEqual(ast.dump(source), original)

    def test_offline_expansion_rejects_keywords_instead_of_dropping_them(self):
        source = ast.parse('def source():\n    yield 1').body[0]
        wrapper = SimpleNamespace(_original_ast=source)
        consumer = ast.parse('''
def consumer():
    for value in source(ignored=effectful()):
        yield value
''').body[0]
        with self.assertRaisesRegex(TypeError, 'nested yield keyword calls'):
            inline_yield_function_iterators(consumer, {'source': wrapper})


if __name__ == '__main__':
    unittest.main()
