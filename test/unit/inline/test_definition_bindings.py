"""Definition references retain binding identity without declaring new owners."""

import ast
import unittest
from types import SimpleNamespace

from pythoc import i32
from pythoc._inline.closure_capture import build_closure_capture_plan
from pythoc._inline.definition_bindings import bind_definition
from pythoc._inline.exit_rules import ReturnExitRule
from pythoc._inline.kernel import MetaInlineRequest, expand_inline
from pythoc._inline.scope_analyzer import ScopeContext
from pythoc.ast_visitor.base import LLVMIRVisitor
from pythoc.context import VariableInfo
from pythoc.scope_manager import ScopeManager, ScopeType
from pythoc.valueref import wrap_python_constant


class DefinitionBindingsTest(unittest.TestCase):
    def test_capture_plan_keeps_the_actual_variable_binding(self):
        original = VariableInfo('value', value_ref=wrap_python_constant(3))
        plan = build_closure_capture_plan({'value'}, {'value': original})
        original.value_ref = wrap_python_constant(7)
        self.assertIs(plan.references['value'], original)
        self.assertEqual(plan.references['value'].value_ref.get_python_value(), 7)
        self.assertEqual(plan.bindings['value'], 3)

    def test_reference_alias_has_no_new_owner_or_variable_lifetime(self):
        manager = ScopeManager()
        original = VariableInfo('value', value_ref=wrap_python_constant(3))
        manager.declare_variable(original)
        owner = original.var_id
        with manager.binding_aliases({'capture': original}):
            self.assertIs(manager.lookup_variable('capture'), original)
            self.assertEqual(manager.lookup_variable('capture').var_id, owner)
            self.assertIs(manager.get_all_visible()['capture'], original)
            self.assertEqual(manager.get_function_variables(), [original])
            self.assertEqual(manager.get_all_in_current_scope(), [original])
        self.assertIsNone(manager.lookup_variable('capture'))
        self.assertEqual(original.name, 'value')

    def test_nested_alias_scopes_restore_after_failure(self):
        manager = ScopeManager()
        first = VariableInfo('first', value_ref=wrap_python_constant(3))
        second = VariableInfo('second', value_ref=wrap_python_constant(7))
        with manager.binding_aliases({'capture': first}):
            with self.assertRaisesRegex(ValueError, 'test failure'):
                with manager.binding_aliases({'capture': second}):
                    self.assertIs(manager.lookup_variable('capture'), second)
                    raise ValueError('test failure')
            self.assertIs(manager.lookup_variable('capture'), first)
        self.assertIsNone(manager.lookup_variable('capture'))

    def test_lexical_declaration_shadows_an_alias_until_its_scope_exits(self):
        manager = ScopeManager()
        original = VariableInfo('value', value_ref=wrap_python_constant(3))
        shadow = VariableInfo('capture', value_ref=wrap_python_constant(7))
        control_flow = SimpleNamespace(is_terminated=lambda: True)
        with manager.binding_aliases({'capture': original}):
            with manager.scope(ScopeType.MATCH, control_flow):
                manager.declare_variable(shadow)
                self.assertIs(manager.lookup_variable('capture'), shadow)
                self.assertIs(manager.get_all_visible()['capture'], shadow)
            self.assertIs(manager.lookup_variable('capture'), original)
            self.assertIs(manager.get_all_visible()['capture'], original)

    def test_preparation_uses_definition_binding_through_a_caller_shadow(self):
        visitor = LLVMIRVisitor()
        original = VariableInfo('value', value_ref=wrap_python_constant(3))
        shadow = VariableInfo('value', value_ref=wrap_python_constant(7))
        visitor.scope_manager.declare_variable(shadow)
        source = ast.parse('def read() -> i32:\n    return value').body[0]
        before = ast.dump(source)
        bound = bind_definition(visitor, source, {'i32': i32}, {'value': original})
        name = bound.syntax.body[0].value.id
        self.assertNotEqual(name, 'value')
        self.assertIs(bound.references[name], original)
        self.assertEqual(ast.dump(source), before)

    def test_nested_local_masks_a_definition_global(self):
        visitor = LLVMIRVisitor()
        source = ast.parse('''
def read():
    def inner():
        value = 9
        return value
    return inner() + value
''').body[0]
        bound = bind_definition(visitor, source, {'value': 7})
        inner = bound.syntax.body[0]
        self.assertEqual(inner.body[0].targets[0].id, 'value')
        self.assertEqual(inner.body[1].value.id, 'value')
        name = bound.syntax.body[1].value.right.id
        self.assertNotEqual(name, 'value')
        self.assertEqual(bound.globals[name], 7)

    def test_unresolved_definition_name_cannot_reuse_a_consumer_binding(self):
        visitor = LLVMIRVisitor()
        source = ast.parse('def read():\n    return missing').body[0]
        bound = bind_definition(visitor, source, {})
        name = bound.syntax.body[0].value.id
        self.assertNotEqual(name, 'missing')
        self.assertNotIn(name, bound.globals)

    def test_comprehension_targets_mask_globals_after_their_iterable(self):
        visitor = LLVMIRVisitor()
        source = ast.parse('''
def read():
    return (item + other for item in item for other in (item,))
''').body[0]
        bound = bind_definition(visitor, source, {'item': (1, 2), 'other': 7})
        expression = bound.syntax.body[0].value
        iterable = expression.generators[0].iter.id
        self.assertNotEqual(iterable, 'item')
        self.assertEqual(bound.globals[iterable], (1, 2))
        self.assertEqual(expression.generators[0].target.id, 'item')
        self.assertEqual(expression.generators[1].iter.elts[0].id, 'item')
        self.assertEqual(expression.elt.left.id, 'item')
        self.assertEqual(expression.elt.right.id, 'other')

    def test_kernel_never_infers_captures_from_consumer_names(self):
        source = ast.parse('def read():\n    value = 7\n    return value').body[0]
        request = MetaInlineRequest(
            callee_ast=source, callee_globals={}, call_args=[],
            call_site=ast.parse('read()', mode='eval').body,
            caller_context=ScopeContext({'value'}),
            exit_rule=ReturnExitRule(result_var='result'))
        result = expand_inline(request)
        assigned = {node.id for statement in result.stmts for node in ast.walk(statement)
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
        self.assertNotIn('value', assigned)


if __name__ == '__main__':
    unittest.main()
