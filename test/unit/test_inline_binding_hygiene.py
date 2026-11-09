"""Scope analysis and renaming agree about bindings in ordinary source ASTs."""

import ast
import inspect
import textwrap
import unittest

from pythoc._inline.exit_rules import VariableRenamer
from pythoc._inline.names import ExpansionNames
from pythoc._inline.scope_analyzer import ScopeAnalyzer, ScopeContext


def pattern_forms(subject):
    match subject:
        case [first, *rest] as whole if first:
            return first, rest, whole
        case {'key': value, **extra}:
            return value, extra


def nested_parameter(seed):
    def inner(name=seed):
        return name + 1
    return inner(name=3)


def nested_annotated():
    def inner():
        name: int = 3
        return name
    return inner()


def nested_capture():
    def inner():
        name += 2
        return name
    return inner()


def nested_layers():
    def inner(name):
        def deepest():
            return name
        return deepest()
    return inner(3)


def lambda_shadow():
    return (lambda name=name: name + saved)(3)


def source_ast(function):
    source = textwrap.dedent(inspect.getsource(function))
    return ast.parse(source).body[0]


class InlineBindingHygieneTest(unittest.TestCase):
    def test_unused_mapping_rest_binding_is_reserved_for_generated_names(self):
        names = ExpansionNames()
        names.reserve_ast(ast.parse('''
match subject:
    case {'key': value, **_arg_inline}:
        pass
'''))
        self.assertNotEqual(names.fresh('_arg_inline'), '_arg_inline')

    def test_pattern_assignments_are_local_without_a_capture_context(self):
        function = source_ast(pattern_forms)
        captured, local, parameters = ScopeAnalyzer(ScopeContext.empty()).analyze(
            function.body, function.args.args)
        self.assertEqual(captured, set())
        self.assertEqual(local, {'first', 'rest', 'whole', 'value', 'extra'})
        self.assertEqual(parameters, {'subject'})

    def test_pattern_assignments_retain_explicit_closure_capture_policy(self):
        function = source_ast(pattern_forms)
        captured, local, _ = ScopeAnalyzer(ScopeContext({'first'})).analyze(
            function.body, function.args.args)
        self.assertEqual(captured, {'first'})
        self.assertEqual(local, {'rest', 'whole', 'value', 'extra'})

    def test_pattern_binders_guards_and_references_use_one_rename_map(self):
        function = source_ast(pattern_forms)
        original = {'first', 'rest', 'whole', 'value', 'extra'}
        rename = {name: name + '_scope' for name in original}
        transformed = VariableRenamer(rename).visit(function.body[0])
        bound = {node.name for node in ast.walk(transformed)
                 if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name}
        bound.update(node.rest for node in ast.walk(transformed)
                     if isinstance(node, ast.MatchMapping) and node.rest)
        self.assertEqual(bound, set(rename.values()))
        names = {node.id for node in ast.walk(transformed) if isinstance(node, ast.Name)}
        self.assertFalse(names & original)
        self.assertTrue(set(rename.values()) <= names)

    def test_nested_parameters_preserve_keyword_names_and_outer_defaults(self):
        function = source_ast(nested_parameter)
        renamer = VariableRenamer({'name': 'name_scope', 'seed': 'seed_scope',
                                   'inner': 'inner_scope'})
        inner = renamer.visit(function.body[0])
        call = renamer.visit(function.body[1]).value
        self.assertEqual(inner.name, 'inner_scope')
        self.assertEqual(inner.args.args[0].arg, 'name')
        self.assertEqual(inner.args.defaults[0].id, 'seed_scope')
        self.assertIsInstance(inner.body[0], ast.Return)
        self.assertEqual(inner.body[0].value.left.id, 'name')
        self.assertEqual(call.func.id, 'inner_scope')
        self.assertEqual(call.keywords[0].arg, 'name')

    def test_annotated_inner_local_shadows_outer_mapping(self):
        inner = source_ast(nested_annotated).body[0]
        transformed = VariableRenamer({'name': 'name_scope'}).visit(inner)
        self.assertEqual(transformed.body[0].target.id, 'name')
        self.assertEqual(transformed.body[1].value.id, 'name')

    def test_unannotated_inner_assignment_retains_by_reference_capture(self):
        inner = source_ast(nested_capture).body[0]
        transformed = VariableRenamer({'name': 'name_scope'}).visit(inner)
        self.assertEqual(transformed.body[0].target.id, 'name_scope')
        self.assertEqual(transformed.body[1].value.id, 'name_scope')

    def test_parameter_shadow_survives_two_nested_function_levels(self):
        inner = source_ast(nested_layers).body[0]
        transformed = VariableRenamer({'name': 'name_scope'}).visit(inner)
        self.assertEqual(transformed.args.args[0].arg, 'name')
        self.assertEqual(transformed.body[0].body[0].value.id, 'name')

    def test_lambda_parameter_shadows_outer_mapping_but_default_does_not(self):
        expression = source_ast(lambda_shadow).body[0].value
        transformed = VariableRenamer({'name': 'name_scope',
                                        'saved': 'saved_scope'}).visit(expression)
        function = transformed.func
        self.assertEqual(function.args.args[0].arg, 'name')
        self.assertEqual(function.args.defaults[0].id, 'name_scope')
        self.assertEqual(function.body.left.id, 'name')
        self.assertEqual(function.body.right.id, 'saved_scope')


if __name__ == '__main__':
    unittest.main()
