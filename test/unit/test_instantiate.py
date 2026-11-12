"""Instantiate capture bindings, type resolution, and iterator expansion."""

import ast
from collections.abc import Mapping
from types import SimpleNamespace
import unittest

from llvmlite import ir

from pythoc import CompileSession, array, f64, i32, struct
from pythoc._inline.closure_capture import snapshot_closure_captures
from pythoc._inline.state_field_rewriter import (
    StateFieldRewriter,
    StateFieldRewritePolicy,
)
from pythoc._inline.yield_state_machine import inline_yield_function_iterators
from pythoc.ast_visitor.functions import FunctionsMixin
from pythoc.context import VariableInfo
from pythoc.meta.instantiate import (
    _build_callee_globals,
    _compile_closure_fn_pipeline,
    _instantiate_const_iterable,
    _instantiate_genexpr,
    _require_yield_type,
    _splice_capture_bindings,
    _typed_zero,
)
from pythoc.type_resolver import TypeResolver
from pythoc.valueref import ValueRef, wrap_python_constant


class TestClosureCapturePlan(unittest.TestCase):
    def test_definition_work_is_bounded_by_referenced_names(self):
        class Symbols(Mapping):
            def __init__(self):
                self.values = {}
                self.work = 0

            def __getitem__(self, name):
                self.work += 1
                return self.values[name]

            def __iter__(self):
                for name in self.values:
                    self.work += 1
                    yield name

            def __len__(self):
                return len(self.values)

        for count in (64, 128, 256):
            with self.subTest(count=count):
                globals_, locals_ = Symbols(), Symbols()
                scope = SimpleNamespace(
                    lookup_variable=locals_.get,
                    get_all_visible=lambda: dict(locals_))
                visitor = SimpleNamespace(
                    ctx=SimpleNamespace(user_globals=globals_), scope_manager=scope)
                retained = 0
                for index in range(count):
                    globals_.values[f'global_{index}'] = object()
                    locals_.values[f'local_{index}'] = object()
                    syntax = ast.parse('def unused():\n    return 0').body[0]
                    wrapper = FunctionsMixin._make_closure_wrapper(visitor, syntax, {})
                    retained += len(wrapper.func_globals) + len(wrapper._capture_references)
                self.assertEqual(retained, 0)
                self.assertLessEqual(globals_.work + locals_.work, 8 * count)

    def test_selected_capture_keeps_the_existing_binding(self):
        value = VariableInfo('captured', value_ref=wrap_python_constant(7))
        unrelated = VariableInfo('unrelated', value_ref=wrap_python_constant(99))
        bindings = {'captured': value, 'unrelated': unrelated}
        visitor = SimpleNamespace(
            ctx=SimpleNamespace(user_globals={'global_value': 3, 'unused': 4}),
            scope_manager=SimpleNamespace(lookup_variable=bindings.get))
        syntax = ast.parse('def work():\n    return captured + global_value').body[0]
        wrapper = FunctionsMixin._make_closure_wrapper(visitor, syntax, {})
        self.assertEqual(wrapper._capture_references, {'captured': value})
        self.assertEqual(wrapper.func_globals, {'global_value': 3})
        value.value_ref = wrap_python_constant(8)
        self.assertEqual(wrapper._capture_references['captured'].value_ref.value, 8)

    def test_splits_compiletime_and_runtime_captures(self):
        references = {
            "scale_alias": VariableInfo('scale', value_ref=wrap_python_constant(3)),
            "offset_alias": VariableInfo('offset', value_ref=ValueRef(
                'value', ir.Constant(ir.IntType(32), 7), i32)),
        }

        plan = snapshot_closure_captures(
            references, {'scale_alias': 'scale', 'offset_alias': 'offset'})

        self.assertEqual(plan.bindings, {"scale_alias": 3})
        self.assertEqual(len(plan.runtime), 1)
        self.assertEqual(plan.runtime[0].name, "offset")
        self.assertIs(plan.runtime[0].pc_type, i32)
        self.assertEqual(plan.runtime[0].value_ref.var_name, 'offset')
        self.assertEqual(plan.runtime[0].value_ref.linear_owner_id,
                         references['offset_alias'].var_id)


class TestCaptureBindings(unittest.TestCase):
    def test_unused_parameter_cannot_shadow_capture_alias(self):
        tree = ast.parse('''
def work(__pc_capture_captured):
    return captured
''')
        namespace = {}
        _splice_capture_bindings(tree.body[0], {'captured': 10}, namespace)
        exec(compile(ast.fix_missing_locations(tree), '<capture>', 'exec'),
             namespace)
        self.assertEqual(namespace['work'](99), 10)

    def test_none_capture_survives_global_resolution(self):
        tree = ast.parse('def work():\n    return captured')
        namespace = {}
        _splice_capture_bindings(tree.body[0], {'captured': None}, namespace)
        namespace = _build_callee_globals(
            namespace, {'captured'}, set(), tree.body[0],
            include_intrinsics=False,
        )
        exec(compile(ast.fix_missing_locations(tree), '<capture>', 'exec'),
             namespace)
        self.assertIsNone(namespace['work']())

    def test_type_capture_preserves_identity_and_avoids_name_collisions(self):
        tree = ast.parse('''
def work():
    __pc_capture_T = 'local'
    return T, __pc_capture_T
''')
        namespace = {'T': str, '__pc_capture_T': 'global'}
        _splice_capture_bindings(tree.body[0], {'T': int}, namespace)
        exec(compile(ast.fix_missing_locations(tree), '<capture>', 'exec'),
             namespace)
        self.assertEqual(namespace['work'](), (int, 'local'))
        self.assertEqual(namespace['__pc_capture_T'], 'global')

    def test_callable_capture_remains_callable(self):
        tree = ast.parse('def work():\n    return callback(41)')
        namespace = {}
        _splice_capture_bindings(
            tree.body[0], {'callback': lambda value: value + 1}, namespace,
        )
        exec(compile(ast.fix_missing_locations(tree), '<capture>', 'exec'),
             namespace)
        self.assertEqual(namespace['work'](), 42)


class TestStateFieldRewriter(unittest.TestCase):
    def test_capture_alias_maps_to_the_public_state_field(self):
        fn = ast.parse('''
def work():
    _capture_offset += 1
    return _capture_offset
''').body[0]
        rewriter = StateFieldRewriter(StateFieldRewritePolicy(
            fields={'_capture_offset': 'offset'}, protect_names={'s'}))
        body = [rewriter.visit(stmt) for stmt in fn.body]
        self.assertEqual(body[0].target.attr, 'offset')
        self.assertEqual(body[1].value.attr, 'offset')
        self.assertEqual(body[1].value.value.id, 's')

    def test_preserves_nested_function_and_rewrites_state_capture(self):
        fn = ast.parse(
            """
def outer(x: i32) -> i32:
    local: i32 = 3
    def inner(y: i32) -> i32:
        return y + local + offset
    return inner(x)
"""
        ).body[0]

        rewriter = StateFieldRewriter(StateFieldRewritePolicy(
            fields={"offset": "offset"},
            protect_names={"s"},
            preserve_nested_functions=True,
        ))
        rewritten = [rewriter.visit(stmt) for stmt in fn.body]
        src = "\n".join(ast.unparse(stmt) for stmt in rewritten)

        self.assertIn("def inner", src)
        self.assertIn("s.offset", src)
        self.assertIn("local", src)


class TestClosureGeneratorBoundary(unittest.TestCase):
    def test_closure_with_yield_is_rejected(self):
        fn = ast.parse(
            """
def bad() -> i32:
    yield 1
"""
        ).body[0]

        with self.assertRaises((TypeError, SystemExit)):
            _compile_closure_fn_pipeline(
                fn,
                source_object_id=1,
                func_name_hint="bad",
                callee_globals={"i32": i32},
            )


class TestInstantiateTypes(unittest.TestCase):
    def test_missing_yield_annotation_is_an_error(self):
        syntax = ast.parse('def source():\n    yield 1.25').body[0]
        with self.assertRaisesRegex(TypeError, 'declared yield type'):
            _require_yield_type(syntax, TypeResolver())

    def test_declared_yield_types_keep_their_identity(self):
        resolver = TypeResolver(user_globals={'Alias': f64, 'i32': i32})
        for annotation, expected in (('Alias', f64), ('i32', i32)):
            syntax = ast.parse(
                f'def source() -> {annotation}:\n    yield 1').body[0]
            self.assertIs(_require_yield_type(syntax, resolver), expected)

    def test_generator_names_calls_and_arithmetic_do_not_guess(self):
        events = []

        def value():
            events.append(1)
            return 1.25

        for expression in ('VALUE', 'value()', 'VALUE + 0.5', 'f64(index)'):
            with self.subTest(expression=expression), CompileSession():
                source = SimpleNamespace(_pc_generator_expr_info={
                    'kind': 'generator_expression',
                    'ast': ast.parse(
                        f'({expression} for index in range(2))', mode='eval').body,
                    'globals': {'VALUE': 1.25, 'value': value, 'f64': f64},
                })
                with self.assertRaisesRegex(TypeError, 'declared yield type'):
                    _instantiate_genexpr(source)
        self.assertEqual(events, [])


    def test_empty_iterable_has_no_implicit_element_type(self):
        with self.assertRaisesRegex(TypeError, 'element type of an empty iterable'):
            _instantiate_const_iterable([])

    def test_type_expression_preserves_the_actual_object(self):
        with CompileSession():
            for hint in (array[i32, 2], struct['left': i32, 'right': f64]):
                self.assertIs(TypeResolver().parse_annotation(
                    _typed_zero(hint).func), hint)


class TestYieldIteratorExpansion(unittest.TestCase):
    def test_offline_yield_resolution_uses_normal_callees_and_definition_bindings(self):
        source = ast.parse('def source() -> i32:\n    yield OFFSET').body[0]
        wrapper = SimpleNamespace(_original_ast=source,
                                  __globals__={'i32': i32, 'OFFSET': 7})
        namespace = SimpleNamespace(inner=SimpleNamespace(source=wrapper))
        globals_ = {'source': wrapper, 'namespace': namespace,
                    'choose': lambda: wrapper, 'OFFSET': 100}
        for callee in ('source', 'namespace.inner.source', 'choose()'):
            with self.subTest(callee=callee), CompileSession():
                syntax = ast.parse(
                    f'def consumer():\n    for value in {callee}():\n'
                    '        yield value').body[0]
                result = inline_yield_function_iterators(syntax, globals_)
                self.assertFalse(any(isinstance(node, ast.For) for node in ast.walk(result)))
                bindings = [getattr(node, '_definition_symbol', None)
                            for node in ast.walk(result)]
                self.assertTrue(any(symbol is not None and symbol.value == 7
                                    for symbol in bindings))


if __name__ == "__main__":
    unittest.main()
