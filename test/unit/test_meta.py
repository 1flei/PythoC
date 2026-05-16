"""
Unit tests for pythoc.meta module (post fragment-as-currency redesign).

Tests Fragment, GeneratedFunction, MetaArtifact, MetaTemplate, the
single ``const`` binding helper, and the unified ``@quote`` decorator.
"""

import ast
import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc.meta.fragment import Fragment
from pythoc.meta.generated import GeneratedFunction, MetaArtifact, func, artifact
from pythoc.meta.template import (
    MetaTemplate,
    quote,
    const,
    _coerce_to_ast,
    _ParamSubstituter,
    _SpliceMarker,
)


# ============================================================================
# Fragment tests
# ============================================================================

class TestFragment(unittest.TestCase):

    def test_stmts_always_succeeds(self):
        body = [ast.Pass()]
        frag = Fragment(body=body)
        self.assertIs(frag.stmts, body)

    def test_stmt_single(self):
        node = ast.Pass()
        frag = Fragment(body=[node])
        self.assertIs(frag.stmt, node)

    def test_stmt_multi_raises(self):
        frag = Fragment(body=[ast.Pass(), ast.Pass()])
        with self.assertRaises(TypeError):
            _ = frag.stmt

    def test_expr_from_return(self):
        inner = ast.Constant(value=42)
        frag = Fragment(body=[ast.Return(value=inner)])
        self.assertIs(frag.expr, inner)

    def test_expr_from_expr_stmt(self):
        inner = ast.Name(id='x', ctx=ast.Load())
        frag = Fragment(body=[ast.Expr(value=inner)])
        self.assertIs(frag.expr, inner)

    def test_expr_from_non_expr_stmt_raises(self):
        frag = Fragment(body=[ast.Pass()])
        with self.assertRaises(TypeError):
            _ = frag.expr

    def test_expr_from_multi_stmt_raises(self):
        frag = Fragment(body=[
            ast.Return(value=ast.Constant(value=1)),
            ast.Return(value=ast.Constant(value=2)),
        ])
        with self.assertRaises(TypeError):
            _ = frag.expr

    def test_with_func_name(self):
        func_def = ast.FunctionDef(
            name="foo", args=ast.arguments(
                posonlyargs=[], args=[], vararg=None,
                kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[],
            ),
            body=[ast.Pass()], decorator_list=[], returns=None,
        )
        ast.fix_missing_locations(func_def)
        frag = Fragment(body=[func_def])
        renamed = frag.with_func_name("bar")
        self.assertEqual(renamed.body[0].name, "bar")
        # Original unchanged
        self.assertEqual(frag.body[0].name, "foo")

    def test_with_func_name_wrong_kind(self):
        frag = Fragment(body=[ast.Pass()])
        with self.assertRaises(TypeError):
            frag.with_func_name("bar")


# ============================================================================
# GeneratedFunction tests
# ============================================================================

class TestGeneratedFunction(unittest.TestCase):

    def test_to_func_def(self):
        body = [ast.Return(value=ast.BinOp(
            left=ast.Name(id='a', ctx=ast.Load()),
            op=ast.Add(),
            right=ast.Name(id='b', ctx=ast.Load()),
        ))]
        gf = GeneratedFunction(
            name="add",
            params=[("a", None), ("b", None)],
            return_type=None,
            body=body,
        )
        func_def = gf.to_func_def()
        self.assertIsInstance(func_def, ast.FunctionDef)
        self.assertEqual(func_def.name, "add")
        self.assertEqual(len(func_def.args.args), 2)

    def test_param_type_hints(self):
        gf = GeneratedFunction(
            name="f",
            params=[("x", "i32_type"), ("y", "f64_type")],
            return_type="i32_type",
            body=[ast.Pass()],
        )
        hints = gf.get_param_type_hints()
        self.assertEqual(hints, {"x": "i32_type", "y": "f64_type"})
        self.assertEqual(gf.get_return_type_hint(), "i32_type")

    def test_empty_body_gets_pass(self):
        gf = GeneratedFunction(name="f", params=[], return_type=None, body=[])
        func_def = gf.to_func_def()
        self.assertEqual(len(func_def.body), 1)
        self.assertIsInstance(func_def.body[0], ast.Pass)

    def test_body_from_fragment(self):
        frag = Fragment(body=[ast.Pass()])
        gf = GeneratedFunction(name="f", params=[], return_type=None, body=frag)
        func_def = gf.to_func_def()
        self.assertEqual(len(func_def.body), 1)


# ============================================================================
# MetaArtifact tests
# ============================================================================

class TestMetaArtifact(unittest.TestCase):

    def test_basic_artifact(self):
        primary = GeneratedFunction(
            name="main_fn", params=[], return_type=None, body=[ast.Pass()]
        )
        helper = GeneratedFunction(
            name="helper_fn", params=[], return_type=None, body=[ast.Pass()]
        )
        art = MetaArtifact(primary=primary, helpers=(helper,))
        self.assertEqual(art.primary.name, "main_fn")
        self.assertEqual(len(art.helpers), 1)

    def test_func_builder(self):
        gf = func("test", [("a", int)], int, [ast.Pass()])
        self.assertIsInstance(gf, GeneratedFunction)

    def test_artifact_builder(self):
        primary = func("p", [], None, [ast.Pass()])
        helper = func("h", [], None, [ast.Pass()])
        art = artifact(primary, helpers=[helper])
        self.assertIsInstance(art, MetaArtifact)


# ============================================================================
# Coercion tests (position-aware)
# ============================================================================

class TestCoercionExprPosition(unittest.TestCase):

    def test_str_to_name(self):
        result = _coerce_to_ast("x", {}, 'expr')
        self.assertIsInstance(result, ast.Name)
        self.assertEqual(result.id, "x")
        self.assertIsInstance(result.ctx, ast.Load)

    def test_const_helper_to_constant(self):
        result = _coerce_to_ast(const("hello"), {}, 'expr')
        self.assertIsInstance(result, ast.Constant)
        self.assertEqual(result.value, "hello")

    def test_int_to_constant(self):
        result = _coerce_to_ast(42, {}, 'expr')
        self.assertIsInstance(result, ast.Constant)
        self.assertEqual(result.value, 42)

    def test_float_to_constant(self):
        result = _coerce_to_ast(3.14, {}, 'expr')
        self.assertEqual(result.value, 3.14)

    def test_bool_to_constant(self):
        result = _coerce_to_ast(True, {}, 'expr')
        self.assertIs(result.value, True)

    def test_none_to_constant(self):
        result = _coerce_to_ast(None, {}, 'expr')
        self.assertIsNone(result.value)

    def test_ast_expr_passthrough(self):
        node = ast.BinOp(
            left=ast.Name(id='x', ctx=ast.Load()),
            op=ast.Add(),
            right=ast.Constant(value=1),
        )
        self.assertIs(_coerce_to_ast(node, {}, 'expr'), node)

    def test_fragment_in_expr_unwraps_return(self):
        inner = ast.Constant(value=7)
        frag = Fragment(body=[ast.Return(value=inner)])
        self.assertIs(_coerce_to_ast(frag, {}, 'expr'), inner)

    def test_fragment_multi_stmt_in_expr_raises(self):
        frag = Fragment(body=[ast.Pass(), ast.Pass()])
        with self.assertRaises(TypeError):
            _coerce_to_ast(frag, {}, 'expr')

    def test_named_callable_registered(self):
        extra = {}

        def my_func():
            pass

        result = _coerce_to_ast(my_func, extra, 'expr')
        self.assertIsInstance(result, ast.Name)
        self.assertEqual(result.id, "my_func")
        self.assertIn("my_func", extra)


class TestCoercionStorePosition(unittest.TestCase):

    def test_str_to_store_name(self):
        result = _coerce_to_ast("x", {}, 'store')
        self.assertIsInstance(result, ast.Name)
        self.assertEqual(result.id, "x")
        self.assertIsInstance(result.ctx, ast.Store)

    def test_ast_name_passthrough(self):
        node = ast.Name(id='x', ctx=ast.Store())
        self.assertIs(_coerce_to_ast(node, {}, 'store'), node)

    def test_tuple_passthrough(self):
        node = ast.Tuple(elts=[ast.Name(id='a', ctx=ast.Store())], ctx=ast.Store())
        self.assertIs(_coerce_to_ast(node, {}, 'store'), node)

    def test_int_in_store_raises(self):
        with self.assertRaises(TypeError):
            _coerce_to_ast(42, {}, 'store')


class TestCoercionSplicePosition(unittest.TestCase):

    def test_list_of_stmts_to_splice_marker(self):
        stmts = [ast.Pass(), ast.Pass()]
        result = _coerce_to_ast(stmts, {}, 'splice')
        self.assertIsInstance(result, _SpliceMarker)
        self.assertEqual(result.stmts, stmts)

    def test_fragment_to_splice_marker(self):
        body = [ast.Pass()]
        frag = Fragment(body=body)
        result = _coerce_to_ast(frag, {}, 'splice')
        self.assertIsInstance(result, _SpliceMarker)
        self.assertEqual(result.stmts, body)

    def test_single_stmt_to_splice_marker(self):
        s = ast.Pass()
        result = _coerce_to_ast(s, {}, 'splice')
        self.assertIsInstance(result, _SpliceMarker)
        self.assertEqual(result.stmts, [s])

    def test_str_in_splice_wraps_in_expr(self):
        result = _coerce_to_ast("x", {}, 'splice')
        self.assertIsInstance(result, _SpliceMarker)
        self.assertEqual(len(result.stmts), 1)
        self.assertIsInstance(result.stmts[0], ast.Expr)
        self.assertIsInstance(result.stmts[0].value, ast.Name)


# ============================================================================
# ParamSubstituter tests
# ============================================================================

class TestParamSubstituter(unittest.TestCase):

    def test_simple_name_replacement(self):
        # Manually tag positions (production code does this in _PositionAnalyzer)
        x_node = ast.Name(id='x', ctx=ast.Load())
        x_node._meta_position = 'expr'
        y_node = ast.Name(id='y', ctx=ast.Load())
        y_node._meta_position = 'expr'
        tree = ast.BinOp(left=x_node, op=ast.Add(), right=y_node)
        ast.fix_missing_locations(tree)

        substituter = _ParamSubstituter({'x': 'lhs', 'y': 'rhs'}, {})
        result = substituter.visit(tree)
        self.assertEqual(result.left.id, 'lhs')
        self.assertEqual(result.right.id, 'rhs')

    def test_unbound_name_unchanged(self):
        z_node = ast.Name(id='z', ctx=ast.Load())
        ast.fix_missing_locations(z_node)
        substituter = _ParamSubstituter({'x': 1}, {})
        result = substituter.visit(z_node)
        self.assertIsInstance(result, ast.Name)
        self.assertEqual(result.id, 'z')


# ============================================================================
# Quote decorator tests
# ============================================================================

class TestQuote(unittest.TestCase):

    def test_basic_expr(self):
        @quote
        def add_expr(x, y):
            return x + y

        self.assertIsInstance(add_expr, MetaTemplate)
        self.assertEqual(add_expr.param_names, ('x', 'y'))

    def test_instantiate_returns_fragment(self):
        @quote
        def add_expr(x, y):
            return x + y

        frag = add_expr("a", "b")
        self.assertIsInstance(frag, Fragment)
        # .expr extracts the BinOp from inside the Return
        binop = frag.expr
        self.assertIsInstance(binop, ast.BinOp)
        self.assertEqual(binop.left.id, "a")
        self.assertEqual(binop.right.id, "b")

    def test_int_args_become_constants(self):
        @quote
        def literal(x, y):
            return x + y

        binop = literal(10, 20).expr
        self.assertEqual(binop.left.value, 10)
        self.assertEqual(binop.right.value, 20)

    def test_string_literal_via_const(self):
        @quote
        def echo(x):
            return x

        # bare str -> Name
        self.assertIsInstance(echo("varname").expr, ast.Name)
        # const("...") -> Constant
        self.assertIsInstance(echo(const("varname")).expr, ast.Constant)

    def test_stmts_form(self):
        @quote
        def assign_two(a, b):
            x: i32 = a
            y: i32 = b

        stmts = assign_two("p", "q").stmts
        self.assertEqual(len(stmts), 2)
        self.assertIsInstance(stmts[0], ast.AnnAssign)

    def test_splice_position_with_list(self):
        @quote
        def if_template(cond, body):
            if cond:
                body

        body_stmts = [ast.Pass(), ast.Pass()]
        frag = if_template("c", body_stmts)
        if_node = frag.stmt
        self.assertIsInstance(if_node, ast.If)
        # Body was spliced
        self.assertEqual(len(if_node.body), 2)

    def test_store_position(self):
        @quote
        def assign_one(target, value):
            target: i32 = value

        frag = assign_one("x", 99)
        ann = frag.stmt
        self.assertIsInstance(ann, ast.AnnAssign)
        self.assertIsInstance(ann.target, ast.Name)
        self.assertEqual(ann.target.id, "x")
        self.assertIsInstance(ann.target.ctx, ast.Store)
        self.assertEqual(ann.value.value, 99)

    def test_keyword_args(self):
        @quote
        def tmpl(a, b):
            return a + b

        binop = tmpl(b="rhs", a="lhs").expr
        self.assertEqual(binop.left.id, "lhs")
        self.assertEqual(binop.right.id, "rhs")

    def test_too_many_args_raises(self):
        @quote
        def tmpl(x):
            return x

        with self.assertRaises(TypeError):
            tmpl("a", "b")

    def test_unknown_kwarg_raises(self):
        @quote
        def tmpl(x):
            return x

        with self.assertRaises(TypeError):
            tmpl(z="a")

    def test_decorator_with_kwargs(self):
        @quote(debug_source=False)
        def tmpl(x):
            return x

        self.assertIsInstance(tmpl, MetaTemplate)

    def test_func_template_with_func_name(self):
        @quote
        def add_template(ret_ty):
            def generated(x, y) -> ret_ty:
                tmp = x + y
                return tmp

        frag = add_template("i32")
        # body[0] is the inner FunctionDef
        renamed = frag.with_func_name("add_i32")
        self.assertEqual(renamed.body[0].name, "add_i32")
        # Inner FunctionDef has return annotation substituted
        self.assertEqual(renamed.body[0].returns.id, "i32")


# ============================================================================
# MetaInlineRequest tests (unchanged)
# ============================================================================

class TestMetaInlineRequest(unittest.TestCase):

    def test_construction(self):
        from pythoc.inline.kernel import MetaInlineRequest
        from pythoc.inline.exit_rules import ReturnExitRule
        from pythoc.inline.scope_analyzer import ScopeContext

        callee_ast = ast.FunctionDef(
            name="foo",
            args=ast.arguments(
                posonlyargs=[], args=[ast.arg(arg='x')],
                vararg=None, kwonlyargs=[], kw_defaults=[],
                kwarg=None, defaults=[],
            ),
            body=[ast.Return(value=ast.Name(id='x', ctx=ast.Load()))],
            decorator_list=[], returns=None,
        )
        ast.fix_missing_locations(callee_ast)

        call_site = ast.Call(
            func=ast.Name(id='foo', ctx=ast.Load()),
            args=[ast.Constant(value=42)],
            keywords=[],
        )

        request = MetaInlineRequest(
            callee_ast=callee_ast,
            callee_globals={},
            call_args=[ast.Constant(value=42)],
            call_site=call_site,
            caller_context=ScopeContext(available_vars=set()),
            exit_rule=ReturnExitRule(result_var="_result"),
        )

        self.assertIsNotNone(request)
        self.assertEqual(request.callee_ast.name, "foo")
        self.assertEqual(len(request.call_args), 1)
        self.assertEqual(request.exit_rule.result_var, "_result")


# ============================================================================
# Normalize factory key tests (unchanged)
# ============================================================================

class TestNormalizeFactoryKey(unittest.TestCase):

    def setUp(self):
        from pythoc.meta.normalize import normalize_factory_key
        self.normalize = normalize_factory_key

    def test_empty(self):
        self.assertEqual(self.normalize(), "empty")

    def test_int(self):
        self.assertIn("42", self.normalize(42))

    def test_str(self):
        self.assertIn("hello", self.normalize("hello"))

    def test_tuple(self):
        self.assertIn("T(", self.normalize((1, 2, 3)))

    def test_list(self):
        self.assertIn("L(", self.normalize([1, 2]))

    def test_dict(self):
        self.assertIn("D(", self.normalize({"a": 1, "b": 2}))

    def test_determinism(self):
        a = self.normalize(1, "x", (2, 3))
        b = self.normalize(1, "x", (2, 3))
        self.assertEqual(a, b)

    def test_dict_key_order_independence(self):
        d1 = {"b": 2, "a": 1}
        d2 = {"a": 1, "b": 2}
        self.assertEqual(self.normalize(d1), self.normalize(d2))


# ============================================================================
# Factory decorator tests (unchanged)
# ============================================================================

class TestFactory(unittest.TestCase):

    def test_factory_returns_callable(self):
        from pythoc.meta.factory import factory

        @factory
        def make_fn(n):
            return GeneratedFunction(
                name="fn_{}".format(n),
                params=[],
                return_type=None,
                body=[ast.Pass()],
            )

        self.assertTrue(callable(make_fn))
        self.assertTrue(hasattr(make_fn, '_is_meta_factory'))

    def test_factory_caching(self):
        from pythoc.meta.factory import factory

        call_count = [0]

        @factory
        def make_fn(n):
            call_count[0] += 1
            return GeneratedFunction(
                name="fn_{}".format(n),
                params=[],
                return_type=None,
                body=[ast.Pass()],
            )

        r1 = make_fn(1)
        r2 = make_fn(1)
        self.assertIs(r1, r2)
        self.assertEqual(call_count[0], 1)

    def test_factory_wrong_return_type_raises(self):
        from pythoc.meta.factory import factory

        @factory
        def make_fn():
            return "not a GeneratedFunction"

        with self.assertRaises(TypeError):
            make_fn()


if __name__ == '__main__':
    unittest.main()
