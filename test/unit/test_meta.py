"""
Unit tests for pythoc.meta module.

Tests MetaFragment, GeneratedFunction, MetaArtifact, MetaTemplate,
binding helpers, and quote decorators.
"""

import ast
import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc.meta.fragment import MetaFragment, FragmentKind
from pythoc.meta.generated import GeneratedFunction, MetaArtifact, func, artifact
from pythoc.meta.template import (
    MetaTemplate,
    quote_expr, quote_stmt, quote_stmts, quote_func,
    ref, ident, const, type_expr, splice_stmts,
    _coerce_to_ast, _ParamSubstituter,
)


# ============================================================================
# MetaFragment tests
# ============================================================================

class TestMetaFragment(unittest.TestCase):

    def test_expr_kind(self):
        node = ast.Constant(value=42)
        frag = MetaFragment(kind=FragmentKind.EXPR, node=node)
        self.assertEqual(frag.kind, FragmentKind.EXPR)
        self.assertIs(frag.as_expr, node)

    def test_expr_wrong_accessor(self):
        node = ast.Constant(value=42)
        frag = MetaFragment(kind=FragmentKind.EXPR, node=node)
        with self.assertRaises(TypeError):
            _ = frag.as_stmt

    def test_stmt_kind(self):
        node = ast.Pass()
        frag = MetaFragment(kind=FragmentKind.STMT, node=node)
        self.assertEqual(frag.kind, FragmentKind.STMT)
        self.assertIs(frag.as_stmt, node)

    def test_stmts_kind(self):
        nodes = [ast.Pass(), ast.Pass()]
        frag = MetaFragment(kind=FragmentKind.STMTS, node=nodes)
        self.assertEqual(len(frag.as_stmts), 2)

    def test_func_kind(self):
        func_def = ast.FunctionDef(
            name="foo", args=ast.arguments(
                posonlyargs=[], args=[], vararg=None,
                kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[],
            ),
            body=[ast.Pass()], decorator_list=[], returns=None,
        )
        ast.fix_missing_locations(func_def)
        frag = MetaFragment(kind=FragmentKind.FUNC, node=func_def)
        self.assertEqual(frag.as_func.name, "foo")

    def test_to_ast(self):
        node = ast.Constant(value=42)
        frag = MetaFragment(kind=FragmentKind.EXPR, node=node)
        self.assertIs(frag.to_ast(), node)

    def test_with_name(self):
        func_def = ast.FunctionDef(
            name="foo", args=ast.arguments(
                posonlyargs=[], args=[], vararg=None,
                kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[],
            ),
            body=[ast.Pass()], decorator_list=[], returns=None,
        )
        ast.fix_missing_locations(func_def)
        frag = MetaFragment(kind=FragmentKind.FUNC, node=func_def)
        renamed = frag.with_name("bar")
        self.assertEqual(renamed.as_func.name, "bar")
        # Original unchanged
        self.assertEqual(frag.as_func.name, "foo")

    def test_with_name_wrong_kind(self):
        node = ast.Constant(value=42)
        frag = MetaFragment(kind=FragmentKind.EXPR, node=node)
        with self.assertRaises(TypeError):
            frag.with_name("bar")

    def test_frozen(self):
        node = ast.Constant(value=42)
        frag = MetaFragment(kind=FragmentKind.EXPR, node=node)
        with self.assertRaises(AttributeError):
            frag.kind = FragmentKind.STMT


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
        self.assertEqual(func_def.args.args[0].arg, "a")
        self.assertEqual(func_def.args.args[1].arg, "b")

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
        stmts = [ast.Pass()]
        frag = MetaFragment(kind=FragmentKind.STMTS, node=stmts)
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
        self.assertEqual(art.helpers[0].name, "helper_fn")

    def test_func_builder(self):
        gf = func("test", [("a", int)], int, [ast.Pass()])
        self.assertIsInstance(gf, GeneratedFunction)
        self.assertEqual(gf.name, "test")

    def test_artifact_builder(self):
        primary = func("p", [], None, [ast.Pass()])
        helper = func("h", [], None, [ast.Pass()])
        art = artifact(primary, helpers=[helper])
        self.assertIsInstance(art, MetaArtifact)
        self.assertEqual(len(art.helpers), 1)


# ============================================================================
# Coercion tests
# ============================================================================

class TestCoercion(unittest.TestCase):

    def test_ref_to_ast(self):
        extra = {}
        result = _coerce_to_ast(ref("lhs"), extra)
        self.assertIsInstance(result, ast.Name)
        self.assertEqual(result.id, "lhs")

    def test_const_to_ast(self):
        extra = {}
        result = _coerce_to_ast(const(42), extra)
        self.assertIsInstance(result, ast.Constant)
        self.assertEqual(result.value, 42)

    def test_ident_to_ast(self):
        extra = {}
        result = _coerce_to_ast(ident("var_name"), extra)
        self.assertIsInstance(result, ast.Name)
        self.assertEqual(result.id, "var_name")

    def test_scalar_auto_coerce(self):
        extra = {}
        result = _coerce_to_ast(42, extra)
        self.assertIsInstance(result, ast.Constant)
        self.assertEqual(result.value, 42)

    def test_string_auto_coerce(self):
        extra = {}
        result = _coerce_to_ast("hello", extra)
        self.assertIsInstance(result, ast.Constant)
        self.assertEqual(result.value, "hello")

    def test_none_auto_coerce(self):
        extra = {}
        result = _coerce_to_ast(None, extra)
        self.assertIsInstance(result, ast.Constant)
        self.assertIsNone(result.value)

    def test_ast_node_passthrough(self):
        extra = {}
        node = ast.BinOp(
            left=ast.Name(id='x', ctx=ast.Load()),
            op=ast.Add(),
            right=ast.Constant(value=1),
        )
        result = _coerce_to_ast(node, extra)
        self.assertIs(result, node)

    def test_named_callable_coerce(self):
        extra = {}

        def my_func():
            pass

        result = _coerce_to_ast(my_func, extra)
        self.assertIsInstance(result, ast.Name)
        self.assertEqual(result.id, "my_func")
        self.assertIn("my_func", extra)

    def test_uncoercible_raises(self):
        extra = {}
        with self.assertRaises(TypeError):
            _coerce_to_ast(object(), extra)


# ============================================================================
# ParamSubstituter tests
# ============================================================================

class TestParamSubstituter(unittest.TestCase):

    def test_simple_name_replacement(self):
        # AST for: x + y
        tree = ast.BinOp(
            left=ast.Name(id='x', ctx=ast.Load()),
            op=ast.Add(),
            right=ast.Name(id='y', ctx=ast.Load()),
        )
        ast.fix_missing_locations(tree)

        bindings = {
            'x': ast.Name(id='lhs', ctx=ast.Load()),
            'y': ast.Name(id='rhs', ctx=ast.Load()),
        }
        result = _ParamSubstituter(bindings).visit(tree)
        self.assertEqual(result.left.id, 'lhs')
        self.assertEqual(result.right.id, 'rhs')

    def test_const_replacement(self):
        # AST for: x
        tree = ast.Name(id='x', ctx=ast.Load())
        ast.fix_missing_locations(tree)

        bindings = {'x': ast.Constant(value=42)}
        result = _ParamSubstituter(bindings).visit(tree)
        self.assertIsInstance(result, ast.Constant)
        self.assertEqual(result.value, 42)

    def test_unbound_name_unchanged(self):
        tree = ast.Name(id='z', ctx=ast.Load())
        ast.fix_missing_locations(tree)

        bindings = {'x': ast.Constant(value=1)}
        result = _ParamSubstituter(bindings).visit(tree)
        self.assertIsInstance(result, ast.Name)
        self.assertEqual(result.id, 'z')


# ============================================================================
# Quote decorator tests (design-doc compatible)
# ============================================================================

class TestQuoteDecorators(unittest.TestCase):

    def test_quote_expr_basic(self):
        """Design doc example 15.1: quoted expression with positional args"""
        @quote_expr
        def add_expr(x, y):
            return x + y

        self.assertIsInstance(add_expr, MetaTemplate)
        self.assertEqual(add_expr.kind, FragmentKind.EXPR)
        self.assertEqual(add_expr.param_names, ('x', 'y'))

    def test_quote_expr_instantiate_with_ref(self):
        """Design doc: add_expr(meta.ref("a"), meta.ref("b"))"""
        @quote_expr
        def add_expr(x, y):
            return x + y

        frag = add_expr(ref("a"), ref("b"))
        self.assertIsInstance(frag, MetaFragment)
        self.assertEqual(frag.kind, FragmentKind.EXPR)
        # Should be BinOp(Name("a"), Add, Name("b"))
        expr = frag.as_expr
        self.assertIsInstance(expr, ast.BinOp)
        self.assertIsInstance(expr.left, ast.Name)
        self.assertEqual(expr.left.id, "a")
        self.assertIsInstance(expr.right, ast.Name)
        self.assertEqual(expr.right.id, "b")

    def test_quote_expr_instantiate_with_scalars(self):
        """Plain scalars auto-lower to ast.Constant"""
        @quote_expr
        def literal_expr(x, y):
            return x + y

        frag = literal_expr(10, 20)
        expr = frag.as_expr
        self.assertIsInstance(expr, ast.BinOp)
        self.assertIsInstance(expr.left, ast.Constant)
        self.assertEqual(expr.left.value, 10)
        self.assertIsInstance(expr.right, ast.Constant)
        self.assertEqual(expr.right.value, 20)

    def test_quote_stmts(self):
        @quote_stmts
        def my_template(val):
            x = val

        self.assertIsInstance(my_template, MetaTemplate)
        self.assertEqual(my_template.kind, FragmentKind.STMTS)
        self.assertEqual(my_template.param_names, ('val',))

    def test_quote_func_design_doc(self):
        """Design doc example 15.2: quoted function with type param"""
        @quote_func
        def add_template(ret_ty):
            def generated(x, y):
                tmp = x + y
                return tmp

        self.assertIsInstance(add_template, MetaTemplate)
        self.assertEqual(add_template.kind, FragmentKind.FUNC)
        self.assertEqual(add_template.param_names, ('ret_ty',))

    def test_quote_func_instantiate_and_rename(self):
        """Design doc: add_template(i32).with_name("add_i32")"""
        @quote_func
        def add_template(ret_ty):
            def generated(x, y):
                return x + y

        # Use a scalar as placeholder for ret_ty (auto-coerced)
        frag = add_template(ref("i32"))
        self.assertEqual(frag.kind, FragmentKind.FUNC)
        self.assertEqual(frag.as_func.name, "generated")

        renamed = frag.with_name("add_i32")
        self.assertEqual(renamed.as_func.name, "add_i32")
        # Original unchanged
        self.assertEqual(frag.as_func.name, "generated")

    def test_quote_func_type_in_annotation(self):
        """ret_ty parameter used in annotation position gets substituted"""
        @quote_func
        def template(ret_ty):
            def generated(x) -> ret_ty:
                return x

        frag = template(ref("i32"))
        func_def = frag.as_func
        # The return annotation should now be Name("i32")
        self.assertIsInstance(func_def.returns, ast.Name)
        self.assertEqual(func_def.returns.id, "i32")

    def test_keyword_args(self):
        @quote_expr
        def tmpl(a, b):
            return a + b

        frag = tmpl(b=ref("rhs"), a=ref("lhs"))
        expr = frag.as_expr
        self.assertEqual(expr.left.id, "lhs")
        self.assertEqual(expr.right.id, "rhs")

    def test_mixed_positional_and_keyword(self):
        @quote_expr
        def tmpl(a, b):
            return a + b

        frag = tmpl(ref("lhs"), b=ref("rhs"))
        expr = frag.as_expr
        self.assertEqual(expr.left.id, "lhs")
        self.assertEqual(expr.right.id, "rhs")

    def test_too_many_args_raises(self):
        @quote_expr
        def tmpl(x):
            return x

        with self.assertRaises(TypeError):
            tmpl(ref("a"), ref("b"))

    def test_unknown_kwarg_raises(self):
        @quote_expr
        def tmpl(x):
            return x

        with self.assertRaises(TypeError):
            tmpl(z=ref("a"))

    def test_duplicate_binding_raises(self):
        @quote_expr
        def tmpl(x):
            return x

        with self.assertRaises(TypeError):
            tmpl(ref("a"), x=ref("b"))

    def test_quote_with_decorator_args(self):
        """Quote decorators support optional keyword arguments"""
        @quote_expr(debug_source=False)
        def tmpl(x):
            return x

        self.assertIsInstance(tmpl, MetaTemplate)


if __name__ == '__main__':
    unittest.main()
