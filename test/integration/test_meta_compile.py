#!/usr/bin/env python3
"""
Integration tests for pythoc.meta module.

Tests compile_ast, compile_generated, compile_artifact, and
template-based code generation with actual compilation and execution.

IMPORTANT: All @compile and meta.compile_* calls must happen at module
level before any test function is invoked, to avoid the "Cannot define
new compiled function after native execution has started" error.
"""

import sys
import os
import ast
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc import compile, i32, i64, meta


# ============================================================================
# Setup: All compilation at module level
# ============================================================================

# --- Test 1: compile_ast basic add ---

def _make_add_ast():
    """Build AST for: def add(a, b): return a + b"""
    return ast.FunctionDef(
        name="meta_add",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg='a'), ast.arg(arg='b')],
            vararg=None, kwonlyargs=[], kw_defaults=[],
            kwarg=None, defaults=[],
        ),
        body=[
            ast.Return(value=ast.BinOp(
                left=ast.Name(id='a', ctx=ast.Load()),
                op=ast.Add(),
                right=ast.Name(id='b', ctx=ast.Load()),
            ))
        ],
        decorator_list=[],
        returns=None,
        lineno=1, col_offset=0,
    )


add_ast = _make_add_ast()
ast.fix_missing_locations(add_ast)

meta_add = meta.compile_ast(
    add_ast,
    param_types={'a': i32, 'b': i32},
    return_type=i32,
    user_globals=globals(),
)

# --- Test 2: compile_generated basic ---

gen_mul = meta.func(
    name="meta_mul",
    params=[("x", i32), ("y", i32)],
    return_type=i32,
    body=[
        ast.Return(value=ast.BinOp(
            left=ast.Name(id='x', ctx=ast.Load()),
            op=ast.Mult(),
            right=ast.Name(id='y', ctx=ast.Load()),
        ))
    ],
)

meta_mul = meta.compile_generated(gen_mul, user_globals=globals())

# --- Test 3: compile_generated sub ---

gen_sub = meta.func(
    name="meta_sub",
    params=[("a", i32), ("b", i32)],
    return_type=i32,
    body=[
        ast.Return(value=ast.BinOp(
            left=ast.Name(id='a', ctx=ast.Load()),
            op=ast.Sub(),
            right=ast.Name(id='b', ctx=ast.Load()),
        ))
    ],
)

meta_sub = meta.compile_generated(gen_sub, user_globals=globals())

# --- Test 4: compile_ast with suffix ---

add_ast_i64 = _make_add_ast()
add_ast_i64.name = "meta_add_typed"
ast.fix_missing_locations(add_ast_i64)

meta_add_i64 = meta.compile_ast(
    add_ast_i64,
    param_types={'a': i64, 'b': i64},
    return_type=i64,
    suffix=i64,
    user_globals=globals(),
)

# --- Test 5 & 6: quote_expr (AST-only, no compilation needed) ---

@meta.quote_expr
def add_expr(x, y):
    return x + y

@meta.quote_expr
def scalar_expr(x, y):
    return x + y

# --- Test 7: quote_func + compile (design doc 15.2) ---

@meta.quote_func
def add_template(ret_ty):
    def generated(x, y) -> ret_ty:
        tmp = x + y
        return tmp

add_i32_frag = add_template(i32).with_name("add_i32")

add_i32_fn = meta.compile_ast(
    add_i32_frag.to_ast(),
    param_types={"x": i32, "y": i32},
    return_type=i32,
    user_globals=globals(),
)

# --- Test 8: meta.func builder + compile ---

def make_doubler():
    body = [
        ast.Return(value=ast.BinOp(
            left=ast.Name(id='n', ctx=ast.Load()),
            op=ast.Add(),
            right=ast.Name(id='n', ctx=ast.Load()),
        ))
    ]
    return meta.func(
        name="meta_double",
        params=[("n", i32)],
        return_type=i32,
        body=body,
    )


gen_double = make_doubler()
meta_double = meta.compile_generated(gen_double, user_globals=globals())

# --- Test 9: cross-call ---

@compile
def base_add(a: i32, b: i32) -> i32:
    return a + b


cross_call_body = [
    ast.Return(value=ast.Call(
        func=ast.Name(id='base_add', ctx=ast.Load()),
        args=[
            ast.Name(id='x', ctx=ast.Load()),
            ast.Name(id='x', ctx=ast.Load()),
        ],
        keywords=[],
    ))
]

cross_call_ast = ast.FunctionDef(
    name="meta_cross_call",
    args=ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg='x')],
        vararg=None, kwonlyargs=[], kw_defaults=[],
        kwarg=None, defaults=[],
    ),
    body=cross_call_body,
    decorator_list=[],
    returns=None,
    lineno=1, col_offset=0,
)
ast.fix_missing_locations(cross_call_ast)

meta_cross = meta.compile_ast(
    cross_call_ast,
    param_types={'x': i32},
    return_type=i32,
    user_globals={**globals(), 'base_add': base_add},
)

# --- Test 10: quote_func with type_expr ---

@meta.quote_func
def typed_template(T):
    def neg(x) -> T:
        return 0 - x


neg_frag = typed_template(meta.type_expr(i32)).with_name("neg_i32")

neg_i32 = meta.compile_ast(
    neg_frag.to_ast(),
    param_types={"x": i32},
    return_type=i32,
    user_globals={**globals(), **getattr(neg_frag, '_user_globals', {})},
)


# --- Test 11: compile_factory basic ---

@meta.compile_factory
def make_adder(offset):
    return meta.func(
        name="adder",
        params=[("x", i32)],
        return_type=i32,
        body=[
            ast.Return(value=ast.BinOp(
                left=ast.Name(id='x', ctx=ast.Load()),
                op=ast.Add(),
                right=ast.Constant(value=offset),
            ))
        ],
    )


# Instantiate factories at module level (before any test execution)
adder_10 = make_adder(10)
adder_20 = make_adder(20)

# --- Test 12: compile_factory cache ---
adder_10_again = make_adder(10)  # Should be cached


# ============================================================================
# Test functions (no compilation, only execution)
# ============================================================================

def test_compile_ast_basic():
    """Test compile_ast with a hand-crafted AST"""
    assert meta_add(10, 20) == 30
    assert meta_add(5, 7) == 12
    assert meta_add(-3, 8) == 5
    print("OK test_compile_ast_basic passed")


def test_compile_generated_basic():
    """Test compile_generated with a GeneratedFunction"""
    assert meta_mul(3, 4) == 12
    assert meta_mul(10, 10) == 100
    assert meta_mul(-2, 5) == -10
    print("OK test_compile_generated_basic passed")


def test_compile_generated_sub():
    """Test compile_generated with subtraction"""
    assert meta_sub(10, 3) == 7
    assert meta_sub(5, 5) == 0
    assert meta_sub(0, 5) == -5
    print("OK test_compile_generated_sub passed")


def test_compile_ast_with_suffix():
    """Test compile_ast with a suffix for specialization"""
    assert meta_add_i64(100, 200) == 300
    assert meta_add_i64(1000000, 2000000) == 3000000
    print("OK test_compile_ast_with_suffix passed")


def test_quote_expr():
    """Test quote_expr: param-based holes with ref() binding helpers"""
    expr_ast = add_expr(meta.ref("a"), meta.ref("b"))
    assert isinstance(expr_ast, meta.MetaFragment)
    assert expr_ast.kind == meta.FragmentKind.EXPR
    expr = expr_ast.as_expr
    assert isinstance(expr, ast.BinOp)
    assert isinstance(expr.left, ast.Name)
    assert expr.left.id == "a"
    assert isinstance(expr.right, ast.Name)
    assert expr.right.id == "b"
    print("OK test_quote_expr passed")


def test_quote_expr_scalar():
    """Test that plain scalars auto-lower to ast.Constant"""
    frag = scalar_expr(10, 20)
    expr = frag.as_expr
    assert isinstance(expr, ast.BinOp)
    assert isinstance(expr.left, ast.Constant)
    assert expr.left.value == 10
    assert isinstance(expr.right, ast.Constant)
    assert expr.right.value == 20
    print("OK test_quote_expr_scalar passed")


def test_quote_func_compile():
    """Test quote_func: type param substitution + with_name + compile"""
    assert add_i32_fn(5, 7) == 12
    assert add_i32_fn(0, 0) == 0
    assert add_i32_fn(-3, 8) == 5
    print("OK test_quote_func_compile passed")


def test_meta_func_builder():
    """Test meta.func() builder creates valid GeneratedFunction"""
    assert meta_double(5) == 10
    assert meta_double(0) == 0
    assert meta_double(-3) == -6
    print("OK test_meta_func_builder passed")


def test_cross_call():
    """Test meta-generated function calling a @compile function"""
    assert meta_cross(5) == 10
    assert meta_cross(0) == 0
    assert meta_cross(-7) == -14
    print("OK test_cross_call passed")


def test_quote_func_type_expr():
    """Test quote_func with type_expr() for explicit type binding"""
    assert neg_i32(5) == -5
    assert neg_i32(0) == 0
    assert neg_i32(-3) == 3
    print("OK test_quote_func_type_expr passed")


def test_compile_factory_basic():
    """Test compile_factory produces working compiled functions"""
    assert adder_10(5) == 15
    assert adder_10(0) == 10
    assert adder_10(-3) == 7
    assert adder_20(5) == 25
    assert adder_20(0) == 20
    print("OK test_compile_factory_basic passed")


def test_compile_factory_cache():
    """Test compile_factory caches same-arg results"""
    assert adder_10_again is adder_10
    print("OK test_compile_factory_cache passed")


def test_wrapper_binding_state_split():
    """Compiled wrappers should expose semantic info and binding separately."""
    assert hasattr(meta_add, "_binding")
    assert meta_add._binding is meta_add._state
    assert meta_add._signature is meta_add._func_info
    assert meta_add._func_info.binding_state is meta_add._binding
    assert not hasattr(meta_add._binding, "current_function")
    print("OK test_wrapper_binding_state_split passed")


# ============================================================================
# Run all tests
# ============================================================================

if __name__ == '__main__':
    test_compile_ast_basic()
    test_compile_generated_basic()
    test_compile_generated_sub()
    test_compile_ast_with_suffix()
    test_quote_expr()
    test_quote_expr_scalar()
    test_quote_func_compile()
    test_meta_func_builder()
    test_cross_call()
    test_quote_func_type_expr()
    test_compile_factory_basic()
    test_compile_factory_cache()
    test_wrapper_binding_state_split()
    print("\nAll meta compile tests passed!")
