#!/usr/bin/env python3
"""
Integration tests for lambda support (compile-level closures)

Lambda lowers to the closure machinery:
    lambda a, b: EXPR  ===  def _anon(a, b): return EXPR

Semantics pinned here:
- Calling a lambda inline-expands its body at the call site (zero-cost).
- Free variables are captured by reference: the expanded body reads the
  enclosing frame at the call site.
- Default arguments are evaluated eagerly at the lambda definition site
  (Python semantics), giving the eager-capture idiom `lambda x=x: ...`.
- A lambda is a compile-level callable, so defer(lambda: ...) works.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc.decorators.compile import compile
from pythoc.builtin_entities import void, i32, ptr, defer, linear, consume, func, defer, linear, consume, func
from pythoc.builtin_entities.instantiate import instantiate
from pythoc.logger import set_raise_on_error
from pythoc.build.output_manager import flush_all_pending_outputs, clear_failed_group

set_raise_on_error(True)


# ============================================================================
# Test 1: Basic lambda called in place
# ============================================================================
@compile(suffix="lambda_basic")
def lambda_basic(n: i32) -> i32:
    inc = lambda x: x + 1
    return inc(n)


def test_lambda_basic():
    assert lambda_basic(1) == 2
    assert lambda_basic(41) == 42
    assert lambda_basic(-1) == 0
    print("OK test_lambda_basic passed")


# ============================================================================
# Test 2: Multi-argument lambda
# ============================================================================
@compile(suffix="lambda_multi_arg")
def lambda_multi_arg(a: i32, b: i32) -> i32:
    add = lambda x, y: x + y
    return add(a, b)


def test_lambda_multi_arg():
    assert lambda_multi_arg(3, 4) == 7
    assert lambda_multi_arg(-5, 5) == 0
    print("OK test_lambda_multi_arg passed")


# ============================================================================
# Test 3: Free variables are captured by reference
# ============================================================================
@compile(suffix="lambda_capture_by_ref")
def lambda_capture_by_ref() -> i32:
    base: i32 = 10
    add_base = lambda x: x + base
    base = 20  # reassignment AFTER definition is visible at the call site
    return add_base(1)


def test_lambda_capture_by_ref():
    assert lambda_capture_by_ref() == 21
    print("OK test_lambda_capture_by_ref passed")


# ============================================================================
# Test 4: Default arguments capture eagerly at the definition site
# ============================================================================
@compile(suffix="lambda_default_eager")
def lambda_default_eager() -> i32:
    base: i32 = 10
    add_base = lambda x, b=base: x + b
    base = 20  # does NOT affect the eagerly bound default
    return add_base(1)


@compile(suffix="lambda_default_partial")
def lambda_default_partial() -> i32:
    add = lambda x, y, scale=3: (x + y) * scale
    return add(1, 2)


def test_lambda_default_eager():
    assert lambda_default_eager() == 11
    assert lambda_default_partial() == 9
    print("OK test_lambda_default_eager passed")


# ============================================================================
# Test 5: Lambda called inside a loop
# ============================================================================
@compile(suffix="lambda_in_loop")
def lambda_in_loop(n: i32) -> i32:
    offset: i32 = 100
    add_offset = lambda x: x + offset
    total: i32 = 0
    i: i32 = 0
    while i < n:
        total = total + add_offset(i)
        i = i + 1
    return total


def test_lambda_in_loop():
    # sum(i + 100 for i in range(3)) = 100 + 101 + 102 = 303
    assert lambda_in_loop(3) == 303
    assert lambda_in_loop(0) == 0
    print("OK test_lambda_in_loop passed")


# ============================================================================
# Test 6: defer(lambda: ...) - lambda as a compile-level deferred callable
# ============================================================================
@compile(suffix="lambda_defer_order")
def lambda_defer_order() -> i32:
    result: i32 = 0

    def poke(p: ptr[i32], d: i32) -> void:
        p[0] = p[0] * 10 + d

    if result == 0:
        defer(lambda: poke(ptr(result), 1))
        defer(lambda: poke(ptr(result), 2))

    # The if-scope exits before the return expression is evaluated, so the
    # deferred lambdas run first (LIFO: 2 then 1) and the result is visible.
    return result


def test_lambda_defer_order():
    assert lambda_defer_order() == 21
    print("OK test_lambda_defer_order passed")


# ============================================================================
# Test 7: Linear token consumed inside a deferred lambda
# ============================================================================
@compile(suffix="lambda_defer_linear")
def lambda_defer_linear() -> i32:
    t = linear()
    defer(lambda: consume(t))
    return 0


def test_lambda_defer_linear():
    assert lambda_defer_linear() == 0
    print("OK test_lambda_defer_linear passed")


# ============================================================================
# Error tests
# ============================================================================
def test_lambda_vararg_error():
    """lambda with *args must produce a lambda-specific error"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'lambda_bad_vararg')
    try:
        @compile(suffix="lambda_bad_vararg")
        def bad() -> i32:
            f = lambda *xs: 1
            return f(1)

        flush_all_pending_outputs()
        print("FAIL test_lambda_vararg_error failed - should have raised TypeError")
    except TypeError as e:
        if "lambda" in str(e).lower() and "*args" in str(e):
            print(f"OK test_lambda_vararg_error passed: {e}")
        else:
            print(f"FAIL test_lambda_vararg_error failed - wrong error: {e}")
    finally:
        clear_failed_group(group_key)


def test_nested_decorated_def_error():
    """A decorated nested def must be rejected, not silently treated as closure"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'nested_decorated')
    try:
        @compile(suffix="nested_decorated")
        def bad() -> i32:
            @compile
            def inner(x: i32) -> i32:
                return x + 1

            return inner(1)

        flush_all_pending_outputs()
        print("FAIL test_nested_decorated_def_error failed - should have raised TypeError")
    except TypeError as e:
        if "decorator" in str(e).lower() and "closure" in str(e).lower():
            print(f"OK test_nested_decorated_def_error passed: {e}")
        else:
            print(f"FAIL test_nested_decorated_def_error failed - wrong error: {e}")
    finally:
        clear_failed_group(group_key)


# ============================================================================
# Boundary errors: compile-level callables used as runtime values
# ============================================================================
def _check_boundary_error(label, suffix, define_func):
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', suffix)
    try:
        define_func()
        flush_all_pending_outputs()
        print(f"FAIL {label} failed - should have raised TypeError")
    except TypeError as e:
        if "compile-level callable" in str(e).lower():
            print(f"OK {label} passed: {e}")
        else:
            print(f"FAIL {label} failed - wrong error: {e}")
    finally:
        clear_failed_group(group_key)


def test_lambda_to_func_pointer_error():
    """Assigning a lambda to a func[...] variable hits the boundary error"""
    def define():
        @compile(suffix="lambda_as_func_ptr")
        def bad() -> i32:
            f: func[i32, i32] = lambda x: x + 1
            return f(1)
    _check_boundary_error(
        "test_lambda_to_func_pointer_error", "lambda_as_func_ptr", define)


def test_closure_to_func_pointer_error():
    """Assigning a nested-def closure to a func[...] variable hits the same error"""
    def define():
        @compile(suffix="closure_as_func_ptr")
        def bad() -> i32:
            base: i32 = 10

            def add_base(x: i32) -> i32:
                return x + base

            f: func[i32, i32] = add_base
            return f(1)
    _check_boundary_error(
        "test_closure_to_func_pointer_error", "closure_as_func_ptr", define)


def test_lambda_returned_as_value_error():
    """Returning a lambda where i32 is expected hits the boundary error"""
    def define():
        @compile(suffix="lambda_as_return")
        def bad() -> i32:
            inc = lambda x: x + 1
            return inc
    _check_boundary_error(
        "test_lambda_returned_as_value_error", "lambda_as_return", define)


# ============================================================================
# Test 8: instantiate(lambda) - per-call-site type specialization
# ============================================================================
@compile(suffix="lambda_inst_untyped")
def lambda_inst_untyped() -> i32:
    api = instantiate(lambda x: x * 2)
    o = api.init()
    return api.call(ptr(o), i32(7))


@compile(suffix="lambda_inst_capture")
def lambda_inst_capture() -> i32:
    base: i32 = 100
    api = instantiate(lambda x: x + base)
    o = api.init()
    return api.call(ptr(o), i32(5))


@compile(suffix="lambda_inst_default")
def lambda_inst_default() -> i32:
    base: i32 = 10
    api = instantiate(lambda x, b=base: x + b)
    base = 99  # default was captured eagerly at the lambda definition
    o = api.init()
    return api.call(ptr(o), i32(5))


def test_lambda_instantiate():
    assert lambda_inst_untyped() == 14
    assert lambda_inst_capture() == 105
    assert lambda_inst_default() == 15
    print("OK test_lambda_instantiate passed")


def main():
    print("Lambda Integration Tests")
    print("=" * 60)

    try:
        test_lambda_basic()
        test_lambda_multi_arg()
        test_lambda_capture_by_ref()
        test_lambda_default_eager()
        test_lambda_in_loop()
        test_lambda_defer_order()
        test_lambda_defer_linear()
        test_lambda_vararg_error()
        test_nested_decorated_def_error()
        test_lambda_to_func_pointer_error()
        test_closure_to_func_pointer_error()
        test_lambda_returned_as_value_error()
        test_lambda_instantiate()

        print()
        print("=" * 60)
        print("All lambda tests passed! OK")
        return 0
    except Exception as e:
        print(f"\nFAIL Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
