#!/usr/bin/env python3
"""Compile-time failure tests for parametric polymorphism.

These tests verify that illegal uses of ``param`` are rejected with a clear
error.  Each test should raise during decoration or during compilation.
"""

from __future__ import annotations

import sys
import os

from pythoc import compile, param, i32, i64, f64, ptr, struct
from pythoc.build.output_manager import flush_all_pending_outputs, clear_failed_group
from pythoc.logger import set_raise_on_error

# Enable exception raising for tests that expect to catch exceptions.
set_raise_on_error(True)


source_file = os.path.abspath(__file__)


def _group_key(suffix: str):
    return (source_file, 'module', suffix)


def _expect_type_error(fn, must_contain: str) -> bool:
    try:
        fn()
        print("  FAIL: Should have raised TypeError but did not")
        return False
    except TypeError as e:
        msg = str(e)
        if must_contain in msg:
            print(f"  PASS: Got expected TypeError: {e}")
            return True
        print(f"  FAIL: Got TypeError but message mismatch: {e}")
        return False


def _expect_not_implemented_error(fn, must_contain: str) -> bool:
    try:
        fn()
        print("  FAIL: Should have raised NotImplementedError but did not")
        return False
    except NotImplementedError as e:
        msg = str(e)
        if must_contain in msg:
            print(f"  PASS: Got expected NotImplementedError: {e}")
            return True
        print(f"  FAIL: Got NotImplementedError but message mismatch: {e}")
        return False


# Parametric helpers used by the failure tests.  They must be defined at module
# level before any native execution begins.

@compile
def identity(T: param, x: T) -> T:
    return x


@compile
def param_at_end(x: i32, T: param) -> T:
    return T(x)


@compile
def runtime_value_helper() -> i32:
    return 42


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------

def test_param_as_return_type() -> bool:
    print("Test: 'param' as function return type (should fail)...")
    return _expect_type_error(
        lambda: _compile_bad_return(),
        "'param' cannot be used as a runtime type",
    )


def _compile_bad_return():
    @compile(suffix="bad_return")
    def bad_return() -> param:
        return 0
    flush_all_pending_outputs()
    clear_failed_group(_group_key("bad_return"))


def test_param_as_local_variable_type() -> bool:
    print("Test: 'param' as local variable type (should fail)...")
    return _expect_type_error(
        lambda: _compile_bad_local(),
        "'param' cannot be used as a runtime type",
    )


def _compile_bad_local():
    @compile(suffix="bad_local")
    def bad_local() -> i32:
        x: param = 0  # ERROR: param cannot be a runtime variable type
        return x

    flush_all_pending_outputs()
    clear_failed_group(_group_key("bad_local"))


def test_param_as_struct_field() -> bool:
    print("Test: 'param' as struct field type (should fail)...")
    return _expect_type_error(
        lambda: _compile_bad_struct(),
        "'param' cannot be used as a runtime type",
    )


def _compile_bad_struct():
    @compile(suffix="bad_struct")
    class BadStruct(struct):
        value: param  # ERROR: param cannot be a field type

    flush_all_pending_outputs()
    clear_failed_group(_group_key("bad_struct"))


def test_parametric_wrapper_as_first_class_value() -> bool:
    print("Test: parametric wrapper used as first-class value (should fail)...")
    try:
        @compile(suffix="bad_first_class")
        def bad_first_class() -> i32:
            # param_at_end is still a parametric factory; it cannot be passed
            # as a runtime pointer argument.
            p: ptr[i32] = param_at_end  # ERROR
            return p[0]

        flush_all_pending_outputs()
        print("  FAIL: Should have raised TypeError but did not")
        return False
    except TypeError as e:
        if "cannot be used as a first-class value" in str(e):
            print(f"  PASS: Got expected error: {e}")
            return True
        print(f"  FAIL: Got unexpected error: {e}")
        return False
    finally:
        clear_failed_group(_group_key("bad_first_class"))


def test_keyword_arguments_rejected() -> bool:
    print("Test: keyword arguments on parametric wrapper (should fail)...")
    try:
        # Phase 1 only supports positional arguments.
        result = param_at_end(x=7, T=i32)
        print(f"  FAIL: Should have raised TypeError but got {result}")
        return False
    except TypeError as e:
        if "positional arguments" in str(e):
            print(f"  PASS: Got expected error: {e}")
            return True
        print(f"  FAIL: Got unexpected error: {e}")
        return False


def test_param_with_default_value() -> bool:
    print("Test: parametric parameter with default value (should fail)...")
    try:
        @compile(suffix="bad_default")
        def bad_default(T: param = i32, x: i32 = 0) -> i32:  # ERROR
            return x

        print("  FAIL: Should have raised TypeError but did not")
        return False
    except TypeError as e:
        if "cannot have a default value" in str(e):
            print(f"  PASS: Got expected error: {e}")
            return True
        print(f"  FAIL: Got unexpected error: {e}")
        return False
    finally:
        clear_failed_group(_group_key("bad_default"))


def test_varargs_parametric() -> bool:
    print("Test: parametric *args (should fail)...")
    return _expect_not_implemented_error(
        lambda: _compile_bad_varargs(),
        "parametric *args is not supported",
    )


def _compile_bad_varargs():
    @compile(suffix="bad_varargs")
    def bad_varargs(*args: param) -> i32:  # ERROR
        return 0

    clear_failed_group(_group_key("bad_varargs"))


def test_kwargs_parametric() -> bool:
    print("Test: parametric **kwargs (should fail)...")
    return _expect_not_implemented_error(
        lambda: _compile_bad_kwargs(),
        "parametric **kwargs is not supported",
    )


def _compile_bad_kwargs():
    @compile(suffix="bad_kwargs")
    def bad_kwargs(**kwargs: param) -> i32:  # ERROR
        return 0

    clear_failed_group(_group_key("bad_kwargs"))


def test_runtime_value_as_param_argument() -> bool:
    print("Test: runtime PythoC value used as param argument (should fail)...")
    try:
        @compile(suffix="bad_runtime_param")
        def bad_runtime_param() -> i32:
            x: i32 = runtime_value_helper()
            # x is a runtime value, but identity expects a compile-time param.
            return identity(x, x)  # ERROR

        flush_all_pending_outputs()
        print("  FAIL: Should have raised TypeError but did not")
        return False
    except TypeError as e:
        if "must be a compile-time Python value" in str(e):
            print(f"  PASS: Got expected error: {e}")
            return True
        print(f"  FAIL: Got unexpected error: {e}")
        return False
    finally:
        clear_failed_group(_group_key("bad_runtime_param"))


def test_unhashable_python_value_as_param() -> bool:
    print("Test: unhashable Python value as param argument (should fail)...")
    try:
        identity([i32], 42)  # ERROR: list is not hashable
        print("  FAIL: Should have raised TypeError but did not")
        return False
    except TypeError as e:
        if "hashable" in str(e).lower() or "unhashable" in str(e).lower():
            print(f"  PASS: Got expected error: {e}")
            return True
        print(f"  FAIL: Got unexpected error: {e}")
        return False


def test_wrong_number_of_arguments() -> bool:
    print("Test: wrong number of arguments to parametric wrapper (should fail)...")
    try:
        result = param_at_end(1, i32, 2)  # ERROR: too many args
        print(f"  FAIL: Should have raised TypeError but got {result}")
        return False
    except TypeError as e:
        if "positional arguments" in str(e):
            print(f"  PASS: Got expected error: {e}")
            return True
        print(f"  FAIL: Got unexpected error: {e}")
        return False


def test_too_few_arguments() -> bool:
    print("Test: too few arguments to parametric wrapper (should fail)...")
    try:
        # Providing only the parametric arg is partial application and is
        # explicitly allowed; providing none of the required args is not.
        result = identity()  # ERROR: missing T and x
        print(f"  FAIL: Should have raised TypeError but got {result}")
        return False
    except TypeError as e:
        if "positional arguments" in str(e):
            print(f"  PASS: Got expected error: {e}")
            return True
        print(f"  FAIL: Got unexpected error: {e}")
        return False


def main() -> int:
    print("=" * 70)
    print("Parametric polymorphism failure tests")
    print("=" * 70)
    print()

    tests = [
        test_param_as_return_type,
        test_param_as_local_variable_type,
        test_param_as_struct_field,
        test_parametric_wrapper_as_first_class_value,
        test_keyword_arguments_rejected,
        test_param_with_default_value,
        test_varargs_parametric,
        test_kwargs_parametric,
        test_runtime_value_as_param_argument,
        test_unhashable_python_value_as_param,
        test_wrong_number_of_arguments,
        test_too_few_arguments,
    ]

    results = []
    for test_func in tests:
        try:
            results.append(test_func())
        except Exception as e:
            print(f"  UNEXPECTED ERROR in {test_func.__name__}: {type(e).__name__}: {e}")
            results.append(False)
        print()

    passed = sum(results)
    total = len(results)

    print("=" * 70)
    print(f"Results: {passed}/{total} tests behaved as expected")

    if passed == total:
        print("SUCCESS: All parametric polymorphism errors detected!")
        return 0
    print("PARTIAL: Some error cases were not properly handled")
    return 1


if __name__ == "__main__":
    sys.exit(main())
