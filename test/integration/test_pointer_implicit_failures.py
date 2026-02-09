#!/usr/bin/env python3
"""Compile-time failure tests for implicit pointer coercions.

These tests verify that implicit int -> ptr conversions are rejected.
Explicit casts (e.g. ptr[T](x)) may still allow inttoptr, but implicit
coercions in assignments / argument passing / returns must not.
"""

import sys
import os

from pythoc import compile, i32, ptr
from pythoc.builtin_entities.types import void
from pythoc.build.output_manager import flush_all_pending_outputs, clear_failed_group
from pythoc.logger import set_raise_on_error

# Enable exception raising for tests that expect to catch exceptions
set_raise_on_error(True)


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


def test_implicit_int_to_ptr_assignment_rejected() -> bool:
    print("Test: implicit Python int -> ptr in assignment (should fail)...")
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_implicit_int_to_ptr_assign')

    def _compile_bad():
        @compile(suffix="bad_implicit_int_to_ptr_assign")
        def should_fail() -> i32:
            p: ptr[i32] = 123  # ERROR: implicit int -> ptr is not allowed
            return i32(p)

        flush_all_pending_outputs()

    try:
        return _expect_type_error(_compile_bad, "Cannot implicitly convert Python int constant")
    finally:
        clear_failed_group(group_key)


def test_implicit_int_to_ptr_call_arg_rejected() -> bool:
    print("Test: implicit Python int -> ptr in call arg (should fail)...")
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_implicit_int_to_ptr_call')

    @compile
    def takes_ptr(p: ptr[i32]) -> i32:
        return 0

    def _compile_bad():
        @compile(suffix="bad_implicit_int_to_ptr_call")
        def should_fail() -> i32:
            return takes_ptr(123)  # ERROR

        flush_all_pending_outputs()

    try:
        return _expect_type_error(_compile_bad, "Cannot implicitly convert Python int constant")
    finally:
        clear_failed_group(group_key)


def test_implicit_int_to_ptr_return_rejected() -> bool:
    print("Test: implicit Python int -> ptr in return (should fail)...")
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_implicit_int_to_ptr_return')

    def _compile_bad():
        @compile(suffix="bad_implicit_int_to_ptr_return")
        def should_fail() -> ptr[i32]:
            return 123  # ERROR

        flush_all_pending_outputs()

    try:
        return _expect_type_error(_compile_bad, "Cannot implicitly convert Python int constant")
    finally:
        clear_failed_group(group_key)


def test_ptr_void_name_is_canonical() -> bool:
    print("Test: ptr[void] canonical spelling (should be ptr[void])...")
    try:
        name = ptr[void].get_name()
        if name == "ptr[void]":
            print("  PASS: ptr[void].get_name() == ptr[void]")
            return True
        print(f"  FAIL: ptr[void].get_name() == {name}")
        return False
    except Exception as e:
        print(f"  FAIL: Unexpected exception: {type(e).__name__}: {e}")
        return False


def main() -> int:
    print("=" * 70)
    print("Implicit pointer coercion failure tests")
    print("=" * 70)
    print()

    tests = [
        test_implicit_int_to_ptr_assignment_rejected,
        test_implicit_int_to_ptr_call_arg_rejected,
        test_implicit_int_to_ptr_return_rejected,
        test_ptr_void_name_is_canonical,
    ]

    results = []
    for test_func in tests:
        try:
            results.append(test_func())
        except Exception as e:
            print(f"  UNEXPECTED ERROR in {test_func.__name__}: {e}")
            results.append(False)
        print()

    passed = sum(results)
    total = len(results)

    print("=" * 70)
    print(f"Results: {passed}/{total} tests behaved as expected")

    if passed == total:
        print("SUCCESS: All implicit pointer coercion errors detected!")
        return 0
    print("PARTIAL: Some error cases were not properly handled")
    return 1


if __name__ == "__main__":
    sys.exit(main())
