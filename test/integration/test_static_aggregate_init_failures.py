#!/usr/bin/env python3
"""Failure tests: static initializers must be link-time constants.

Static storage cannot run instructions, so seeds containing runtime-computed
values must be rejected at compile time.
"""

import os
import sys

from pythoc import compile, i32, static, array
from pythoc.build.output_manager import flush_all_pending_outputs, clear_failed_group
from pythoc.logger import set_raise_on_error

# Enable exception raising for tests that expect to catch exceptions
set_raise_on_error(True)


def _expect_error(fn, exc_type, must_contain: str) -> bool:
    try:
        fn()
        print(f"  FAIL: Should have raised {exc_type.__name__} but did not")
        return False
    except exc_type as e:
        msg = str(e)
        if must_contain in msg:
            print(f"  PASS: Got expected {exc_type.__name__}: {e}")
            return True
        print(f"  FAIL: Got {exc_type.__name__} but message mismatch: {e}")
        return False


def test_static_array_runtime_seed_rejected() -> bool:
    print("Test: runtime value in static array seed (should fail)...")
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_static_array_runtime_seed')

    def _compile_bad():
        @compile(suffix="bad_static_array_runtime_seed")
        def should_fail(x: i32) -> i32:
            s: static[array[i32, 2]] = (x + 1, 0)  # ERROR: not a compile-time constant
            return s[0]

        flush_all_pending_outputs()

    try:
        return _expect_error(
            _compile_bad, RuntimeError, "requires compile-time constant initializer")
    finally:
        clear_failed_group(group_key)


def test_static_scalar_runtime_seed_rejected() -> bool:
    print("Test: runtime value in static scalar seed (should fail)...")
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_static_scalar_runtime_seed')

    def _compile_bad():
        @compile(suffix="bad_static_scalar_runtime_seed")
        def should_fail(x: i32) -> i32:
            s: static[i32] = x + 1  # ERROR: not a compile-time constant
            return s

        flush_all_pending_outputs()

    try:
        return _expect_error(
            _compile_bad, RuntimeError, "requires compile-time constant initializer")
    finally:
        clear_failed_group(group_key)


def main() -> int:
    print("=" * 70)
    print("Static aggregate initializer failure tests")
    print("=" * 70)
    print()

    tests = [
        test_static_array_runtime_seed_rejected,
        test_static_scalar_runtime_seed_rejected,
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
        print("SUCCESS: All static initializer errors detected!")
        return 0
    print("PARTIAL: Some error cases were not properly handled")
    return 1


if __name__ == "__main__":
    sys.exit(main())
