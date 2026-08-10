#!/usr/bin/env python3
"""
Integration tests for closure functionality

Covered here:
- Simple closures with single/multiple captures (called at top level)
- Closures with control flow (if statements)
- Closures called inside loops
- Nested closures

See also: test_lambda.py (lambda syntax lowering to the same machinery),
test_nested_loops_and_funcs.py and test_defer_advanced.py for more
closure-in-loop / nested-closure / closure+defer coverage.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc import compile, i32, bool
from pythoc.logger import set_raise_on_error
from pythoc.build.output_manager import flush_all_pending_outputs, clear_failed_group

set_raise_on_error(True)


# ============================================================================
# Test 1: Simple closure with single capture
# ============================================================================
@compile
def simple_closure_test(n: i32) -> i32:
    base: i32 = 100
    
    def add_base(x: i32) -> i32:
        return x + base
    
    result: i32 = add_base(n)
    return result


def test_simple_closure():
    assert simple_closure_test(10) == 110
    assert simple_closure_test(0) == 100
    assert simple_closure_test(-50) == 50
    print("OK test_simple_closure passed")


# ============================================================================
# Test 2: Closure with multiple captures
# ============================================================================
@compile
def multi_capture_test(x: i32) -> i32:
    multiplier: i32 = 2
    offset: i32 = 10
    
    def transform(n: i32) -> i32:
        temp: i32 = n * multiplier
        result: i32 = temp + offset
        return result
    
    return transform(x)


def test_multi_capture():
    assert multi_capture_test(5) == 20   # 5 * 2 + 10
    assert multi_capture_test(10) == 30  # 10 * 2 + 10
    assert multi_capture_test(0) == 10   # 0 * 2 + 10
    print("OK test_multi_capture passed")


# ============================================================================
# Test 3: Closure with if statement
# ============================================================================
@compile
def closure_with_if_test(n: i32) -> i32:
    threshold: i32 = 50
    
    def clamp_upper(x: i32) -> i32:
        if x > threshold:
            return threshold
        return x
    
    result: i32 = clamp_upper(n)
    return result


def test_closure_with_if():
    assert closure_with_if_test(30) == 30
    assert closure_with_if_test(60) == 50
    assert closure_with_if_test(50) == 50
    print("OK test_closure_with_if passed")


# ============================================================================
# Test 4: Closure in loop
# ============================================================================
@compile
def closure_in_loop_test(n: i32) -> i32:
    base: i32 = 100

    def add_base(x: i32) -> i32:
        return x + base

    result: i32 = 0
    i: i32 = 0
    while i < n:
        result = add_base(i)
        i = i + 1

    return result


def test_closure_in_loop():
    result = closure_in_loop_test(3)
    assert result == 102  # 2 + 100
    print("OK test_closure_in_loop passed")


# ============================================================================
# Test 5: Nested closures
# ============================================================================
@compile
def nested_closure_test(x: i32) -> i32:
    a: i32 = 10

    def outer(y: i32) -> i32:
        b: i32 = 20

        def inner(z: i32) -> i32:
            return z + a + b

        return inner(y)

    return outer(x)


def test_nested_closure():
    assert nested_closure_test(5) == 35
    assert nested_closure_test(0) == 30
    print("OK test_nested_closure passed")


# ============================================================================
# Test 6: Closure without return annotation (single return)
# ============================================================================
@compile
def unannotated_closure_test(n: i32) -> i32:
    base: i32 = 7

    def add_base(x: i32):
        return x + base

    return add_base(n)


def test_unannotated_closure():
    assert unannotated_closure_test(3) == 10
    assert unannotated_closure_test(-7) == 0
    print("OK test_unannotated_closure passed")


# ============================================================================
# Test 7: Unannotated closure with multiple returns is a clear error
# ============================================================================
def test_unannotated_multi_return_error():
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'unannotated_multi_return')
    try:
        @compile(suffix="unannotated_multi_return")
        def bad(n: i32) -> i32:
            def pick(x: i32):
                if x > 0:
                    return x
                return 0

            return pick(n)

        flush_all_pending_outputs()
        print("FAIL test_unannotated_multi_return_error failed - should have raised")
    except RuntimeError as e:
        if "annotation" in str(e).lower():
            print(f"OK test_unannotated_multi_return_error passed: {e}")
        else:
            print(f"FAIL test_unannotated_multi_return_error failed - wrong error: {e}")
    finally:
        clear_failed_group(group_key)


def main():
    """Run all tests"""
    print("Closure Integration Tests")
    print("=" * 60)
    
    try:
        test_simple_closure()
        test_multi_capture()
        test_closure_with_if()
        test_closure_in_loop()
        test_nested_closure()
        test_unannotated_closure()
        test_unannotated_multi_return_error()

        print()
        print("=" * 60)
        print("All closure tests passed! OK")
        return 0
    except Exception as e:
        print(f"\nFAIL Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

