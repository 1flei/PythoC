#!/usr/bin/env python3
"""
Integration tests for @inline decorator

Tests the inline decorator which allows PC functions to be executed
inline at call sites, generating IR instructions directly in the
caller's basic block.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc import compile, inline, i32, i64, bool


# ============================================================================
# Test 1: Simple inline function
# ============================================================================

@inline
def add(a, b) -> i32:
    return a + b


@compile
def simple_inline(x: i32, y: i32) -> i32:
    result: i32 = add(x, y)
    return result


def test_simple():
    """Test basic inline function"""
    assert simple_inline(10, 20) == 30
    assert simple_inline(5, 7) == 12
    assert simple_inline(-3, 8) == 5
    print("OK test_simple passed")


# ============================================================================
# Test 2: Inline with if statement
# ============================================================================

@inline
def clamp(value: i32, min_val: i32, max_val: i32) -> i32:
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    return value


@compile
def clamp_runtime(x: i32) -> i32:
    return clamp(x, 0, 100)


def test_clamp():
    """Test inline function with if statement"""
    assert clamp_runtime(-10) == 0
    assert clamp_runtime(50) == 50
    assert clamp_runtime(150) == 100
    assert clamp_runtime(0) == 0
    assert clamp_runtime(100) == 100
    print("OK test_clamp passed")


# ============================================================================
# Test 3: Inline with multiple statements
# ============================================================================

@inline
def abs_diff(a: i32, b: i32) -> i32:
    diff: i32 = a - b
    if diff < 0:
        diff = -diff
    return diff


@compile
def abs_diff_runtime(x: i32, y: i32) -> i32:
    result: i32 = abs_diff(x, y)
    return result


def test_abs_diff():
    """Test inline function with local variables"""
    assert abs_diff_runtime(10, 5) == 5
    assert abs_diff_runtime(5, 10) == 5
    assert abs_diff_runtime(-3, 7) == 10
    assert abs_diff_runtime(0, 0) == 0
    print("OK test_abs_diff passed")


# ============================================================================
# Test 4: Multiple inline calls
# ============================================================================

@inline
def square(x: i32) -> i32:
    return x * x


@compile
def multiple_inline(a: i32, b: i32) -> i32:
    x: i32 = square(a)
    y: i32 = square(b)
    return x + y


def test_multiple():
    """Test multiple inline function calls"""
    print(multiple_inline(3, 4))
    assert multiple_inline(3, 4) == 25  # 9 + 16
    assert multiple_inline(5, 5) == 50  # 25 + 25
    assert multiple_inline(0, 10) == 100  # 0 + 100
    print("OK test_multiple passed")


# ============================================================================
# Test 5: Nested inline calls
# ============================================================================

@inline
def double(x: i32) -> i32:
    return x + x


@inline
def quadruple(x: i32) -> i32:
    temp: i32 = double(x)
    return double(temp)


@compile
def nested_inline(x: i32) -> i32:
    return quadruple(x)


def test_nested():
    """Test nested inline function calls"""
    assert nested_inline(5) == 20
    assert nested_inline(10) == 40
    assert nested_inline(0) == 0
    print("OK test_nested passed")


# ============================================================================
# Test 6: Inline with arithmetic
# ============================================================================

@inline
def compute(a: i32, b: i32, c: i32) -> i32:
    temp1: i32 = a + b
    temp2: i32 = temp1 * c
    temp3: i32 = temp2 - a
    return temp3


@compile
def arithmetic_inline(x: i32, y: i32, z: i32) -> i32:
    result: i32 = compute(x, y, z)
    return result


def test_arithmetic():
    """Test inline function with complex arithmetic"""
    assert arithmetic_inline(2, 3, 4) == 18  # (2+3)*4 - 2 = 18
    assert arithmetic_inline(5, 5, 2) == 15  # (5+5)*2 - 5 = 15
    assert arithmetic_inline(1, 1, 1) == 1   # (1+1)*1 - 1 = 1
    print("OK test_arithmetic passed")


# ============================================================================
# Test 7: Compile-time constant evaluation
# ============================================================================

@inline
def max_val(a: i32, b: i32) -> i32:
    if a > b:
        return a
    return b


@compile
def constant_inline() -> i32:
    # This should use pure Python evaluation at compile time
    result: i32 = max_val(10, 20)
    return result


def test_constant():
    """Test that compile-time constants use pure Python"""
    assert constant_inline() == 20
    print("OK test_constant passed")


# ============================================================================
# Test 8: Inline with comparison
# ============================================================================

@inline
def is_in_range(value: i32, low: i32, high: i32) -> i32:
    if value >= low:
        if value <= high:
            return 1
    return 0


@compile
def range_check(x: i32) -> i32:
    result: i32 = is_in_range(x, 10, 20)
    return result


def test_range():
    """Test inline function with nested if statements"""
    assert range_check(5) == 0
    assert range_check(10) == 1
    assert range_check(15) == 1
    assert range_check(20) == 1
    assert range_check(25) == 0
    print("OK test_range passed")


# ============================================================================
# Test 9: Complex example combining multiple features
# ============================================================================

@inline
def safe_divide(numerator: i32, denominator: i32, default: i32) -> i32:
    if denominator == 0:
        return default
    return numerator / denominator


@inline
def another_abs(x) -> i32:
    if x < 0:
        x = -x
    return x

@compile
def complex_calc(a: i32, b: i32, c: i32) -> i32:
    # Combine multiple inline functions
    sum_val: i32 = add(a, b)
    clamped: i32 = clamp(sum_val, 0, 100)
    result: i32 = safe_divide(clamped, c, -1)
    return result

@compile
def abs_example(x: i32) -> i32:
    return another_abs(x)

def test_complex_example():
    """Test complex combination of inline functions"""
    assert complex_calc(10, 20, 5) == 6   # (10+20) clamped to 30, 30/5 = 6
    assert complex_calc(50, 60, 2) == 50  # (50+60) clamped to 100, 100/2 = 50
    assert complex_calc(10, 20, 0) == -1  # Division by zero, return default

    assert abs_example(-10) == 10
    assert abs_example(10) == 10
    assert abs_example(0) == 0
    print("OK test_complex_example passed")

# ============================================================================
# Main test runner
# ============================================================================

def main():
    print("=" * 70)
    print("Testing @inline decorator")
    print("=" * 70)
    print()
    
    tests = [
        ("Simple inline", test_simple),
        ("Inline with if", test_clamp),
        ("Inline with local vars", test_abs_diff),
        ("Multiple inline calls", test_multiple),
        ("Nested inline calls", test_nested),
        ("Inline with arithmetic", test_arithmetic),
        ("Compile-time constants", test_constant),
        ("Inline with comparisons", test_range),
        ("Complex example", test_complex_example),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        print(f"Running: {name}...")
        test_func()
        passed += 1
    
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed > 0:
        sys.exit(1)
    
    print()
    print("Key features demonstrated:")
    print("  OK Basic inline execution")
    print("  OK Control flow (if/elif/else)")
    print("  OK Local variables")
    print("  OK Multiple inline calls")
    print("  OK Nested inline calls")
    print("  OK Complex arithmetic")
    print("  OK Compile-time constant evaluation")
    print("  OK Comparison operations")
    print()
    print("All tests passed! @inline decorator is working correctly.")


if __name__ == "__main__":
    main()
