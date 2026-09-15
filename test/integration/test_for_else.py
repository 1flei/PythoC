#!/usr/bin/env python3
"""
Test for-else statement

Tests the for-else control flow:
- else executes when loop completes normally (no break)
- else skips when loop exits via break
- else executes when loop body never runs (empty iterator)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc import compile, i32


# Test 1: for-else with normal completion (no break)
@compile
def test_for_else_normal() -> i32:
    """Loop completes normally, else executes"""
    sum: i32 = 0
    for i in [1, 2, 3]:
        sum = sum + i
    else:
        sum = sum + 100  # Should execute
    return sum  # 1+2+3+100 = 106


# Test 2: for-else with break (else skips)
@compile
def test_for_else_with_break() -> i32:
    """Loop exits via break, else skips"""
    sum: i32 = 0
    for i in [1, 2, 3, 4, 5]:
        sum = sum + i
        if i == 3:
            break
    else:
        sum = sum + 100  # Should NOT execute
    return sum  # 1+2+3 = 6


# Test 3: for-else with empty iterator (else executes)
@compile
def test_for_else_empty() -> i32:
    """Empty iterator, else still executes"""
    sum: i32 = 0
    for i in []:
        sum = sum + i  # Never runs
    else:
        sum = sum + 100  # Should execute
    return sum  # 100


# Test 4: for-else with conditional break
@compile
def test_for_else_conditional_break(target: i32) -> i32:
    """Break only if target found"""
    found: i32 = 0
    for i in [10, 20, 30, 40]:
        if i == target:
            found = 1
            break
    else:
        found = -1  # Not found
    return found


# Test 5: nested for-else
@compile
def test_nested_for_else() -> i32:
    """Nested loops with else clauses"""
    result: i32 = 0
    
    # Outer loop completes normally
    for i in [1, 2]:
        # Inner loop breaks
        for j in [10, 20, 30]:
            result = result + j
            if j == 20:
                break
        else:
            result = result + 1000  # Should NOT execute (inner broke)
        result = result + 100  # Always execute after inner loop
    else:
        result = result + 10000  # Should execute (outer completed normally)
    
    # result = (10+20+100) + (10+20+100) + 10000 = 130 + 130 + 10000 = 10260
    return result


# Test 6: for-else with continue (else still executes)
@compile
def test_for_else_with_continue() -> i32:
    """Continue doesn't prevent else execution"""
    sum: i32 = 0
    for i in [1, 2, 3, 4, 5]:
        if i == 3:
            continue  # Skip 3
        sum = sum + i
    else:
        sum = sum + 100  # Should execute (no break)
    return sum  # 1+2+4+5+100 = 112


# Test 7: for-else with early return (else skips)
@compile
def test_for_else_with_return() -> i32:
    """Return exits function, else doesn't execute"""
    sum: i32 = 0
    for i in [1, 2, 3, 4, 5]:
        sum = sum + i
        if i == 3:
            return sum  # Early return
    else:
        sum = sum + 100  # Should NOT execute
    return sum


# Test 8: for-else with yield function (when we have yield support)
@compile
def simple_range(n: i32) -> i32:
    """Generate 0..n-1"""
    i: i32 = 0
    while i < n:
        yield i
        i = i + 1


@compile
def test_for_else_with_yield() -> i32:
    """Test for-else with yield-based iterator"""
    sum: i32 = 0
    for i in simple_range(5):
        sum = sum + i
    else:
        sum = sum + 100  # Should execute
    return sum  # 0+1+2+3+4+100 = 110


@compile
def test_for_else_yield_with_break() -> i32:
    """Test for-else with yield and break"""
    sum: i32 = 0
    for i in simple_range(10):
        sum = sum + i
        if i == 4:
            break
    else:
        sum = sum + 100  # Should NOT execute
    return sum  # 0+1+2+3+4 = 10


@compile
def empty_generator() -> i32:
    """Yield nothing"""
    i: i32 = 0
    while i < 0:
        yield i
        i = i + 1


@compile
def test_for_else_empty_yield() -> i32:
    """Test for-else with empty yield generator"""
    sum: i32 = 0
    for i in empty_generator():
        sum = sum + i
    else:
        sum = sum + 100  # Should execute
    return sum  # 100


# =============================================================================
# while-else tests (same semantics: else runs unless the loop broke)
# =============================================================================

# Test 9: while-else with normal completion (no break)
@compile(suffix="while_else")
def test_while_else_normal(n: i32) -> i32:
    """While loop completes normally, else executes"""
    sum: i32 = 0
    i: i32 = 0
    while i < n:
        sum = sum + i
        i = i + 1
    else:
        sum = sum + 100  # Should execute
    return sum  # n=5: 0+1+2+3+4+100 = 110


# Test 10: while-else with conditional break
@compile(suffix="while_else")
def test_while_else_with_break(target: i32) -> i32:
    """Break only if target found"""
    found: i32 = 0
    i: i32 = 0
    while i < 10:
        if i == target:
            found = 1
            break
        i = i + 1
    else:
        found = -1  # Not found
    return found


# Test 11: while-else with condition false on entry (else executes)
@compile(suffix="while_else")
def test_while_else_empty() -> i32:
    """Loop body never runs, else still executes"""
    sum: i32 = 0
    i: i32 = 0
    while i < 0:
        sum = sum + 1  # Never runs
        i = i + 1
    else:
        sum = sum + 100  # Should execute
    return sum  # 100


# Test 12: while-else with continue (else still executes)
@compile(suffix="while_else")
def test_while_else_with_continue() -> i32:
    """Continue doesn't prevent else execution"""
    sum: i32 = 0
    i: i32 = 0
    while i < 5:
        i = i + 1
        if i == 3:
            continue  # Skip 3
        sum = sum + i
    else:
        sum = sum + 100  # Should execute (no break)
    return sum  # 1+2+4+5+100 = 112


# Test 13: while-else with early return (else skips)
@compile(suffix="while_else")
def test_while_else_with_return() -> i32:
    """Return exits function, else doesn't execute"""
    sum: i32 = 0
    i: i32 = 0
    while i < 5:
        sum = sum + i
        if i == 3:
            return sum  # Early return
        i = i + 1
    else:
        sum = sum + 100  # Should NOT execute
    return sum


def main():
    """Run all tests"""
    print("Testing for-else statements...")
    print()
    
    # Test normal completion
    result = test_for_else_normal()
    print(f"test_for_else_normal: {'PASS' if result == 106 else 'FAIL'} (result={result}, expected=106)")
    assert result == 106
    
    # Test with break
    result = test_for_else_with_break()
    print(f"test_for_else_with_break: {'PASS' if result == 6 else 'FAIL'} (result={result}, expected=6)")
    assert result == 6
    
    # Test empty iterator
    result = test_for_else_empty()
    print(f"test_for_else_empty: {'PASS' if result == 100 else 'FAIL'} (result={result}, expected=100)")
    assert result == 100
    
    # Test conditional break - found
    result = test_for_else_conditional_break(30)
    print(f"test_for_else_conditional_break(30): {'PASS' if result == 1 else 'FAIL'} (result={result}, expected=1)")
    assert result == 1
    
    # Test conditional break - not found
    result = test_for_else_conditional_break(99)
    print(f"test_for_else_conditional_break(99): {'PASS' if result == -1 else 'FAIL'} (result={result}, expected=-1)")
    assert result == -1
    
    # Test nested for-else
    result = test_nested_for_else()
    print(f"test_nested_for_else: {'PASS' if result == 10260 else 'FAIL'} (result={result}, expected=10260)")
    assert result == 10260
    
    # Test with continue
    result = test_for_else_with_continue()
    print(f"test_for_else_with_continue: {'PASS' if result == 112 else 'FAIL'} (result={result}, expected=112)")
    assert result == 112
    
    # Test with return
    result = test_for_else_with_return()
    print(f"test_for_else_with_return: {'PASS' if result == 6 else 'FAIL'} (result={result}, expected=6)")
    assert result == 6
    
    # Test with yield
    result = test_for_else_with_yield()
    print(f"test_for_else_with_yield: {'PASS' if result == 110 else 'FAIL'} (result={result}, expected=110)")
    assert result == 110
    
    result = test_for_else_yield_with_break()
    print(f"test_for_else_yield_with_break: {'PASS' if result == 10 else 'FAIL'} (result={result}, expected=10)")
    assert result == 10
    
    result = test_for_else_empty_yield()
    print(f"test_for_else_empty_yield: {'PASS' if result == 100 else 'FAIL'} (result={result}, expected=100)")
    assert result == 100

    # Test while-else: normal completion
    result = test_while_else_normal(5)
    print(f"test_while_else_normal(5): {'PASS' if result == 110 else 'FAIL'} (result={result}, expected=110)")
    assert result == 110

    # Test while-else: normal completion with n=0 (body never runs)
    result = test_while_else_normal(0)
    print(f"test_while_else_normal(0): {'PASS' if result == 100 else 'FAIL'} (result={result}, expected=100)")
    assert result == 100

    # Test while-else: break - found
    result = test_while_else_with_break(4)
    print(f"test_while_else_with_break(4): {'PASS' if result == 1 else 'FAIL'} (result={result}, expected=1)")
    assert result == 1

    # Test while-else: no break - not found
    result = test_while_else_with_break(99)
    print(f"test_while_else_with_break(99): {'PASS' if result == -1 else 'FAIL'} (result={result}, expected=-1)")
    assert result == -1

    # Test while-else: condition false on entry
    result = test_while_else_empty()
    print(f"test_while_else_empty: {'PASS' if result == 100 else 'FAIL'} (result={result}, expected=100)")
    assert result == 100

    # Test while-else: continue
    result = test_while_else_with_continue()
    print(f"test_while_else_with_continue: {'PASS' if result == 112 else 'FAIL'} (result={result}, expected=112)")
    assert result == 112

    # Test while-else: early return
    result = test_while_else_with_return()
    print(f"test_while_else_with_return: {'PASS' if result == 6 else 'FAIL'} (result={result}, expected=6)")
    assert result == 6

    print()
    print("All for-else tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
