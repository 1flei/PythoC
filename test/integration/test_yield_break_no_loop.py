#!/usr/bin/env python3
"""
Test yield break/continue when yield function has NO internal loop.

This is the problematic case: when the yield function body doesn't have
a loop, the break/continue in the for-loop body has nowhere to go.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc import compile, i32
from pythoc.libc.stdio import printf


# =============================================================================
# Yield functions WITHOUT internal loops
# =============================================================================

@compile
def gen_three_values() -> i32:
    """Yield 3 values without a loop"""
    yield 1
    yield 2
    yield 3


@compile
def gen_conditional(n: i32) -> i32:
    """Conditional yield without loop"""
    if n > 0:
        yield n
    else:
        yield 0


@compile
def gen_multi_if(code: i32) -> i32:
    """Multiple if branches, each with yield"""
    if code == 1:
        yield 10
    elif code == 2:
        yield 20
    else:
        yield 30


# =============================================================================
# Tests with break
# =============================================================================

@compile
def test_three_values_break() -> i32:
    """Break in loop over non-loop yield function"""
    total: i32 = 0
    for x in gen_three_values():
        if x == 2:
            break
        total = total + x
    # Expected: 1 (only first value before break)
    return total


@compile
def test_three_values_no_break() -> i32:
    """No break - should get all values"""
    total: i32 = 0
    for x in gen_three_values():
        total = total + x
    # Expected: 1+2+3 = 6
    return total


@compile
def test_conditional_break() -> i32:
    """Break with conditional yield"""
    total: i32 = 0
    for x in gen_conditional(5):
        if x > 0:
            break
        total = total + x
    # Expected: 0 (break immediately)
    return total


# =============================================================================
# Tests with continue
# =============================================================================

@compile
def test_three_values_continue() -> i32:
    """Continue in loop over non-loop yield function"""
    total: i32 = 0
    for x in gen_three_values():
        if x == 2:
            continue
        total = total + x
    # Expected: 1+3 = 4 (skip 2)
    return total


# =============================================================================
# Tests with else clause
# =============================================================================

@compile
def test_three_values_else_no_break() -> i32:
    """Else should execute when no break"""
    total: i32 = 0
    for x in gen_three_values():
        total = total + x
    else:
        total = total + 100
    # Expected: 1+2+3+100 = 106
    return total


@compile
def test_three_values_else_with_break() -> i32:
    """Else should NOT execute when break"""
    total: i32 = 0
    for x in gen_three_values():
        if x == 2:
            break
        total = total + x
    else:
        total = total + 100  # Should NOT execute
    # Expected: 1 (no 100)
    return total


# =============================================================================
# Main
# =============================================================================

@compile
def main() -> i32:
    printf("=== Yield Break/Continue (No Internal Loop) ===\n\n")
    
    result: i32
    
    result = test_three_values_no_break()
    printf("test_three_values_no_break: %d (expected 6)\n", result)
    if result != 6:
        return 1
    
    result = test_three_values_break()
    printf("test_three_values_break: %d (expected 1)\n", result)
    if result != 1:
        return 1
    
    result = test_conditional_break()
    printf("test_conditional_break: %d (expected 0)\n", result)
    if result != 0:
        return 1
    
    result = test_three_values_continue()
    printf("test_three_values_continue: %d (expected 4)\n", result)
    if result != 4:
        return 1
    
    result = test_three_values_else_no_break()
    printf("test_three_values_else_no_break: %d (expected 106)\n", result)
    if result != 106:
        return 1
    
    result = test_three_values_else_with_break()
    printf("test_three_values_else_with_break: %d (expected 1)\n", result)
    if result != 1:
        return 1
    
    printf("\n=== All Tests Passed ===\n")
    return 0


if __name__ == '__main__':
    result = main()
    print(f"main() returned: {result}")
