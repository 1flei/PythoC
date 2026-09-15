#!/usr/bin/env python3
"""
Test break/continue behavior in for loops over yield functions

This tests the correct handling of break and continue statements
inside the body of a for loop that iterates over a yield function.

Key semantics:
- break: exit the for loop, skip else clause
- continue: skip rest of current iteration, proceed to next yield
- else: executes only if no break occurred
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import unittest
from pythoc import compile, i32
from pythoc.libc.stdio import printf


# =============================================================================
# Yield functions for testing
# =============================================================================

@compile
def gen_0_to_n(n: i32) -> i32:
    """Generate sequence 0, 1, 2, ..., n-1"""
    i: i32 = 0
    while i < n:
        yield i
        i = i + 1


@compile
def gen_multi_yield(n: i32) -> i32:
    """Generate multiple yields per call: 0, 1, 2, ..., n-1 each doubled"""
    i: i32 = 0
    while i < n:
        yield i
        yield i * 10
        i = i + 1


@compile
def gen_conditional_yield(n: i32) -> i32:
    """Conditional yields based on value"""
    i: i32 = 0
    while i < n:
        if i % 2 == 0:
            yield i
        else:
            yield i * 100
        i = i + 1


# =============================================================================
# Generators with control flow inside their OWN body (break/continue)
# =============================================================================

@compile(suffix="gen_body_flow")
def gen_body_break(n: i32) -> i32:
    """Generator whose own loop uses break to end iteration"""
    i: i32 = 0
    while True:
        if i >= n:
            break
        yield i
        i = i + 1


@compile(suffix="gen_body_flow")
def gen_body_continue(n: i32) -> i32:
    """Generator whose own loop uses continue to skip yields"""
    i: i32 = 0
    while i < n:
        i = i + 1
        if i % 2 == 0:
            continue
        yield i


@compile(suffix="gen_body_flow")
def gen_body_break_continue(n: i32) -> i32:
    """Generator whose own loop combines continue and break"""
    i: i32 = 0
    while i < n:
        i = i + 1
        if i % 2 == 0:
            continue
        if i > 7:
            break
        yield i


# =============================================================================
# Generators with early return inside their OWN body
# =============================================================================

@compile(suffix="gen_body_return")
def gen_body_return(n: i32) -> i32:
    """Bare return inside the generator's own loop ends iteration."""
    i: i32 = 0
    while i < n:
        if i == 2:
            return
        yield i
        i = i + 1


@compile(suffix="gen_body_return")
def gen_body_return_value(n: i32) -> i32:
    """Return with a value: evaluated for side effects, then discarded;
    a for loop never observes a generator's return value."""
    i: i32 = 0
    while i < n:
        if i == 2:
            return i * 100
        yield i
        i = i + 1


@compile(suffix="gen_body_return")
def gen_body_return_nested(n: i32) -> i32:
    """Return from a nested loop inside the generator body."""
    i: i32 = 0
    while i < n:
        j: i32 = 0
        while j < 3:
            if j == 1 and i == 1:
                return
            j = j + 1
        yield i
        i = i + 1


@compile(suffix="gen_body_return")
def gen_body_return_toplevel(flag: i32) -> i32:
    """Top-level return guard (before any loop)."""
    if flag == 0:
        return
    yield 7
    yield 8


@compile(suffix="gen_body_return")
def drive_body_return() -> i32:
    total: i32 = 0
    for v in gen_body_return(5):
        total = total + v
    return total  # 0 + 1 = 1


@compile(suffix="gen_body_return")
def drive_body_return_value() -> i32:
    total: i32 = 0
    for v in gen_body_return_value(5):
        total = total + v
    return total  # 0 + 1 = 1


@compile(suffix="gen_body_return")
def drive_body_return_nested() -> i32:
    total: i32 = 0
    for v in gen_body_return_nested(5):
        total = total + v
    return total  # only i=0 yields -> 0


@compile(suffix="gen_body_return")
def drive_body_return_toplevel_off() -> i32:
    total: i32 = 0
    for v in gen_body_return_toplevel(0):
        total = total + v
    return total  # 0


@compile(suffix="gen_body_return")
def drive_body_return_toplevel_on() -> i32:
    total: i32 = 0
    for v in gen_body_return_toplevel(1):
        total = total + v
    return total  # 7 + 8 = 15


@compile(suffix="gen_body_return")
def drive_body_return_else() -> i32:
    """for-else: the else clause runs when the generator returns (normal
    exhaustion), it does not count as a break."""
    total: i32 = 0
    for v in gen_body_return_value(5):
        total = total + v
    else:
        total = total + 1000
    return total  # 1 + 1000 = 1001


# =============================================================================
# Test: break in yield loop
# =============================================================================

@compile
def test_yield_break_simple() -> i32:
    """Break exits the loop immediately"""
    total: i32 = 0
    for x in gen_0_to_n(10):
        if x == 5:
            break
        total = total + x
    # 0 + 1 + 2 + 3 + 4 = 10
    return total


@compile
def test_yield_break_else_skipped() -> i32:
    """Break should skip else clause"""
    total: i32 = 0
    for x in gen_0_to_n(10):
        if x == 5:
            break
        total = total + x
    else:
        total = total + 1000  # Should NOT execute
    # 0 + 1 + 2 + 3 + 4 = 10 (no 1000)
    return total


@compile
def test_yield_no_break_else_executes() -> i32:
    """Without break, else should execute"""
    total: i32 = 0
    for x in gen_0_to_n(5):
        total = total + x
    else:
        total = total + 1000  # Should execute
    # 0 + 1 + 2 + 3 + 4 + 1000 = 1010
    return total


# =============================================================================
# Test: continue in yield loop
# =============================================================================

@compile
def test_yield_continue_simple() -> i32:
    """Continue skips rest of current iteration"""
    total: i32 = 0
    for x in gen_0_to_n(10):
        if x == 5:
            continue
        total = total + x
    # 0+1+2+3+4+6+7+8+9 = 45 - 5 = 40
    return total


@compile
def test_yield_continue_else_executes() -> i32:
    """Continue should NOT prevent else execution"""
    total: i32 = 0
    for x in gen_0_to_n(5):
        if x == 2:
            continue
        total = total + x
    else:
        total = total + 1000  # Should execute
    # 0+1+3+4+1000 = 1008
    return total


@compile
def test_yield_continue_multiple() -> i32:
    """Multiple continues in loop"""
    total: i32 = 0
    for x in gen_0_to_n(10):
        if x % 2 == 0:
            continue  # Skip even numbers
        total = total + x
    # 1+3+5+7+9 = 25
    return total


# =============================================================================
# Test: break and continue combined
# =============================================================================

@compile
def test_yield_break_and_continue() -> i32:
    """Both break and continue in same loop"""
    total: i32 = 0
    for x in gen_0_to_n(10):
        if x % 2 == 0:
            continue  # Skip even
        if x >= 7:
            break  # Stop at 7
        total = total + x
    # 1+3+5 = 9
    return total


@compile
def test_yield_break_and_continue_else() -> i32:
    """Break and continue with else clause"""
    total: i32 = 0
    for x in gen_0_to_n(10):
        if x % 2 == 0:
            continue
        if x >= 7:
            break
        total = total + x
    else:
        total = total + 1000  # Should NOT execute (break happened)
    # 1+3+5 = 9
    return total


# =============================================================================
# Test: break/continue with multi-yield functions
# =============================================================================

@compile
def test_multi_yield_break() -> i32:
    """Break in multi-yield function"""
    total: i32 = 0
    for x in gen_multi_yield(5):
        if x >= 20:
            break
        total = total + x
    # 0, 0, 1, 10, 2, 20 -> break at 20
    # 0 + 0 + 1 + 10 + 2 = 13
    return total


@compile
def test_multi_yield_continue() -> i32:
    """Continue in multi-yield function"""
    total: i32 = 0
    for x in gen_multi_yield(3):
        if x >= 10:
            continue  # Skip values >= 10
        total = total + x
    # 0, 0, 1, 10(skip), 2, 20(skip)
    # 0 + 0 + 1 + 2 = 3
    return total


# =============================================================================
# Test: break/continue with conditional yield
# =============================================================================

@compile
def test_conditional_yield_break() -> i32:
    """Break with conditional yield function"""
    total: i32 = 0
    for x in gen_conditional_yield(6):
        if x >= 300:
            break
        total = total + x
    # i=0: 0, i=1: 100, i=2: 2, i=3: 300 -> break
    # 0 + 100 + 2 = 102
    return total


@compile
def test_conditional_yield_continue() -> i32:
    """Continue with conditional yield function"""
    total: i32 = 0
    for x in gen_conditional_yield(6):
        if x >= 100:
            continue  # Skip large values
        total = total + x
    # i=0: 0, i=1: 100(skip), i=2: 2, i=3: 300(skip), i=4: 4, i=5: 500(skip)
    # 0 + 2 + 4 = 6
    return total


# =============================================================================
# Test: nested loops with break/continue
# =============================================================================

@compile
def test_nested_yield_break_inner() -> i32:
    """Break in inner yield loop only"""
    total: i32 = 0
    for i in gen_0_to_n(3):
        for j in gen_0_to_n(5):
            if j == 2:
                break  # Break inner loop only
            total = total + j
        total = total + 100  # After inner loop
    # i=0: j=0,1 + 100 = 101
    # i=1: j=0,1 + 100 = 101
    # i=2: j=0,1 + 100 = 101
    # Total = 303
    return total


@compile
def test_nested_yield_continue_inner() -> i32:
    """Continue in inner yield loop"""
    total: i32 = 0
    for i in gen_0_to_n(3):
        for j in gen_0_to_n(5):
            if j == 2:
                continue  # Skip j=2
            total = total + j
    # Each inner: 0+1+3+4 = 8, outer runs 3 times
    # Total = 8 * 3 = 24
    return total


# =============================================================================
# Test: edge cases
# =============================================================================

@compile
def test_yield_break_first_iteration() -> i32:
    """Break on first iteration"""
    total: i32 = 0
    for x in gen_0_to_n(10):
        break
        total = total + x  # Never reached
    else:
        total = total + 1000  # Should NOT execute
    return total  # 0


@compile
def test_yield_continue_all() -> i32:
    """Continue on every iteration"""
    total: i32 = 0
    for x in gen_0_to_n(5):
        continue
        total = total + x  # Never reached
    else:
        total = total + 1000  # Should execute
    return total  # 1000


@compile
def test_yield_break_in_else_branch() -> i32:
    """Break inside if-else branch"""
    total: i32 = 0
    for x in gen_0_to_n(10):
        if x < 5:
            total = total + x
        else:
            break
    # 0+1+2+3+4 = 10
    return total


# =============================================================================
# Test: break/continue inside the generator's OWN body
# =============================================================================

@compile(suffix="gen_body_flow")
def test_gen_body_break() -> i32:
    """Iterate a generator whose own loop breaks"""
    total: i32 = 0
    for x in gen_body_break(5):
        total = total + x
    # 0+1+2+3+4 = 10
    return total


@compile(suffix="gen_body_flow")
def test_gen_body_continue() -> i32:
    """Iterate a generator whose own loop continues (skips even i)"""
    total: i32 = 0
    for x in gen_body_continue(6):
        total = total + x
    # i=1..6, yields only odd i: 1+3+5 = 9
    return total


@compile(suffix="gen_body_flow")
def test_gen_body_break_continue() -> i32:
    """Iterate a generator whose own loop combines continue and break"""
    total: i32 = 0
    for x in gen_body_break_continue(20):
        total = total + x
    # yields 1,3,5,7 then breaks: 1+3+5+7 = 16
    return total


# =============================================================================
# Main test runner
# =============================================================================

@compile
def main() -> i32:
    printf("=== Yield Break/Continue Tests ===\n\n")
    
    result: i32
    
    # Break tests
    result = test_yield_break_simple()
    printf("test_yield_break_simple: %d (expected 10)\n", result)
    if result != 10:
        return 1
    
    result = test_yield_break_else_skipped()
    printf("test_yield_break_else_skipped: %d (expected 10)\n", result)
    if result != 10:
        return 1
    
    result = test_yield_no_break_else_executes()
    printf("test_yield_no_break_else_executes: %d (expected 1010)\n", result)
    if result != 1010:
        return 1
    
    # Continue tests
    result = test_yield_continue_simple()
    printf("test_yield_continue_simple: %d (expected 40)\n", result)
    if result != 40:
        return 1
    
    result = test_yield_continue_else_executes()
    printf("test_yield_continue_else_executes: %d (expected 1008)\n", result)
    if result != 1008:
        return 1
    
    result = test_yield_continue_multiple()
    printf("test_yield_continue_multiple: %d (expected 25)\n", result)
    if result != 25:
        return 1
    
    # Break and continue combined
    result = test_yield_break_and_continue()
    printf("test_yield_break_and_continue: %d (expected 9)\n", result)
    if result != 9:
        return 1
    
    result = test_yield_break_and_continue_else()
    printf("test_yield_break_and_continue_else: %d (expected 9)\n", result)
    if result != 9:
        return 1
    
    # Multi-yield tests
    result = test_multi_yield_break()
    printf("test_multi_yield_break: %d (expected 13)\n", result)
    if result != 13:
        return 1
    
    result = test_multi_yield_continue()
    printf("test_multi_yield_continue: %d (expected 3)\n", result)
    if result != 3:
        return 1
    
    # Conditional yield tests
    result = test_conditional_yield_break()
    printf("test_conditional_yield_break: %d (expected 102)\n", result)
    if result != 102:
        return 1
    
    result = test_conditional_yield_continue()
    printf("test_conditional_yield_continue: %d (expected 6)\n", result)
    if result != 6:
        return 1
    
    # Nested loop tests
    result = test_nested_yield_break_inner()
    printf("test_nested_yield_break_inner: %d (expected 303)\n", result)
    if result != 303:
        return 1
    
    result = test_nested_yield_continue_inner()
    printf("test_nested_yield_continue_inner: %d (expected 24)\n", result)
    if result != 24:
        return 1
    
    # Edge cases
    result = test_yield_break_first_iteration()
    printf("test_yield_break_first_iteration: %d (expected 0)\n", result)
    if result != 0:
        return 1
    
    result = test_yield_continue_all()
    printf("test_yield_continue_all: %d (expected 1000)\n", result)
    if result != 1000:
        return 1
    
    result = test_yield_break_in_else_branch()
    printf("test_yield_break_in_else_branch: %d (expected 10)\n", result)
    if result != 10:
        return 1

    # Generator-body control flow
    result = test_gen_body_break()
    printf("test_gen_body_break: %d (expected 10)\n", result)
    if result != 10:
        return 1

    result = test_gen_body_continue()
    printf("test_gen_body_continue: %d (expected 9)\n", result)
    if result != 9:
        return 1

    result = test_gen_body_break_continue()
    printf("test_gen_body_break_continue: %d (expected 16)\n", result)
    if result != 16:
        return 1

    # Generator-body early return
    result = drive_body_return()
    printf("drive_body_return: %d (expected 1)\n", result)
    if result != 1:
        return 1

    result = drive_body_return_value()
    printf("drive_body_return_value: %d (expected 1)\n", result)
    if result != 1:
        return 1

    result = drive_body_return_nested()
    printf("drive_body_return_nested: %d (expected 0)\n", result)
    if result != 0:
        return 1

    result = drive_body_return_toplevel_off()
    printf("drive_body_return_toplevel_off: %d (expected 0)\n", result)
    if result != 0:
        return 1

    result = drive_body_return_toplevel_on()
    printf("drive_body_return_toplevel_on: %d (expected 15)\n", result)
    if result != 15:
        return 1

    result = drive_body_return_else()
    printf("drive_body_return_else: %d (expected 1001)\n", result)
    if result != 1001:
        return 1
    
    printf("\n=== All Tests Passed ===\n")
    return 0


class TestYieldBreakContinue(unittest.TestCase):
    """Test break/continue in yield loops"""
    
    def test_all(self):
        """Run main test"""
        result = main()
        self.assertEqual(result, 0)


if __name__ == '__main__':
    unittest.main()
