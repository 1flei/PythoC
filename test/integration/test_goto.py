#!/usr/bin/env python3
"""
Test goto/label control flow primitives.

__label("name")  - Define a label
__goto("name")   - Jump to a label
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc import compile, i32, __label, __goto
from pythoc.libc.stdio import printf


# =============================================================================
# Basic goto tests
# =============================================================================

@compile
def test_simple_goto() -> i32:
    """Simple forward goto"""
    x: i32 = 1
    __goto("skip")
    x = 100  # This should be skipped
    __label("skip")
    return x  # Expected: 1


@compile
def test_backward_goto() -> i32:
    """Backward goto (loop simulation)"""
    x: i32 = 0
    __label("loop")
    x = x + 1
    if x < 5:
        __goto("loop")
    return x  # Expected: 5


@compile
def test_multiple_labels() -> i32:
    """Multiple labels and gotos"""
    result: i32 = 0
    
    __goto("first")
    result = result + 100  # Skipped
    
    __label("second")
    result = result + 20
    __goto("end")
    
    __label("first")
    result = result + 1
    __goto("second")
    
    __label("end")
    return result  # Expected: 1 + 20 = 21


@compile
def test_goto_in_if() -> i32:
    """Goto inside if statement"""
    x: i32 = 10
    
    if x > 5:
        __goto("big")
    
    return 0  # Not reached
    
    __label("big")
    return x  # Expected: 10


@compile
def test_computed_loop() -> i32:
    """Use goto to implement a loop with accumulator"""
    sum: i32 = 0
    i: i32 = 1
    
    __label("loop_start")
    sum = sum + i
    i = i + 1
    if i <= 10:
        __goto("loop_start")
    
    # sum = 1+2+3+...+10 = 55
    return sum


@compile
def test_early_exit() -> i32:
    """Goto for early exit from nested logic"""
    result: i32 = 0
    i: i32 = 0
    
    __label("outer")
    if i >= 3:
        __goto("done")
    
    result = result + i * 10
    i = i + 1
    __goto("outer")
    
    __label("done")
    # result = 0*10 + 1*10 + 2*10 = 30
    return result


@compile
def test_fallthrough_label() -> i32:
    """Label with fallthrough (no goto to it)"""
    x: i32 = 1
    __label("point1")
    x = x + 1
    __label("point2")
    x = x + 1
    __label("point3")
    x = x + 1
    return x  # Expected: 4


# =============================================================================
# Complex control flow with goto
# =============================================================================

@compile
def test_goto_nested_if() -> i32:
    """Goto from nested if statements"""
    x: i32 = 5
    y: i32 = 10
    result: i32 = 0
    
    if x > 0:
        if y > 5:
            result = result + 1
            __goto("middle")
        result = result + 100  # Skipped
    result = result + 1000  # Skipped
    
    __label("middle")
    if x < 10:
        result = result + 10
        __goto("end")
    result = result + 10000  # Skipped
    
    __label("end")
    return result  # Expected: 1 + 10 = 11


@compile
def test_goto_if_else_chain() -> i32:
    """Goto with if-elif-else chain"""
    code: i32 = 2
    result: i32 = 0
    
    if code == 1:
        result = 10
        __goto("done")
    elif code == 2:
        result = 20
        __goto("done")
    elif code == 3:
        result = 30
        __goto("done")
    else:
        result = 40
    
    __label("done")
    return result  # Expected: 20


@compile
def test_goto_state_machine() -> i32:
    """State machine implementation with goto"""
    state: i32 = 0
    counter: i32 = 0
    result: i32 = 0
    
    __label("state_0")
    if state == 0:
        result = result + 1
        counter = counter + 1
        if counter < 3:
            __goto("state_0")
        state = 1
        __goto("state_1")
    
    __label("state_1")
    if state == 1:
        result = result + 10
        state = 2
        __goto("state_2")
    
    __label("state_2")
    if state == 2:
        result = result + 100
    
    # result = 1+1+1 + 10 + 100 = 113
    return result


@compile
def test_goto_break_simulation() -> i32:
    """Simulate break with goto"""
    sum: i32 = 0
    i: i32 = 0
    
    __label("loop")
    if i >= 10:
        __goto("loop_end")
    
    if i == 5:
        __goto("loop_end")  # Simulate break at i=5
    
    sum = sum + i
    i = i + 1
    __goto("loop")
    
    __label("loop_end")
    # sum = 0+1+2+3+4 = 10
    return sum


@compile
def test_goto_continue_simulation() -> i32:
    """Simulate continue with goto"""
    sum: i32 = 0
    i: i32 = 0
    
    __label("loop")
    if i >= 10:
        __goto("loop_end")
    
    i = i + 1
    if i == 5:
        __goto("loop")  # Simulate continue, skip adding 5
    
    sum = sum + i
    __goto("loop")
    
    __label("loop_end")
    # sum = 1+2+3+4+6+7+8+9+10 = 50
    return sum


@compile
def test_goto_double_loop_exit() -> i32:
    """Exit from double nested loop with goto"""
    result: i32 = 0
    i: i32 = 0
    j: i32
    
    __label("outer_loop")
    if i >= 5:
        __goto("done")
    
    j = 0
    __label("inner_loop")
    if j >= 5:
        i = i + 1
        __goto("outer_loop")
    
    result = result + 1
    
    # Exit both loops when we've counted 10 times
    if result >= 10:
        __goto("done")
    
    j = j + 1
    __goto("inner_loop")
    
    __label("done")
    return result  # Expected: 10


@compile
def test_goto_with_while() -> i32:
    """Goto combined with while loop"""
    x: i32 = 0
    sum: i32 = 0
    
    while x < 5:
        x = x + 1
        if x == 3:
            __goto("skip_add")
        sum = sum + x
        __label("skip_add")
    
    # sum = 1 + 2 + 4 + 5 = 12 (skip 3)
    return sum


@compile
def test_goto_with_for_range() -> i32:
    """Goto combined with for-range loop"""
    sum: i32 = 0
    
    for i in range(10):
        if i == 7:
            __goto("early_exit")
        sum = sum + i
    
    __label("early_exit")
    # sum = 0+1+2+3+4+5+6 = 21
    return sum


@compile
def test_goto_alternating() -> i32:
    """Alternating between two labels"""
    count: i32 = 0
    result: i32 = 0
    
    __label("ping")
    result = result + 1
    count = count + 1
    if count < 10:
        __goto("pong")
    __goto("done")
    
    __label("pong")
    result = result + 2
    count = count + 1
    if count < 10:
        __goto("ping")
    __goto("done")
    
    __label("done")
    # ping adds 1, pong adds 2, alternating
    # count: 1(ping) 2(pong) 3(ping) 4(pong) 5(ping) 6(pong) 7(ping) 8(pong) 9(ping) 10(pong)
    # result: 1 + 2 + 1 + 2 + 1 + 2 + 1 + 2 + 1 + 2 = 15
    return result


@compile
def test_goto_diamond() -> i32:
    """Diamond control flow pattern with goto"""
    x: i32 = 5
    result: i32 = 0
    
    if x > 3:
        result = result + 10
        __goto("merge")
    else:
        result = result + 20
        __goto("merge")
    
    result = result + 1000  # Never reached
    
    __label("merge")
    result = result + 1
    return result  # Expected: 10 + 1 = 11


@compile
def test_goto_error_handling() -> i32:
    """Simulate error handling with goto"""
    step: i32 = 0
    error_code: i32 = 0
    
    # Step 1
    step = 1
    if step == 1:
        # Success, continue
        pass
    else:
        error_code = 1
        __goto("error")
    
    # Step 2
    step = 2
    if step == 2:
        # Simulate error at step 2
        error_code = 2
        __goto("error")
    
    # Step 3 (not reached)
    step = 3
    return 0  # Success
    
    __label("error")
    return error_code  # Expected: 2


# =============================================================================
# Main
# =============================================================================

@compile
def main() -> i32:
    printf("=== Goto/Label Tests ===\n\n")
    
    result: i32
    
    # Basic tests
    result = test_simple_goto()
    printf("test_simple_goto: %d (expected 1)\n", result)
    if result != 1:
        return 1
    
    result = test_backward_goto()
    printf("test_backward_goto: %d (expected 5)\n", result)
    if result != 5:
        return 1
    
    result = test_multiple_labels()
    printf("test_multiple_labels: %d (expected 21)\n", result)
    if result != 21:
        return 1
    
    result = test_goto_in_if()
    printf("test_goto_in_if: %d (expected 10)\n", result)
    if result != 10:
        return 1
    
    result = test_computed_loop()
    printf("test_computed_loop: %d (expected 55)\n", result)
    if result != 55:
        return 1
    
    result = test_early_exit()
    printf("test_early_exit: %d (expected 30)\n", result)
    if result != 30:
        return 1
    
    result = test_fallthrough_label()
    printf("test_fallthrough_label: %d (expected 4)\n", result)
    if result != 4:
        return 1
    
    # Complex control flow tests
    printf("\n--- Complex Control Flow ---\n")
    
    result = test_goto_nested_if()
    printf("test_goto_nested_if: %d (expected 11)\n", result)
    if result != 11:
        return 1
    
    result = test_goto_if_else_chain()
    printf("test_goto_if_else_chain: %d (expected 20)\n", result)
    if result != 20:
        return 1
    
    result = test_goto_state_machine()
    printf("test_goto_state_machine: %d (expected 113)\n", result)
    if result != 113:
        return 1
    
    result = test_goto_break_simulation()
    printf("test_goto_break_simulation: %d (expected 10)\n", result)
    if result != 10:
        return 1
    
    result = test_goto_continue_simulation()
    printf("test_goto_continue_simulation: %d (expected 50)\n", result)
    if result != 50:
        return 1
    
    result = test_goto_double_loop_exit()
    printf("test_goto_double_loop_exit: %d (expected 10)\n", result)
    if result != 10:
        return 1
    
    result = test_goto_with_while()
    printf("test_goto_with_while: %d (expected 12)\n", result)
    if result != 12:
        return 1
    
    result = test_goto_with_for_range()
    printf("test_goto_with_for_range: %d (expected 21)\n", result)
    if result != 21:
        return 1
    
    result = test_goto_alternating()
    printf("test_goto_alternating: %d (expected 15)\n", result)
    if result != 15:
        return 1
    
    result = test_goto_diamond()
    printf("test_goto_diamond: %d (expected 11)\n", result)
    if result != 11:
        return 1
    
    result = test_goto_error_handling()
    printf("test_goto_error_handling: %d (expected 2)\n", result)
    if result != 2:
        return 1
    
    printf("\n=== All Goto Tests Passed ===\n")
    return 0


if __name__ == '__main__':
    result = main()
    print(f"main() returned: {result}")
