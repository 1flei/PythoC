#!/usr/bin/env python3
"""
Test scoped goto/label control flow primitives.

API:
    with label("name"):  - Define a scoped label
        goto("name")     - Jump to beginning of label scope
        goto_end("name") - Jump to end of label scope

Key properties:
1. Labels define scopes (via with statement)
2. Visibility rules:
   - goto: Can target self, ancestors, siblings, uncles
   - goto_end: Can ONLY target self and ancestors (must be inside target)
3. Defer execution follows parent_scope_depth model:
   - Both goto and goto_end exit to target's parent depth
   - Execute defers for all scopes being exited

Test categories:
1. Basic label/goto operations
2. Nested labels and scope hierarchy
3. Sibling and uncle jumps
4. Defer integration
5. Loop patterns (break/continue simulation)
6. State machine patterns
7. Error handling patterns
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc import compile, i32, ptr, void
from pythoc.builtin_entities import label, goto, goto_end, defer
from pythoc.libc.stdio import printf


# =============================================================================
# Helper functions for defer tests
# =============================================================================

@compile
def inc(p: ptr[i32]) -> void:
    """Increment value at pointer"""
    p[0] = p[0] + 1


@compile
def add_n(p: ptr[i32], n: i32) -> void:
    """Add n to value at pointer"""
    p[0] = p[0] + n


# =============================================================================
# 1. Basic label/goto operations
# =============================================================================

@compile
def test_label_fallthrough() -> i32:
    """Label with fall-through (no goto)"""
    result: i32 = 0
    with label("main"):
        result = 1
    return result  # Expected: 1


@compile
def test_goto_end_simple() -> i32:
    """Simple goto_end to skip code"""
    result: i32 = 0
    with label("main"):
        result = 1
        goto_end("main")
        result = 100  # Skipped
    return result  # Expected: 1


@compile
def test_goto_backward() -> i32:
    """Backward goto (loop simulation)"""
    i: i32 = 0
    with label("loop"):
        if i >= 5:
            goto_end("loop")
        i = i + 1
        goto("loop")
    return i  # Expected: 5


@compile
def test_goto_forward() -> i32:
    """Forward goto to sibling label"""
    result: i32 = 0
    with label("first"):
        result = 1
        goto("second")
        result = 100  # Skipped
    with label("second"):
        result = result + 10
    return result  # Expected: 11


@compile
def test_multiple_labels_fallthrough() -> i32:
    """Multiple labels with fallthrough"""
    x: i32 = 1
    with label("point1"):
        x = x + 1
    with label("point2"):
        x = x + 1
    with label("point3"):
        x = x + 1
    return x  # Expected: 4


# =============================================================================
# 2. Nested labels and scope hierarchy
# =============================================================================

@compile
def test_nested_labels() -> i32:
    """Nested labels with goto_end to inner"""
    result: i32 = 0
    with label("outer"):
        result = result + 1
        with label("inner"):
            result = result + 10
            goto_end("inner")
            result = result + 1000  # Skipped
        result = result + 100
    return result  # Expected: 1 + 10 + 100 = 111


@compile
def test_goto_end_to_outer() -> i32:
    """goto_end to outer label from inner scope"""
    result: i32 = 0
    with label("outer"):
        result = result + 1
        with label("inner"):
            result = result + 10
            goto_end("outer")  # Exit both scopes
            result = result + 1000  # Skipped
        result = result + 100  # Skipped
    return result  # Expected: 1 + 10 = 11


@compile
def test_goto_to_self() -> i32:
    """goto to self (loop within same label)"""
    i: i32 = 0
    sum: i32 = 0
    with label("loop"):
        sum = sum + i
        i = i + 1
        if i <= 10:
            goto("loop")
    return sum  # Expected: 0+1+2+...+10 = 55


@compile
def test_deeply_nested_goto_end() -> i32:
    """goto_end from deeply nested scope"""
    result: i32 = 0
    with label("L1"):
        result = result + 1
        with label("L2"):
            result = result + 10
            with label("L3"):
                result = result + 100
                goto_end("L1")  # Exit all three
                result = result + 1000  # Skipped
            result = result + 10000  # Skipped
        result = result + 100000  # Skipped
    return result  # Expected: 1 + 10 + 100 = 111


# =============================================================================
# 3. Sibling and uncle jumps
# =============================================================================

@compile
def test_sibling_goto() -> i32:
    """Direct sibling goto jump"""
    result: i32 = 0
    with label("A"):
        result = 1
        goto("B")
        result = 100  # Skipped
    with label("B"):
        result = result + 10
    return result  # Expected: 11


@compile
def test_uncle_goto() -> i32:
    """Jump to uncle label (ancestor's sibling)"""
    result: i32 = 0
    with label("A"):
        with label("A1"):
            result = 1
            goto("B")  # Jump to uncle
            result = 100  # Skipped
    with label("B"):
        result = result + 10
    return result  # Expected: 11


@compile
def test_alternating_labels() -> i32:
    """Alternating between two sibling labels"""
    count: i32 = 0
    result: i32 = 0
    
    with label("ping"):
        result = result + 1
        count = count + 1
        if count < 10:
            goto("pong")
        goto("done")
    
    with label("pong"):
        result = result + 2
        count = count + 1
        if count < 10:
            goto("ping")
        goto("done")
    
    with label("done"):
        pass
    
    # ping adds 1, pong adds 2, alternating
    # count: 1(ping) 2(pong) 3(ping) 4(pong) 5(ping) 6(pong) 7(ping) 8(pong) 9(ping) 10(pong)
    # result: 1 + 2 + 1 + 2 + 1 + 2 + 1 + 2 + 1 + 2 = 15
    return result


@compile
def test_multi_phase_control() -> i32:
    """Multi-phase control flow with nested and sibling jumps"""
    result: i32 = 0
    error: i32 = 0
    
    with label("main"):
        with label("phase1"):
            result = result + 1
            if error != 0:
                goto("error_handler")
            goto("phase2")
        with label("phase2"):
            result = result + 10
            if error != 0:
                goto("error_handler")
            goto("done")
    
    with label("error_handler"):
        result = result + 100
    
    with label("done"):
        pass
    
    return result  # Expected: 11 (1 + 10, no error, skip error_handler)


# =============================================================================
# 4. Defer integration
# =============================================================================

@compile
def test_defer_on_goto_end() -> i32:
    """Defer executes on goto_end"""
    result: i32 = 0
    with label("main"):
        defer(inc, ptr(result))
        goto_end("main")
        result = 100  # Skipped
    return result  # Expected: 1 (defer executed)


@compile
def test_defer_on_goto() -> i32:
    """Defer executes on each goto (exits scope, then re-enters)
    
    goto exits to target's parent depth, executing defers along the way,
    then re-enters the scope (which re-registers defers).
    """
    result: i32 = 0
    count: i32 = 0
    with label("loop"):
        defer(inc, ptr(result))  # Re-registered each time we enter
        count = count + 1
        if count < 3:
            goto("loop")  # Executes defer, then re-enters
    return result  # Expected: 3 (defer executed 3 times: 2 goto + 1 normal exit)


@compile
def test_defer_nested_goto_end_outer() -> i32:
    """Nested defers with goto_end to outer"""
    result: i32 = 0
    with label("outer"):
        defer(add_n, ptr(result), i32(100))
        with label("inner"):
            defer(add_n, ptr(result), i32(10))
            result = result + 1
            goto_end("outer")  # Execute both defers
            result = result + 1000  # Skipped
        result = result + 10000  # Skipped
    return result  # Expected: 1 + 10 + 100 = 111


@compile
def test_defer_nested_goto_end_inner() -> i32:
    """Nested defers with goto_end to inner only"""
    result: i32 = 0
    with label("outer"):
        defer(add_n, ptr(result), i32(100))
        with label("inner"):
            defer(add_n, ptr(result), i32(10))
            result = result + 1
            goto_end("inner")  # Execute only inner defer
            result = result + 1000  # Skipped
        result = result + 10000
    return result  # Expected: 1 + 10 + 10000 + 100 = 10111


@compile
def test_defer_sibling_goto() -> i32:
    """Sibling goto executes defers correctly"""
    result: i32 = 0
    with label("A"):
        defer(add_n, ptr(result), i32(1))
        goto("B")  # Executes A's defer
    with label("B"):
        defer(add_n, ptr(result), i32(10))
    return result  # Expected: 11 (1 from A's defer + 10 from B's defer at exit)


@compile
def test_defer_loop_with_goto() -> i32:
    """Defer in loop using goto"""
    result: i32 = 0
    i: i32 = 0
    with label("loop"):
        if i >= 3:
            goto_end("loop")
        defer(inc, ptr(result))  # Re-registered each iteration
        i = i + 1
        goto("loop")  # Executes defer, then re-enters
    return result  # Expected: 3 (defer executed 3 times)


# =============================================================================
# 5. Loop patterns (break/continue simulation)
# =============================================================================

@compile
def test_break_simulation() -> i32:
    """Simulate break with goto_end"""
    sum: i32 = 0
    i: i32 = 0
    with label("loop"):
        if i >= 10:
            goto_end("loop")
        if i == 5:
            goto_end("loop")  # Break at i=5
        sum = sum + i
        i = i + 1
        goto("loop")
    return sum  # Expected: 0+1+2+3+4 = 10


@compile
def test_continue_simulation() -> i32:
    """Simulate continue with goto"""
    sum: i32 = 0
    i: i32 = 0
    with label("loop"):
        if i >= 10:
            goto_end("loop")
        i = i + 1
        if i == 5:
            goto("loop")  # Skip adding 5
        sum = sum + i
        goto("loop")
    return sum  # Expected: 1+2+3+4+6+7+8+9+10 = 50


@compile
def test_double_loop_exit() -> i32:
    """Exit from double nested loop with goto_end"""
    result: i32 = 0
    i: i32 = 0
    j: i32
    
    with label("outer"):
        if i >= 5:
            goto_end("outer")
        j = 0
        with label("inner"):
            if j >= 5:
                i = i + 1
                goto("outer")
            result = result + 1
            if result >= 10:
                goto_end("outer")  # Exit both loops
            j = j + 1
            goto("inner")
    
    return result  # Expected: 10


@compile
def test_goto_with_while() -> i32:
    """Goto combined with while loop"""
    x: i32 = 0
    sum: i32 = 0
    
    while x < 5:
        x = x + 1
        with label("skip_check"):
            if x == 3:
                goto_end("skip_check")
            sum = sum + x
    
    # sum = 1 + 2 + 4 + 5 = 12 (skip 3)
    return sum


@compile
def test_goto_with_for_range() -> i32:
    """Goto combined with for-range loop"""
    sum: i32 = 0
    
    with label("loop_region"):
        for i in range(10):
            if i == 7:
                goto_end("loop_region")
            sum = sum + i
    
    # sum = 0+1+2+3+4+5+6 = 21
    return sum


# =============================================================================
# 6. State machine patterns
# =============================================================================

@compile
def test_simple_state_machine() -> i32:
    """Simple state machine with sibling jumps"""
    count: i32 = 0
    with label("state_A"):
        count = count + 1
        if count >= 3:
            goto("done")
        goto("state_B")
    with label("state_B"):
        count = count + 1
        goto("state_A")
    with label("done"):
        pass
    return count  # Expected: 3 (A->B->A->B->A->done)


@compile
def test_state_machine_with_data() -> i32:
    """State machine that accumulates data"""
    state: i32 = 0
    result: i32 = 0
    counter: i32 = 0
    
    with label("state_0"):
        if state != 0:
            goto_end("state_0")
        result = result + 1
        counter = counter + 1
        if counter < 3:
            goto("state_0")
        state = 1
    
    with label("state_1"):
        if state != 1:
            goto_end("state_1")
        result = result + 10
        state = 2
    
    with label("state_2"):
        if state != 2:
            goto_end("state_2")
        result = result + 100
    
    return result  # Expected: 1+1+1 + 10 + 100 = 113


@compile
def test_retry_pattern() -> i32:
    """Retry pattern with goto"""
    result: i32 = 0
    retry_count: i32 = 0
    
    with label("try_operation"):
        retry_count = retry_count + 1
        if retry_count < 3:
            goto("try_operation")  # Retry (self)
        goto("success")
    
    with label("success"):
        result = retry_count * 10
    
    return result  # Expected: 30 (3 retries)


# =============================================================================
# 7. Error handling patterns
# =============================================================================

@compile
def test_early_exit_with_cleanup() -> i32:
    """Early exit with defer cleanup"""
    result: i32 = 0
    with label("main"):
        defer(add_n, ptr(result), i32(1))
        result = result + 10
        if result > 5:
            goto_end("main")  # Early exit, defer still runs
        result = result + 100  # Skipped
    return result  # Expected: 10 + 1 = 11


@compile
def test_error_handling_pattern() -> i32:
    """Error handling with cleanup"""
    step: i32 = 0
    error_code: i32 = 0
    cleanup_count: i32 = 0
    
    with label("function"):
        defer(inc, ptr(cleanup_count))
        
        # Step 1
        step = 1
        if step != 1:
            error_code = 1
            goto_end("function")
        
        # Step 2 - simulate error
        step = 2
        error_code = 2
        goto_end("function")
        
        # Step 3 - not reached
        step = 3
    
    # cleanup_count should be 1 (defer ran)
    return error_code * 10 + cleanup_count  # Expected: 20 + 1 = 21


@compile
def test_diamond_pattern() -> i32:
    """Diamond control flow pattern"""
    x: i32 = 5
    result: i32 = 0
    
    with label("main"):
        if x > 3:
            result = result + 10
            goto_end("main")
        result = result + 20
    
    result = result + 1
    return result  # Expected: 10 + 1 = 11


@compile
def test_goto_in_nested_if() -> i32:
    """Goto from nested if statements"""
    x: i32 = 5
    y: i32 = 10
    result: i32 = 0
    
    with label("main"):
        if x > 0:
            if y > 5:
                result = result + 1
                goto("middle")
            result = result + 100  # Skipped
        result = result + 1000  # Skipped
        
        with label("middle"):
            if x < 10:
                result = result + 10
                goto_end("main")
            result = result + 10000  # Skipped
    
    return result  # Expected: 1 + 10 = 11


@compile
def test_if_else_chain_with_goto() -> i32:
    """Goto with if-elif-else chain"""
    code: i32 = 2
    result: i32 = 0
    
    with label("main"):
        if code == 1:
            result = 10
            goto_end("main")
        elif code == 2:
            result = 20
            goto_end("main")
        elif code == 3:
            result = 30
            goto_end("main")
        else:
            result = 40
    
    return result  # Expected: 20


# =============================================================================
# Main
# =============================================================================

@compile
def main() -> i32:
    printf("=== Scoped Goto/Label Tests ===\n\n")
    
    result: i32
    
    # 1. Basic label/goto operations
    printf("--- 1. Basic Operations ---\n")
    
    result = test_label_fallthrough()
    printf("test_label_fallthrough: %d (expected 1)\n", result)
    if result != 1:
        return 1
    
    result = test_goto_end_simple()
    printf("test_goto_end_simple: %d (expected 1)\n", result)
    if result != 1:
        return 1
    
    result = test_goto_backward()
    printf("test_goto_backward: %d (expected 5)\n", result)
    if result != 5:
        return 1
    
    result = test_goto_forward()
    printf("test_goto_forward: %d (expected 11)\n", result)
    if result != 11:
        return 1
    
    result = test_multiple_labels_fallthrough()
    printf("test_multiple_labels_fallthrough: %d (expected 4)\n", result)
    if result != 4:
        return 1
    
    # 2. Nested labels and scope hierarchy
    printf("\n--- 2. Nested Labels ---\n")
    
    result = test_nested_labels()
    printf("test_nested_labels: %d (expected 111)\n", result)
    if result != 111:
        return 1
    
    result = test_goto_end_to_outer()
    printf("test_goto_end_to_outer: %d (expected 11)\n", result)
    if result != 11:
        return 1
    
    result = test_goto_to_self()
    printf("test_goto_to_self: %d (expected 55)\n", result)
    if result != 55:
        return 1
    
    result = test_deeply_nested_goto_end()
    printf("test_deeply_nested_goto_end: %d (expected 111)\n", result)
    if result != 111:
        return 1
    
    # 3. Sibling and uncle jumps
    printf("\n--- 3. Sibling/Uncle Jumps ---\n")
    
    result = test_sibling_goto()
    printf("test_sibling_goto: %d (expected 11)\n", result)
    if result != 11:
        return 1
    
    result = test_uncle_goto()
    printf("test_uncle_goto: %d (expected 11)\n", result)
    if result != 11:
        return 1
    
    result = test_alternating_labels()
    printf("test_alternating_labels: %d (expected 15)\n", result)
    if result != 15:
        return 1
    
    result = test_multi_phase_control()
    printf("test_multi_phase_control: %d (expected 11)\n", result)
    if result != 11:
        return 1
    
    # 4. Defer integration
    printf("\n--- 4. Defer Integration ---\n")
    
    result = test_defer_on_goto_end()
    printf("test_defer_on_goto_end: %d (expected 1)\n", result)
    if result != 1:
        return 1
    
    result = test_defer_on_goto()
    printf("test_defer_on_goto: %d (expected 3)\n", result)
    if result != 3:
        return 1
    
    result = test_defer_nested_goto_end_outer()
    printf("test_defer_nested_goto_end_outer: %d (expected 111)\n", result)
    if result != 111:
        return 1
    
    result = test_defer_nested_goto_end_inner()
    printf("test_defer_nested_goto_end_inner: %d (expected 10111)\n", result)
    if result != 10111:
        return 1
    
    result = test_defer_sibling_goto()
    printf("test_defer_sibling_goto: %d (expected 11)\n", result)
    if result != 11:
        return 1
    
    result = test_defer_loop_with_goto()
    printf("test_defer_loop_with_goto: %d (expected 3)\n", result)
    if result != 3:
        return 1
    
    # 5. Loop patterns
    printf("\n--- 5. Loop Patterns ---\n")
    
    result = test_break_simulation()
    printf("test_break_simulation: %d (expected 10)\n", result)
    if result != 10:
        return 1
    
    result = test_continue_simulation()
    printf("test_continue_simulation: %d (expected 50)\n", result)
    if result != 50:
        return 1
    
    result = test_double_loop_exit()
    printf("test_double_loop_exit: %d (expected 10)\n", result)
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
    
    # 6. State machine patterns
    printf("\n--- 6. State Machine Patterns ---\n")
    
    result = test_simple_state_machine()
    printf("test_simple_state_machine: %d (expected 3)\n", result)
    if result != 3:
        return 1
    
    result = test_state_machine_with_data()
    printf("test_state_machine_with_data: %d (expected 113)\n", result)
    if result != 113:
        return 1
    
    result = test_retry_pattern()
    printf("test_retry_pattern: %d (expected 30)\n", result)
    if result != 30:
        return 1
    
    # 7. Error handling patterns
    printf("\n--- 7. Error Handling Patterns ---\n")
    
    result = test_early_exit_with_cleanup()
    printf("test_early_exit_with_cleanup: %d (expected 11)\n", result)
    if result != 11:
        return 1
    
    result = test_error_handling_pattern()
    printf("test_error_handling_pattern: %d (expected 21)\n", result)
    if result != 21:
        return 1
    
    result = test_diamond_pattern()
    printf("test_diamond_pattern: %d (expected 11)\n", result)
    if result != 11:
        return 1
    
    result = test_goto_in_nested_if()
    printf("test_goto_in_nested_if: %d (expected 11)\n", result)
    if result != 11:
        return 1
    
    result = test_if_else_chain_with_goto()
    printf("test_if_else_chain_with_goto: %d (expected 20)\n", result)
    if result != 20:
        return 1
    
    printf("\n=== All Scoped Goto Tests Passed ===\n")
    return 0


if __name__ == '__main__':
    result = main()
    print(f"main() returned: {result}")
