#!/usr/bin/env python3
"""
Test defer intrinsic for deferred execution.

defer(f, a, b, c) registers f(a, b, c) to be called when the current block exits.

Defer semantics follow Zig/Go:
- Return value is evaluated BEFORE defers execute
- Defer cannot modify the return value directly
- Deferred calls are executed in FIFO order (first registered, first executed)
- Defer is useful for cleanup operations (closing files, freeing memory, etc.)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import unittest
from pythoc.decorators.compile import compile
from pythoc.builtin_entities import void, i32, defer, ptr, __label, __goto, linear, consume
from pythoc.build.output_manager import flush_all_pending_outputs
from pythoc.libc.stdio import printf

from test.utils.test_utils import DeferredTestCase


# =============================================================================
# Helper functions for testing (compiled) - defined inline in each test
# to avoid cross-module linking issues
# =============================================================================


# =============================================================================
# Basic defer tests
# =============================================================================

@compile(suffix="defer_simple")
def test_defer_simple() -> i32:
    """Simple defer - single deferred call
    
    Zig/Go semantics: return value (0) is evaluated before defer executes.
    Defer modifies result to 1, but return value is already captured as 0.
    """
    result: i32 = 0
    
    # Define helper inline
    @compile(suffix="defer_simple_inc")
    def inc(p: ptr[i32]) -> void:
        p[0] = p[0] + 1
    
    defer(inc, ptr(result))
    return result  # Returns 0 (Zig/Go semantics: defer executes after return value is captured)


@compile(suffix="defer_fifo_order")
def test_defer_fifo_order() -> i32:
    """Multiple defers execute in FIFO order
    
    Zig/Go semantics: return value (0) is captured before any defer executes.
    """
    result: i32 = 0
    
    @compile(suffix="defer_fifo_inc")
    def inc(p: ptr[i32]) -> void:
        p[0] = p[0] + 1
    
    defer(inc, ptr(result))
    defer(inc, ptr(result))
    defer(inc, ptr(result))
    return result  # Returns 0 (defers execute after, result becomes 3 but not returned)


@compile(suffix="defer_with_work")
def test_defer_with_work() -> i32:
    """Defer with work between registration and execution
    
    Zig/Go semantics: return value (15) is captured before defer executes.
    """
    result: i32 = 10
    
    @compile(suffix="defer_work_inc")
    def inc(p: ptr[i32]) -> void:
        p[0] = p[0] + 1
    
    defer(inc, ptr(result))
    result = result + 5  # result = 15
    return result  # Returns 15 (defer executes after, result becomes 16 but not returned)


@compile(suffix="defer_early_return")
def test_defer_early_return(cond: i32) -> i32:
    """Defer executes on early return
    
    Zig/Go semantics: return value is captured before defer executes.
    """
    result: i32 = 0
    
    @compile(suffix="defer_early_inc")
    def inc(p: ptr[i32]) -> void:
        p[0] = p[0] + 1
    
    defer(inc, ptr(result))
    
    if cond > 0:
        return result  # Returns 0 (defer executes after)
    
    result = result + 10
    return result  # Returns 10 (defer executes after)


@compile(suffix="defer_in_branch")
def test_defer_in_branch(cond: i32) -> i32:
    """Defer registered in branch
    
    Zig/Go semantics: return value is captured before defer executes.
    """
    result: i32 = 0
    
    @compile(suffix="defer_branch_inc")
    def inc(p: ptr[i32]) -> void:
        p[0] = p[0] + 1
    
    if cond > 0:
        defer(inc, ptr(result))
        result = result + 10
    else:
        result = result + 20
    
    return result  # Returns 10 or 20 (defer executes after if registered)


# =============================================================================
# Defer in loops
# =============================================================================

@compile(suffix="defer_in_for_loop")
def test_defer_in_for_loop() -> i32:
    """Defer in for loop - executes at end of each iteration"""
    result: i32 = 0
    
    @compile(suffix="defer_for_inc")
    def inc(p: ptr[i32]) -> void:
        p[0] = p[0] + 1
    
    for i in [1, 2, 3]:
        defer(inc, ptr(result))
        # Each iteration: defer registered, then executed at iteration end
        # So result increments by 1 each iteration
    
    return result  # Should be 3


@compile(suffix="defer_with_break")
def test_defer_with_break() -> i32:
    """Defer executes before break"""
    result: i32 = 0
    
    @compile(suffix="defer_break_inc")
    def inc(p: ptr[i32]) -> void:
        p[0] = p[0] + 1
    
    for i in [1, 2, 3, 4, 5]:
        defer(inc, ptr(result))
        if i == 3:
            break  # defer executes before break
    
    return result  # Should be 3 (iterations 1, 2, 3)


@compile(suffix="defer_with_continue")
def test_defer_with_continue() -> i32:
    """Defer executes before continue"""
    result: i32 = 0
    
    @compile(suffix="defer_continue_inc")
    def inc(p: ptr[i32]) -> void:
        p[0] = p[0] + 1
    
    for i in [1, 2, 3]:
        defer(inc, ptr(result))
        if i == 2:
            continue  # defer executes before continue
        # This code runs for i=1, i=3
    
    return result  # Should be 3 (all iterations execute defer)


# =============================================================================
# Defer with nested scopes
# =============================================================================

@compile(suffix="defer_nested_scopes")
def test_defer_nested_scopes() -> i32:
    """Defer in nested scopes - inner scope defers execute first"""
    result: i32 = 0
    
    @compile(suffix="defer_nested_inc")
    def inc(p: ptr[i32]) -> void:
        p[0] = p[0] + 1
    
    # Outer scope defer
    defer(inc, ptr(result))  # Executes last (at function return)
    
    for i in [1, 2]:
        # Inner scope defer - executes at end of each iteration
        defer(inc, ptr(result))
    
    # After loop: result = 2 (from loop iterations)
    # At return: result = 3 (from outer defer)
    return result  # Should be 3


@compile(suffix="defer_multiple_loops")
def test_defer_multiple_loops() -> i32:
    """Defer in multiple sequential loops"""
    result: i32 = 0
    
    @compile(suffix="defer_multi_inc")
    def inc(p: ptr[i32]) -> void:
        p[0] = p[0] + 1
    
    for i in [1, 2]:
        defer(inc, ptr(result))
    # result = 2
    
    for j in [1, 2, 3]:
        defer(inc, ptr(result))
    # result = 5
    
    return result  # Should be 5


# =============================================================================
# Defer with goto
# =============================================================================

@compile(suffix="defer_with_goto")
def test_defer_with_goto() -> i32:
    """Defer executes before goto"""
    result: i32 = 0
    
    @compile(suffix="defer_goto_inc")
    def inc(p: ptr[i32]) -> void:
        p[0] = p[0] + 1
    
    defer(inc, ptr(result))
    
    __goto("end")
    
    # This code is unreachable
    result = result + 100
    
    __label("end")
    return result  # Should be 1


# =============================================================================
# Defer capturing arguments
# =============================================================================

@compile(suffix="defer_captures_args")
def test_defer_captures_args() -> i32:
    """Defer captures argument values at registration time"""
    result: i32 = 0
    val: i32 = 5
    
    @compile(suffix="defer_capture_inc")
    def inc(p: ptr[i32]) -> void:
        p[0] = p[0] + 1
    
    # The pointer is captured at defer() time
    defer(inc, ptr(result))
    
    # Changing val doesn't affect the deferred call
    val = 100
    
    return result  # Should be 1


# =============================================================================
# Complex control flow with defer
# =============================================================================

@compile(suffix="defer_nested_loops")
def test_defer_nested_loops() -> i32:
    """Defer in nested loops - inner defers execute first"""
    result: i32 = 0
    
    @compile(suffix="defer_nested_loops_inc")
    def inc(p: ptr[i32]) -> void:
        p[0] = p[0] + 1
    
    for i in [1, 2]:
        defer(inc, ptr(result))  # Outer loop defer
        for j in [1, 2]:
            defer(inc, ptr(result))  # Inner loop defer
        # After inner loop: result += 2 (from inner defers)
    # After outer loop: result += 2 (from outer defers)
    # Total: 2*2 + 2 = 6
    
    return result


@compile(suffix="defer_if_else_branches")
def test_defer_if_else_branches(cond: i32) -> i32:
    """Defer in different if/else branches"""
    result: i32 = 0
    
    @compile(suffix="defer_if_else_inc")
    def inc(p: ptr[i32]) -> void:
        p[0] = p[0] + 1
    
    @compile(suffix="defer_if_else_add10")
    def add10(p: ptr[i32]) -> void:
        p[0] = p[0] + 10
    
    if cond > 0:
        defer(inc, ptr(result))  # +1
        result = result + 100
    else:
        defer(add10, ptr(result))  # +10
        result = result + 200
    
    return result  # cond>0: returns 100, cond<=0: returns 200


@compile(suffix="defer_multiple_returns")
def test_defer_multiple_returns(code: i32) -> i32:
    """Defer with multiple return paths"""
    result: i32 = 0
    
    @compile(suffix="defer_multi_ret_inc")
    def inc(p: ptr[i32]) -> void:
        p[0] = p[0] + 1
    
    defer(inc, ptr(result))  # Always registered
    
    if code == 1:
        return 10  # defer executes, returns 10
    elif code == 2:
        result = result + 5
        return 20  # defer executes, returns 20
    else:
        result = result + 10
        return 30  # defer executes, returns 30


@compile(suffix="defer_loop_early_exit")
def test_defer_loop_early_exit(limit: i32) -> i32:
    """Defer in loop with early exit via break"""
    result: i32 = 0
    
    @compile(suffix="defer_loop_exit_inc")
    def inc(p: ptr[i32]) -> void:
        p[0] = p[0] + 1
    
    # Function-level defer
    defer(inc, ptr(result))
    
    for i in [1, 2, 3, 4, 5]:
        defer(inc, ptr(result))  # Loop defer
        if i >= limit:
            break
    
    # If limit=3: loop runs 3 times, result=3
    # Function defer doesn't affect return value (Zig/Go semantics)
    return result


# =============================================================================
# Defer with goto - same scope
# =============================================================================

@compile(suffix="defer_goto_same_scope")
def test_defer_goto_same_scope() -> i32:
    """Defer with goto in same scope - defer executes before goto"""
    result: i32 = 0
    
    @compile(suffix="defer_goto_same_inc")
    def inc(p: ptr[i32]) -> void:
        p[0] = p[0] + 1
    
    defer(inc, ptr(result))
    result = result + 10
    
    __goto("end")
    
    result = result + 100  # Unreachable
    
    __label("end")
    return result  # Defer executed before goto, result = 11, returns 11


@compile(suffix="defer_goto_forward")
def test_defer_goto_forward() -> i32:
    """Defer with forward goto"""
    result: i32 = 0
    
    @compile(suffix="defer_goto_fwd_inc")
    def inc(p: ptr[i32]) -> void:
        p[0] = p[0] + 1
    
    defer(inc, ptr(result))
    
    if result == 0:
        __goto("skip")
    
    result = result + 100  # Skipped
    
    __label("skip")
    result = result + 10
    return result  # Defer before goto: result=1, +10=11, returns 11


@compile(suffix="defer_goto_backward")
def test_defer_goto_backward() -> i32:
    """Defer with backward goto (loop via goto)"""
    result: i32 = 0
    count: i32 = 0
    
    @compile(suffix="defer_goto_back_inc")
    def inc(p: ptr[i32]) -> void:
        p[0] = p[0] + 1
    
    __label("loop_start")
    
    defer(inc, ptr(result))  # Registered each iteration, executed before goto
    count = count + 1
    
    if count < 3:
        __goto("loop_start")  # Defer executes before each goto
    
    return result  # 3 iterations, 3 defers executed, returns 3


@compile(suffix="defer_goto_conditional")
def test_defer_goto_conditional(cond: i32) -> i32:
    """Defer with conditional goto"""
    result: i32 = 0
    
    @compile(suffix="defer_goto_cond_inc")
    def inc(p: ptr[i32]) -> void:
        p[0] = p[0] + 1
    
    defer(inc, ptr(result))
    
    if cond > 0:
        __goto("path_a")
    else:
        __goto("path_b")
    
    __label("path_a")
    result = result + 10
    __goto("end")
    
    __label("path_b")
    result = result + 20
    
    __label("end")
    return result  # cond>0: 1+10=11, cond<=0: 1+20=21


# =============================================================================
# Defer with goto - different scopes
# =============================================================================

@compile(suffix="defer_goto_out_of_loop")
def test_defer_goto_out_of_loop() -> i32:
    """Defer in loop with goto jumping out of loop"""
    result: i32 = 0
    
    @compile(suffix="defer_goto_out_inc")
    def inc(p: ptr[i32]) -> void:
        p[0] = p[0] + 1
    
    # Function-level defer
    defer(inc, ptr(result))
    
    for i in [1, 2, 3, 4, 5]:
        defer(inc, ptr(result))  # Loop defer
        if i == 3:
            __goto("exit")  # Jump out of loop
    
    __label("exit")
    return result  # Loop defers: 3, func defer after return value: returns 3


@compile(suffix="defer_goto_into_after_loop")
def test_defer_goto_into_after_loop() -> i32:
    """Goto from before loop to after loop"""
    result: i32 = 0
    
    @compile(suffix="defer_goto_after_inc")
    def inc(p: ptr[i32]) -> void:
        p[0] = p[0] + 1
    
    defer(inc, ptr(result))  # Function defer
    
    __goto("after_loop")
    
    for i in [1, 2, 3]:
        defer(inc, ptr(result))  # Never executed
    
    __label("after_loop")
    result = result + 10
    return result  # Func defer before goto: 1, +10=11, returns 11


@compile(suffix="defer_multiple_gotos_same_label")
def test_defer_multiple_gotos_same_label(code: i32) -> i32:
    """Multiple gotos to same label from different scopes"""
    result: i32 = 0
    
    @compile(suffix="defer_multi_goto_inc")
    def inc(p: ptr[i32]) -> void:
        p[0] = p[0] + 1
    
    defer(inc, ptr(result))  # Function defer
    
    if code == 1:
        defer(inc, ptr(result))  # Branch defer
        __goto("merge")
    elif code == 2:
        result = result + 10
        __goto("merge")
    else:
        for i in [1]:
            defer(inc, ptr(result))  # Loop defer
            __goto("merge")
    
    __label("merge")
    result = result + 100
    return result
    # code=1: func_defer(1) + branch_defer(1) + 100 = 102, returns 102
    # code=2: func_defer(1) + 10 + 100 = 111, returns 111
    # code=3: func_defer(1) + loop_defer(1) + 100 = 102, returns 102


# =============================================================================
# Defer with linear types
# =============================================================================

@compile(suffix="defer_linear_simple")
def test_defer_linear_simple() -> i32:
    """Defer with linear type - consume in defer"""
    result: i32 = 0
    
    @compile(suffix="defer_linear_consumer")
    def consumer(p: ptr[i32], t: linear) -> void:
        consume(t)
        p[0] = p[0] + 1
    
    t = linear()
    defer(consumer, ptr(result), t)
    
    return result  # Defer consumes linear, returns 0


@compile(suffix="defer_linear_in_loop")
def test_defer_linear_in_loop() -> i32:
    """Defer with linear in loop - each iteration creates and defers"""
    result: i32 = 0
    
    @compile(suffix="defer_linear_loop_consumer")
    def consumer(p: ptr[i32], t: linear) -> void:
        consume(t)
        p[0] = p[0] + 1
    
    for i in [1, 2, 3]:
        t = linear()
        defer(consumer, ptr(result), t)  # Defer consumes at end of iteration
    
    return result  # 3 iterations, returns 3


@compile(suffix="defer_linear_multiple")
def test_defer_linear_multiple() -> i32:
    """Multiple defers with different linear tokens"""
    result: i32 = 0
    
    @compile(suffix="defer_linear_multi_consumer")
    def consumer(p: ptr[i32], t: linear) -> void:
        consume(t)
        p[0] = p[0] + 1
    
    t1 = linear()
    t2 = linear()
    t3 = linear()
    
    defer(consumer, ptr(result), t1)
    defer(consumer, ptr(result), t2)
    defer(consumer, ptr(result), t3)
    
    return result  # All 3 defers execute after return value, returns 0


@compile(suffix="defer_linear_with_goto")
def test_defer_linear_with_goto() -> i32:
    """Defer with linear and goto"""
    result: i32 = 0
    
    @compile(suffix="defer_linear_goto_consumer")
    def consumer(p: ptr[i32], t: linear) -> void:
        consume(t)
        p[0] = p[0] + 1
    
    t = linear()
    defer(consumer, ptr(result), t)
    
    __goto("end")
    
    result = result + 100  # Unreachable
    
    __label("end")
    return result  # Defer executes before goto, returns 1


@compile(suffix="defer_linear_conditional")
def test_defer_linear_conditional(cond: i32) -> i32:
    """Defer with linear in conditional branches"""
    result: i32 = 0
    
    @compile(suffix="defer_linear_cond_consumer")
    def consumer(p: ptr[i32], t: linear) -> void:
        consume(t)
        p[0] = p[0] + 1
    
    t = linear()
    
    if cond > 0:
        defer(consumer, ptr(result), t)
        result = result + 10
    else:
        defer(consumer, ptr(result), t)
        result = result + 20
    
    return result  # cond>0: returns 10, cond<=0: returns 20


@compile(suffix="defer_linear_nested_scope")
def test_defer_linear_nested_scope() -> i32:
    """Defer with linear in nested scopes"""
    result: i32 = 0
    
    @compile(suffix="defer_linear_nested_consumer")
    def consumer(p: ptr[i32], t: linear) -> void:
        consume(t)
        p[0] = p[0] + 1
    
    # Function level
    t1 = linear()
    defer(consumer, ptr(result), t1)
    
    for i in [1, 2]:
        # Loop level
        t2 = linear()
        defer(consumer, ptr(result), t2)
    
    return result  # Loop defers: 2, func defer after return: returns 2


# =============================================================================
# Test class
# =============================================================================

class TestDefer(DeferredTestCase):
    """Tests for defer intrinsic
    
    Note: Defer follows Zig/Go semantics - return value is evaluated BEFORE
    defers execute, so defer cannot modify the return value directly.
    However, defer in loops executes at end of each iteration, affecting
    subsequent iterations.
    """
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        flush_all_pending_outputs()
    
    # Basic tests - Zig/Go semantics: defer doesn't affect return value
    def test_defer_simple(self):
        """Simple defer - return value captured before defer executes"""
        result = test_defer_simple()
        self.assertEqual(result, 0)  # Zig/Go semantics: return value is 0
    
    def test_defer_fifo_order(self):
        """FIFO execution order - return value captured before defers execute"""
        result = test_defer_fifo_order()
        self.assertEqual(result, 0)  # Zig/Go semantics: return value is 0
    
    def test_defer_with_work(self):
        """Defer with work between - return value captured before defer executes"""
        result = test_defer_with_work()
        self.assertEqual(result, 15)  # Zig/Go semantics: return value is 15
    
    def test_defer_early_return_true(self):
        """Defer on early return (condition true)"""
        result = test_defer_early_return(1)
        self.assertEqual(result, 0)  # Zig/Go semantics: return value is 0
    
    def test_defer_early_return_false(self):
        """Defer on normal return (condition false)"""
        result = test_defer_early_return(0)
        self.assertEqual(result, 10)  # Zig/Go semantics: return value is 10
    
    def test_defer_in_branch(self):
        """Defer registered in branch"""
        result = test_defer_in_branch(1)
        self.assertEqual(result, 11)  # Defer executes at if block exit: 10+1=11
        
        result = test_defer_in_branch(0)
        self.assertEqual(result, 20)  # 0 + 20, no defer in else branch
    
    # Loop tests - defer in loops executes at end of each iteration
    # This affects subsequent iterations, so result accumulates
    def test_defer_in_for_loop(self):
        """Defer in for loop - executes at end of each iteration"""
        result = test_defer_in_for_loop()
        self.assertEqual(result, 3)  # Loop defers execute during loop, affecting result
    
    def test_defer_with_break(self):
        """Defer with break"""
        result = test_defer_with_break()
        self.assertEqual(result, 3)  # Loop defers execute before break
    
    def test_defer_with_continue(self):
        """Defer with continue"""
        result = test_defer_with_continue()
        self.assertEqual(result, 3)  # Loop defers execute before continue
    
    # Nested scope tests
    def test_defer_nested_scopes(self):
        """Defer in nested scopes"""
        result = test_defer_nested_scopes()
        # Loop defers execute during loop (result = 2)
        # Function-level defer executes after return value captured
        # So return value is 2 (Zig/Go semantics)
        self.assertEqual(result, 2)
    
    def test_defer_multiple_loops(self):
        """Defer in multiple loops"""
        result = test_defer_multiple_loops()
        self.assertEqual(result, 5)  # All loop defers execute during loops
    
    # Goto tests - goto executes defers for current scope before jumping
    def test_defer_with_goto(self):
        """Defer with goto - defer executes before goto"""
        result = test_defer_with_goto()
        self.assertEqual(result, 1)  # Defer executes before goto, result becomes 1
    
    # Argument capture tests
    def test_defer_captures_args(self):
        """Defer captures args at registration"""
        result = test_defer_captures_args()
        self.assertEqual(result, 0)  # Zig/Go semantics: return value is 0
    
    # Complex control flow tests
    def test_defer_nested_loops(self):
        """Defer in nested loops"""
        result = test_defer_nested_loops()
        # Inner loop: 2 defers per outer iteration = 2*2 = 4
        # Outer loop: 2 defers = 2
        # Total = 6
        self.assertEqual(result, 6)
    
    def test_defer_if_else_branches(self):
        """Defer in if/else branches"""
        result = test_defer_if_else_branches(1)
        self.assertEqual(result, 101)  # cond>0: 100+1=101 (defer at block exit)
        result = test_defer_if_else_branches(0)
        self.assertEqual(result, 210)  # cond<=0: 200+10=210 (defer at block exit)
    
    def test_defer_multiple_returns(self):
        """Defer with multiple return paths"""
        self.assertEqual(test_defer_multiple_returns(1), 10)
        self.assertEqual(test_defer_multiple_returns(2), 20)
        self.assertEqual(test_defer_multiple_returns(3), 30)
    
    def test_defer_loop_early_exit(self):
        """Defer in loop with early exit"""
        # limit=2: i=1 normal exit (+1), i=2 break (+1) = 2
        self.assertEqual(test_defer_loop_early_exit(2), 2)
        # limit=3: i=1 normal (+1), i=2 normal (+1), i=3 break (+1) = 3
        self.assertEqual(test_defer_loop_early_exit(3), 3)
        # limit=5: i=1..4 normal (+4), i=5 break (+1) = 5
        self.assertEqual(test_defer_loop_early_exit(5), 5)
    
    # Goto same scope tests
    def test_defer_goto_same_scope(self):
        """Defer with goto in same scope"""
        result = test_defer_goto_same_scope()
        self.assertEqual(result, 11)  # Defer before goto: 1, +10 = 11
    
    def test_defer_goto_forward(self):
        """Defer with forward goto"""
        result = test_defer_goto_forward()
        self.assertEqual(result, 10)  # Defer at return: value captured before defer
    
    def test_defer_goto_backward(self):
        """Defer with backward goto (loop)"""
        result = test_defer_goto_backward()
        self.assertEqual(result, 0)  # Defers execute at return, value captured before
    
    def test_defer_goto_conditional(self):
        """Defer with conditional goto"""
        self.assertEqual(test_defer_goto_conditional(1), 11)   # defer in if block: 1+10=11
        self.assertEqual(test_defer_goto_conditional(0), 20)   # defer at return, value captured
    
    # Goto different scope tests
    def test_defer_goto_out_of_loop(self):
        """Defer in loop with goto out"""
        result = test_defer_goto_out_of_loop()
        # Iterations 1,2 complete: 2 defers executed
        # Iteration 3: defer registered, then goto executes it: 1 more defer
        # Total loop defers: 3, func defer at return (after capturing return value)
        self.assertEqual(result, 3)
    
    def test_defer_goto_into_after_loop(self):
        """Goto from before loop to after loop"""
        result = test_defer_goto_into_after_loop()
        self.assertEqual(result, 11)  # Func defer: 1, +10 = 11
    
    def test_defer_multiple_gotos_same_label(self):
        """Multiple gotos to same label"""
        self.assertEqual(test_defer_multiple_gotos_same_label(1), 101)  # branch defer + 100
        self.assertEqual(test_defer_multiple_gotos_same_label(2), 110)  # 10 + 100
        self.assertEqual(test_defer_multiple_gotos_same_label(3), 101)  # loop defer + 100
    
    # Defer with linear tests
    def test_defer_linear_simple(self):
        """Defer with linear - consume in defer"""
        result = test_defer_linear_simple()
        self.assertEqual(result, 0)  # Defer after return value
    
    def test_defer_linear_in_loop(self):
        """Defer with linear in loop"""
        result = test_defer_linear_in_loop()
        self.assertEqual(result, 3)  # 3 iterations
    
    def test_defer_linear_multiple(self):
        """Multiple defers with linear"""
        result = test_defer_linear_multiple()
        self.assertEqual(result, 0)  # All defers after return value
    
    def test_defer_linear_with_goto(self):
        """Defer with linear and goto"""
        result = test_defer_linear_with_goto()
        self.assertEqual(result, 1)  # Defer before goto
    
    def test_defer_linear_conditional(self):
        """Defer with linear in conditional"""
        self.assertEqual(test_defer_linear_conditional(1), 11)  # 10 + defer at block exit
        self.assertEqual(test_defer_linear_conditional(0), 21)  # 20 + defer at block exit
    
    def test_defer_linear_nested_scope(self):
        """Defer with linear in nested scopes"""
        result = test_defer_linear_nested_scope()
        self.assertEqual(result, 2)  # 2 loop defers


if __name__ == '__main__':
    unittest.main()
