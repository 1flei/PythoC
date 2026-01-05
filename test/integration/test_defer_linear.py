#!/usr/bin/env python3
"""
Tests for defer + linear type interactions with complex control flow.

This module tests:
1. Linear tokens passed to defer in various branch patterns
2. Linear tokens in nested control structures with defer
3. Multiple linear tokens with defer
4. Linear defer with goto/label
5. Linear defer with loops and early exit
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import unittest
from pythoc.decorators.compile import compile
from pythoc.builtin_entities import void, i32, defer, ptr, label, goto, goto_end, linear, consume
from pythoc.build.output_manager import flush_all_pending_outputs

from test.utils.test_utils import DeferredTestCase


# =============================================================================
# Helper functions
# =============================================================================

@compile(suffix="dl_consumer")
def consumer(p: ptr[i32], t: linear) -> void:
    """Consume linear token and increment counter"""
    consume(t)
    p[0] = p[0] + 1


@compile(suffix="dl_consumer2")
def consumer2(p: ptr[i32], t1: linear, t2: linear) -> void:
    """Consume two linear tokens and add 10"""
    consume(t1)
    consume(t2)
    p[0] = p[0] + 10


@compile(suffix="dl_add")
def add_val(p: ptr[i32], val: i32) -> void:
    p[0] = p[0] + val


# =============================================================================
# Section 1: Multi-branch linear defer
# =============================================================================

@compile
def test_linear_3way_branch(sel: i32) -> i32:
    """Linear in defer with 3-way branch"""
    result: i32 = 0
    t = linear()
    
    if sel == 1:
        defer(consumer, ptr(result), t)
        result = result + 100
    elif sel == 2:
        defer(consumer, ptr(result), t)
        result = result + 200
    else:
        defer(consumer, ptr(result), t)
        result = result + 300
    
    return result


@compile
def test_linear_deep_nesting(a: i32, b: i32, c: i32) -> i32:
    """Linear in defer with deeply nested branches"""
    result: i32 = 0
    t = linear()
    
    if a > 0:
        if b > 0:
            if c > 0:
                defer(consumer, ptr(result), t)
                result = result + 1
            else:
                defer(consumer, ptr(result), t)
                result = result + 2
        else:
            if c > 0:
                defer(consumer, ptr(result), t)
                result = result + 4
            else:
                defer(consumer, ptr(result), t)
                result = result + 8
    else:
        if b > 0:
            if c > 0:
                defer(consumer, ptr(result), t)
                result = result + 16
            else:
                defer(consumer, ptr(result), t)
                result = result + 32
        else:
            if c > 0:
                defer(consumer, ptr(result), t)
                result = result + 64
            else:
                defer(consumer, ptr(result), t)
                result = result + 128
    
    return result


@compile
def test_linear_mixed_consume(cond: i32) -> i32:
    """Linear consumed directly in one branch, via defer in another"""
    result: i32 = 0
    t = linear()
    
    if cond > 0:
        defer(consumer, ptr(result), t)
        result = result + 100
    else:
        consume(t)
        result = result + 200
    
    return result


# =============================================================================
# Section 2: Multiple linear tokens with defer
# =============================================================================

@compile
def test_two_linear_same_defer() -> i32:
    """Two linear tokens passed to same defer call"""
    result: i32 = 0
    t1 = linear()
    t2 = linear()
    defer(consumer2, ptr(result), t1, t2)
    result = result + 100
    return result  # 100 (defer executes after return value captured)


@compile
def test_two_linear_separate_defer() -> i32:
    """Two linear tokens passed to separate defer calls"""
    result: i32 = 0
    t1 = linear()
    t2 = linear()
    defer(consumer, ptr(result), t1)
    defer(consumer, ptr(result), t2)
    result = result + 100
    return result  # 100 (defer executes after return value captured)


@compile
def test_two_linear_branch_defer(cond: i32) -> i32:
    """Two linear tokens, different branches determine which goes to defer"""
    result: i32 = 0
    t1 = linear()
    t2 = linear()
    
    if cond > 0:
        defer(consumer, ptr(result), t1)
        consume(t2)
        result = result + 100
    else:
        consume(t1)
        defer(consumer, ptr(result), t2)
        result = result + 200
    
    return result


@compile
def test_multiple_linear_chain() -> i32:
    """Chain of linear tokens with sequential defers"""
    result: i32 = 0
    t1 = linear()
    defer(consumer, ptr(result), t1)
    
    t2 = linear()
    defer(consumer, ptr(result), t2)
    
    t3 = linear()
    defer(consumer, ptr(result), t3)
    
    result = result + 1000
    return result  # 1000 (defer executes after return value captured)


# =============================================================================
# Section 3: Linear defer with goto/label
# =============================================================================

@compile
def test_linear_goto_simple(cond: i32) -> i32:
    """Linear defer with simple goto"""
    result: i32 = 0
    t = linear()
    
    with label("main"):
        defer(consumer, ptr(result), t)
        if cond > 0:
            goto_end("main")
        result = result + 100
    
    return result


@compile
def test_linear_goto_nested_labels(sel: i32) -> i32:
    """Linear defer with nested labels and goto"""
    result: i32 = 0
    t = linear()
    
    with label("outer"):
        with label("inner"):
            if sel == 1:
                defer(consumer, ptr(result), t)
                goto_end("inner")
            elif sel == 2:
                defer(consumer, ptr(result), t)
                goto_end("outer")
            else:
                defer(consumer, ptr(result), t)
            result = result + 10
        result = result + 100
    
    return result


@compile
def test_linear_goto_multi_defer(cond: i32) -> i32:
    """Multiple defers with goto - only relevant ones execute"""
    result: i32 = 0
    t = linear()
    
    defer(add_val, ptr(result), 1000)  # Function level
    
    with label("block"):
        defer(consumer, ptr(result), t)  # Block level
        defer(add_val, ptr(result), 100)
        if cond > 0:
            goto_end("block")
        result = result + 10
    
    return result


# =============================================================================
# Section 4: Linear defer in loops
# =============================================================================

@compile
def test_linear_loop_each_iter() -> i32:
    """Linear created and deferred each iteration"""
    result: i32 = 0
    
    for i in [1, 2, 3, 4, 5]:
        t = linear()
        defer(consumer, ptr(result), t)
    
    return result  # 5


@compile
def test_linear_loop_conditional_defer() -> i32:
    """Linear in loop with conditional defer vs direct consume"""
    result: i32 = 0
    
    for i in [1, 2, 3, 4, 5, 6]:
        t = linear()
        if i % 2 == 0:
            defer(consumer, ptr(result), t)
            result = result + 10
        else:
            consume(t)
            result = result + 1
    
    # Odd (1,3,5): consume directly, +1 each = 3
    # Even (2,4,6): defer, +10 each = 30, defer +1 each = 3
    return result  # 3 + 30 + 3 = 36


@compile
def test_linear_loop_break_defer(n: i32) -> i32:
    """Linear defer in loop with break"""
    result: i32 = 0
    
    for i in [1, 2, 3, 4, 5]:
        t = linear()
        defer(consumer, ptr(result), t)
        if i == n:
            break
    
    return result


@compile
def test_linear_loop_continue_defer() -> i32:
    """Linear defer in loop with continue"""
    result: i32 = 0
    total: i32 = 0
    
    for i in [1, 2, 3, 4, 5]:
        t = linear()
        defer(consumer, ptr(result), t)
        if i == 3:
            continue
        total = total + i
    
    # All 5 defers execute
    # total = 1 + 2 + 4 + 5 = 12
    return result * 100 + total  # 500 + 12 = 512


# =============================================================================
# Section 5: Linear defer with early return
# =============================================================================

@compile
def test_linear_early_return_if(cond: i32) -> i32:
    """Linear defer with early return in if"""
    result: i32 = 0
    t = linear()
    defer(consumer, ptr(result), t)
    
    if cond > 0:
        return 100
    
    return 200


@compile
def test_linear_early_return_nested(a: i32, b: i32) -> i32:
    """Linear defer with nested early returns"""
    result: i32 = 0
    t = linear()
    defer(consumer, ptr(result), t)
    
    if a > 0:
        if b > 0:
            return 1
        return 2
    else:
        if b > 0:
            return 3
    
    return 4


@compile
def test_linear_multi_return_paths(sel: i32) -> i32:
    """Linear defer with multiple return paths"""
    result: i32 = 0
    t = linear()
    defer(consumer, ptr(result), t)
    
    if sel == 1:
        return 100
    elif sel == 2:
        return 200
    elif sel == 3:
        return 300
    
    return 0


# =============================================================================
# Section 6: Complex combinations
# =============================================================================

@compile
def test_linear_loop_goto_defer() -> i32:
    """Linear defer in loop with goto to exit"""
    result: i32 = 0
    
    with label("loop_region"):
        for i in [1, 2, 3, 4, 5]:
            t = linear()
            defer(consumer, ptr(result), t)
            if i == 3:
                goto_end("loop_region")
    
    return result  # 3 defers (1, 2, 3)


@compile
def test_linear_nested_loop_defer() -> i32:
    """Linear defer in nested loops"""
    result: i32 = 0
    
    for i in [1, 2]:
        t1 = linear()
        defer(consumer, ptr(result), t1)
        for j in [1, 2, 3]:
            t2 = linear()
            defer(consumer, ptr(result), t2)
    
    # Outer: 2 defers
    # Inner: 2 * 3 = 6 defers
    return result  # 8


@compile
def test_linear_branch_loop_defer(cond: i32) -> i32:
    """Linear defer with branch inside loop"""
    result: i32 = 0
    
    for i in [1, 2, 3]:
        t = linear()
        if cond > 0:
            defer(consumer, ptr(result), t)
            result = result + 10
        else:
            defer(consumer, ptr(result), t)
            result = result + 100
    
    return result


@compile
def test_linear_defer_after_loop() -> i32:
    """Linear created before loop, deferred after"""
    result: i32 = 0
    t = linear()
    
    for i in [1, 2, 3]:
        result = result + i
    
    defer(consumer, ptr(result), t)
    return result  # 6 (defer executes after return value captured)


@compile
def test_linear_complex_cfg(a: i32, b: i32) -> i32:
    """Complex control flow graph with linear defer"""
    result: i32 = 0
    t = linear()
    
    with label("main"):
        if a > 0:
            defer(consumer, ptr(result), t)
            if b > 0:
                result = result + 1
                goto_end("main")
            result = result + 10
        else:
            if b > 0:
                defer(consumer, ptr(result), t)
                result = result + 100
            else:
                defer(consumer, ptr(result), t)
                result = result + 1000
                goto_end("main")
            result = result + 10000
    
    return result


# =============================================================================
# Test class
# =============================================================================

class TestDeferLinear(DeferredTestCase):
    """Tests for defer + linear type interactions"""
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        flush_all_pending_outputs()
    
    # Section 1: Multi-branch linear defer
    def test_linear_3way_branch_1(self):
        self.assertEqual(test_linear_3way_branch(1), 101)
    
    def test_linear_3way_branch_2(self):
        self.assertEqual(test_linear_3way_branch(2), 201)
    
    def test_linear_3way_branch_other(self):
        self.assertEqual(test_linear_3way_branch(0), 301)
    
    def test_linear_deep_nesting_all_paths(self):
        # Test all 8 paths through the nested branches
        self.assertEqual(test_linear_deep_nesting(1, 1, 1), 2)    # 1 + 1
        self.assertEqual(test_linear_deep_nesting(1, 1, 0), 3)    # 2 + 1
        self.assertEqual(test_linear_deep_nesting(1, 0, 1), 5)    # 4 + 1
        self.assertEqual(test_linear_deep_nesting(1, 0, 0), 9)    # 8 + 1
        self.assertEqual(test_linear_deep_nesting(0, 1, 1), 17)   # 16 + 1
        self.assertEqual(test_linear_deep_nesting(0, 1, 0), 33)   # 32 + 1
        self.assertEqual(test_linear_deep_nesting(0, 0, 1), 65)   # 64 + 1
        self.assertEqual(test_linear_deep_nesting(0, 0, 0), 129)  # 128 + 1
    
    def test_linear_mixed_consume_defer(self):
        self.assertEqual(test_linear_mixed_consume(1), 101)  # defer path
    
    def test_linear_mixed_consume_direct(self):
        self.assertEqual(test_linear_mixed_consume(0), 200)  # direct consume
    
    # Section 2: Multiple linear tokens
    def test_two_linear_same_defer(self):
        self.assertEqual(test_two_linear_same_defer(), 100)
    
    def test_two_linear_separate_defer(self):
        self.assertEqual(test_two_linear_separate_defer(), 100)
    
    def test_two_linear_branch_defer_true(self):
        self.assertEqual(test_two_linear_branch_defer(1), 101)
    
    def test_two_linear_branch_defer_false(self):
        self.assertEqual(test_two_linear_branch_defer(0), 201)
    
    def test_multiple_linear_chain(self):
        self.assertEqual(test_multiple_linear_chain(), 1000)
    
    # Section 3: Linear defer with goto/label
    def test_linear_goto_simple_skip(self):
        self.assertEqual(test_linear_goto_simple(1), 1)  # goto skips +100
    
    def test_linear_goto_simple_noskip(self):
        self.assertEqual(test_linear_goto_simple(0), 101)  # +100 + defer
    
    def test_linear_goto_nested_inner(self):
        # sel=1: goto_end inner, skip +10, execute +100
        self.assertEqual(test_linear_goto_nested_labels(1), 101)
    
    def test_linear_goto_nested_outer(self):
        # sel=2: goto_end outer, skip both +10 and +100
        self.assertEqual(test_linear_goto_nested_labels(2), 1)
    
    def test_linear_goto_nested_fallthrough(self):
        # sel=0: no goto, execute +10 and +100
        self.assertEqual(test_linear_goto_nested_labels(0), 111)
    
    def test_linear_goto_multi_defer_skip(self):
        # cond>0: goto_end block, skip +10, execute block defers (1+100) + func defer (1000)
        self.assertEqual(test_linear_goto_multi_defer(1), 101)
    
    def test_linear_goto_multi_defer_noskip(self):
        # cond<=0: +10, then block defers (1+100), then func defer (1000)
        self.assertEqual(test_linear_goto_multi_defer(0), 111)
    
    # Section 4: Linear defer in loops
    def test_linear_loop_each_iter(self):
        self.assertEqual(test_linear_loop_each_iter(), 5)
    
    def test_linear_loop_conditional_defer(self):
        self.assertEqual(test_linear_loop_conditional_defer(), 36)
    
    def test_linear_loop_break_defer_at_1(self):
        self.assertEqual(test_linear_loop_break_defer(1), 1)
    
    def test_linear_loop_break_defer_at_3(self):
        self.assertEqual(test_linear_loop_break_defer(3), 3)
    
    def test_linear_loop_break_defer_at_5(self):
        self.assertEqual(test_linear_loop_break_defer(5), 5)
    
    def test_linear_loop_break_defer_no_break(self):
        self.assertEqual(test_linear_loop_break_defer(10), 5)
    
    def test_linear_loop_continue_defer(self):
        self.assertEqual(test_linear_loop_continue_defer(), 512)
    
    # Section 5: Linear defer with early return
    def test_linear_early_return_if_true(self):
        self.assertEqual(test_linear_early_return_if(1), 100)
    
    def test_linear_early_return_if_false(self):
        self.assertEqual(test_linear_early_return_if(0), 200)
    
    def test_linear_early_return_nested_all_paths(self):
        self.assertEqual(test_linear_early_return_nested(1, 1), 1)
        self.assertEqual(test_linear_early_return_nested(1, 0), 2)
        self.assertEqual(test_linear_early_return_nested(0, 1), 3)
        self.assertEqual(test_linear_early_return_nested(0, 0), 4)
    
    def test_linear_multi_return_paths(self):
        self.assertEqual(test_linear_multi_return_paths(1), 100)
        self.assertEqual(test_linear_multi_return_paths(2), 200)
        self.assertEqual(test_linear_multi_return_paths(3), 300)
        self.assertEqual(test_linear_multi_return_paths(0), 0)
    
    # Section 6: Complex combinations
    def test_linear_loop_goto_defer(self):
        self.assertEqual(test_linear_loop_goto_defer(), 3)
    
    def test_linear_nested_loop_defer(self):
        self.assertEqual(test_linear_nested_loop_defer(), 8)
    
    def test_linear_branch_loop_defer_true(self):
        # cond>0: 3 iterations, each +10 + defer(+1) = 33
        self.assertEqual(test_linear_branch_loop_defer(1), 33)
    
    def test_linear_branch_loop_defer_false(self):
        # cond<=0: 3 iterations, each +100 + defer(+1) = 303
        self.assertEqual(test_linear_branch_loop_defer(0), 303)
    
    def test_linear_defer_after_loop(self):
        self.assertEqual(test_linear_defer_after_loop(), 6)
    
    def test_linear_complex_cfg_a1_b1(self):
        # a>0, b>0: +1, goto_end, defer -> 2
        self.assertEqual(test_linear_complex_cfg(1, 1), 2)
    
    def test_linear_complex_cfg_a1_b0(self):
        # a>0, b<=0: +10, defer -> 11
        self.assertEqual(test_linear_complex_cfg(1, 0), 11)
    
    def test_linear_complex_cfg_a0_b1(self):
        # a<=0, b>0: +100, +10000, defer -> 10101
        self.assertEqual(test_linear_complex_cfg(0, 1), 10101)
    
    def test_linear_complex_cfg_a0_b0(self):
        # a<=0, b<=0: +1000, goto_end, defer -> 1001
        self.assertEqual(test_linear_complex_cfg(0, 0), 1001)


if __name__ == '__main__':
    unittest.main()
