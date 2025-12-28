#!/usr/bin/env python3
"""
Test goto/label with linear types.

Linear types must be properly consumed on all paths, including goto paths.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import unittest
from pythoc.decorators.compile import compile
from pythoc.builtin_entities import linear, consume, void, i32, __label, __goto
from pythoc.std.utility import move
from pythoc.build.output_manager import flush_all_pending_outputs
from pythoc.libc.stdio import printf

from test.utils.test_utils import DeferredTestCase, expect_error


# =============================================================================
# Valid linear + goto tests
# =============================================================================

@compile(suffix="goto_linear_simple")
def test_goto_linear_simple() -> void:
    """Simple goto with linear - consume before goto"""
    t = linear()
    consume(t)
    __goto("end")
    __label("end")


@compile(suffix="goto_linear_after_label")
def test_goto_linear_after_label() -> void:
    """Create and consume linear after label"""
    __goto("start")
    __label("start")
    t = linear()
    consume(t)


@compile(suffix="goto_linear_in_branch")
def test_goto_linear_in_branch() -> i32:
    """Linear consumed in different branches with goto"""
    x: i32 = 5
    t = linear()
    
    if x > 3:
        consume(t)
        __goto("done")
    else:
        consume(t)
        __goto("done")
    
    __label("done")
    return x


@compile(suffix="goto_linear_loop")
def test_goto_linear_loop() -> i32:
    """Linear in goto-based loop"""
    count: i32 = 0
    
    __label("loop")
    t = linear()
    consume(t)
    count = count + 1
    if count < 3:
        __goto("loop")
    
    return count


@compile(suffix="goto_linear_move_then_goto")
def test_goto_linear_move_then_goto() -> void:
    """Move linear then goto"""
    t1 = linear()
    t2 = move(t1)
    __goto("end")
    __label("end")
    consume(t2)


@compile(suffix="goto_linear_multiple_tokens")
def test_goto_linear_multiple_tokens() -> i32:
    """Multiple linear tokens with goto"""
    t1 = linear()
    t2 = linear()
    
    consume(t1)
    __goto("middle")
    
    __label("middle")
    consume(t2)
    return 1


@compile(suffix="goto_linear_diamond")
def test_goto_linear_diamond() -> i32:
    """Diamond pattern with linear and goto - both paths consume before goto"""
    x: i32 = 10
    t = linear()
    
    if x > 5:
        consume(t)
        __goto("merge")
    else:
        consume(t)
        __goto("merge")
    
    __label("merge")
    return x


@compile(suffix="goto_linear_state_machine")
def test_goto_linear_state_machine() -> i32:
    """State machine with linear tokens"""
    state: i32 = 0
    result: i32 = 0
    
    __label("state_0")
    if state == 0:
        t = linear()
        consume(t)
        result = result + 1
        state = 1
        __goto("state_1")
    
    __label("state_1")
    if state == 1:
        t2 = linear()
        consume(t2)
        result = result + 10
        state = 2
        __goto("state_2")
    
    __label("state_2")
    if state == 2:
        t3 = linear()
        consume(t3)
        result = result + 100
    
    return result  # Expected: 111


# =============================================================================
# Valid merge point tests - consistent linear states
# =============================================================================

@compile(suffix="goto_merge_all_consumed")
def test_goto_merge_all_consumed() -> i32:
    """Multiple gotos to same label, all paths consume the token"""
    code: i32 = 2
    t = linear()
    
    if code == 1:
        consume(t)
        __goto("end")
    elif code == 2:
        consume(t)
        __goto("end")
    else:
        consume(t)
        __goto("end")
    
    __label("end")
    return code  # Expected: 2


@compile(suffix="goto_merge_all_active")
def test_goto_merge_all_active() -> i32:
    """Multiple gotos to same label, all paths keep token active"""
    code: i32 = 2
    t = linear()
    
    if code == 1:
        __goto("end")
    elif code == 2:
        __goto("end")
    else:
        __goto("end")
    
    __label("end")
    consume(t)  # Consume at merge point
    return code  # Expected: 2


@compile(suffix="goto_merge_with_fallthrough_consumed")
def test_goto_merge_with_fallthrough_consumed() -> i32:
    """Goto and fallthrough both consume before reaching label"""
    x: i32 = 3
    t = linear()
    
    if x > 10:
        consume(t)
        __goto("target")
    else:
        consume(t)
        # fallthrough to target
    
    __label("target")
    return x  # Expected: 3


@compile(suffix="goto_merge_with_fallthrough_active")
def test_goto_merge_with_fallthrough_active() -> i32:
    """Goto and fallthrough both keep token active"""
    x: i32 = 3
    t = linear()
    
    if x > 10:
        __goto("target")
    # else: fallthrough - token still active
    
    __label("target")
    consume(t)  # Consume at merge point
    return x  # Expected: 3


@compile(suffix="goto_merge_three_paths")
def test_goto_merge_three_paths() -> i32:
    """Three paths (two gotos + fallthrough) all consistent"""
    x: i32 = 5
    t = linear()
    
    if x > 10:
        consume(t)
        __goto("merge")
    elif x > 20:
        consume(t)
        __goto("merge")
    else:
        consume(t)
        # fallthrough to merge
    
    __label("merge")
    return x  # Expected: 5


@compile(suffix="goto_merge_nested_if")
def test_goto_merge_nested_if() -> i32:
    """Nested if with gotos, all paths consistent"""
    x: i32 = 5
    y: i32 = 10
    t = linear()
    
    if x > 0:
        if y > 5:
            consume(t)
            __goto("done")
        else:
            consume(t)
            __goto("done")
    else:
        consume(t)
        __goto("done")
    
    __label("done")
    return x + y  # Expected: 15


@compile(suffix="goto_merge_loop_consistent")
def test_goto_merge_loop_consistent() -> i32:
    """Loop via goto with consistent linear state - create fresh each iteration"""
    count: i32 = 0
    
    __label("loop")
    t = linear()  # Create fresh token each iteration
    consume(t)    # Consume it
    count = count + 1
    
    if count < 3:
        __goto("loop")  # Loop back - no token active
    
    return count  # Expected: 3


@compile(suffix="goto_merge_alternating_consistent")
def test_goto_merge_alternating_consistent() -> i32:
    """Alternating between labels with consistent state"""
    count: i32 = 0
    result: i32 = 0
    
    __label("ping")
    if count >= 4:
        __goto("done")
    t1 = linear()
    consume(t1)
    result = result + 1
    count = count + 1
    __goto("pong")
    
    __label("pong")
    if count >= 4:
        __goto("done")
    t2 = linear()
    consume(t2)
    result = result + 2
    count = count + 1
    __goto("ping")
    
    __label("done")
    return result  # Expected: 1+2+1+2 = 6


# =============================================================================
# Error tests - linear not consumed on goto path
# =============================================================================

@expect_error(["not consumed"], suffix="goto_linear_not_consumed")
def run_error_goto_linear_not_consumed():
    @compile(suffix="goto_linear_not_consumed")
    def bad() -> void:
        t = linear()
        __goto("end")  # ERROR: t not consumed before goto
        consume(t)
        __label("end")


@expect_error(["not consumed"], suffix="goto_linear_branch_missing")
def run_error_goto_linear_branch_missing():
    @compile(suffix="goto_linear_branch_missing")
    def bad() -> i32:
        x: i32 = 5
        t = linear()
        
        if x > 3:
            consume(t)
            __goto("done")
        else:
            __goto("done")  # ERROR: t not consumed in else branch
        
        __label("done")
        return x


@expect_error(["consumed", "not consumed"], suffix="goto_linear_double_consume")
def run_error_goto_linear_double_consume():
    @compile(suffix="goto_linear_double_consume")
    def bad() -> void:
        t = linear()
        consume(t)
        __goto("again")
        __label("again")
        consume(t)  # ERROR: already consumed


# =============================================================================
# Error tests - merge point inconsistency via goto
# =============================================================================

@expect_error(["Inconsistent", "merge"], suffix="goto_merge_inconsistent_1")
def run_error_goto_merge_inconsistent_1():
    """Two goto paths to same label with different linear states"""
    @compile(suffix="goto_merge_inconsistent_1")
    def bad() -> i32:
        x: i32 = 5
        t = linear()
        
        if x > 3:
            consume(t)
            __goto("merge")  # t consumed
        else:
            __goto("merge")  # t NOT consumed - inconsistent!
        
        __label("merge")
        return x


@expect_error(["Inconsistent", "merge"], suffix="goto_merge_inconsistent_2")
def run_error_goto_merge_inconsistent_2():
    """Goto and fallthrough to same label with different states"""
    @compile(suffix="goto_merge_inconsistent_2")
    def bad() -> i32:
        x: i32 = 5
        t = linear()
        
        if x > 10:
            __goto("target")  # t NOT consumed
        
        # Fallthrough path
        consume(t)  # t consumed here
        
        __label("target")  # merge point: one path consumed, one not
        return x


@expect_error(["Inconsistent", "merge"], suffix="goto_merge_inconsistent_3")
def run_error_goto_merge_inconsistent_3():
    """Multiple gotos to same label with inconsistent states"""
    @compile(suffix="goto_merge_inconsistent_3")
    def bad() -> i32:
        code: i32 = 2
        t = linear()
        
        if code == 1:
            consume(t)
            __goto("end")  # consumed
        elif code == 2:
            __goto("end")  # NOT consumed - inconsistent!
        else:
            consume(t)
            __goto("end")  # consumed
        
        __label("end")
        return code


@expect_error(["Inconsistent", "merge"], suffix="goto_merge_different_tokens")
def run_error_goto_merge_different_tokens():
    """Different tokens active at merge point"""
    @compile(suffix="goto_merge_different_tokens")
    def bad() -> i32:
        x: i32 = 5
        
        if x > 3:
            t1 = linear()
            __goto("merge")  # t1 active
        else:
            t2 = linear()
            __goto("merge")  # t2 active - different token!
        
        __label("merge")
        # Which token to consume here? t1 or t2?
        return x


@expect_error(["Inconsistent", "merge"], suffix="goto_loop_merge_inconsistent")
def run_error_goto_loop_merge_inconsistent():
    """Loop via goto with inconsistent linear state at loop header"""
    @compile(suffix="goto_loop_merge_inconsistent")
    def bad() -> i32:
        count: i32 = 0
        t = linear()
        
        __label("loop")  # merge point: first entry has t, loop back doesn't
        count = count + 1
        
        if count < 3:
            consume(t)  # consume on first iteration
            __goto("loop")  # loop back without t
        
        return count


# =============================================================================
# Test class
# =============================================================================

class TestGotoLinear(DeferredTestCase):
    """Tests for goto/label with linear types"""
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        flush_all_pending_outputs()
    
    # Valid tests - basic
    def test_goto_linear_simple(self):
        """Simple goto with linear"""
        test_goto_linear_simple()
    
    def test_goto_linear_after_label(self):
        """Create linear after label"""
        test_goto_linear_after_label()
    
    def test_goto_linear_in_branch(self):
        """Linear consumed in branches"""
        result = test_goto_linear_in_branch()
        self.assertEqual(result, 5)
    
    def test_goto_linear_loop(self):
        """Linear in goto loop"""
        result = test_goto_linear_loop()
        self.assertEqual(result, 3)
    
    def test_goto_linear_move_then_goto(self):
        """Move then goto"""
        test_goto_linear_move_then_goto()
    
    def test_goto_linear_multiple_tokens(self):
        """Multiple tokens with goto"""
        result = test_goto_linear_multiple_tokens()
        self.assertEqual(result, 1)
    
    def test_goto_linear_diamond(self):
        """Diamond pattern"""
        result = test_goto_linear_diamond()
        self.assertEqual(result, 10)
    
    def test_goto_linear_state_machine(self):
        """State machine pattern"""
        result = test_goto_linear_state_machine()
        self.assertEqual(result, 111)
    
    # Valid tests - merge point consistent
    def test_goto_merge_all_consumed(self):
        """All paths consume before merge"""
        result = test_goto_merge_all_consumed()
        self.assertEqual(result, 2)
    
    def test_goto_merge_all_active(self):
        """All paths keep token active, consume at merge"""
        result = test_goto_merge_all_active()
        self.assertEqual(result, 2)
    
    def test_goto_merge_with_fallthrough_consumed(self):
        """Goto and fallthrough both consume"""
        result = test_goto_merge_with_fallthrough_consumed()
        self.assertEqual(result, 3)
    
    def test_goto_merge_with_fallthrough_active(self):
        """Goto and fallthrough both keep active"""
        result = test_goto_merge_with_fallthrough_active()
        self.assertEqual(result, 3)
    
    def test_goto_merge_three_paths(self):
        """Three paths all consistent"""
        result = test_goto_merge_three_paths()
        self.assertEqual(result, 5)
    
    def test_goto_merge_nested_if(self):
        """Nested if with consistent gotos"""
        result = test_goto_merge_nested_if()
        self.assertEqual(result, 15)
    
    def test_goto_merge_loop_consistent(self):
        """Loop via goto with fresh token each iteration"""
        result = test_goto_merge_loop_consistent()
        self.assertEqual(result, 3)
    
    def test_goto_merge_alternating_consistent(self):
        """Alternating labels with consistent state"""
        result = test_goto_merge_alternating_consistent()
        self.assertEqual(result, 6)
    
    # Error tests - basic
    def test_error_goto_linear_not_consumed(self):
        """Error: linear not consumed before goto"""
        run_error_goto_linear_not_consumed()
    
    def test_error_goto_linear_branch_missing(self):
        """Error: linear not consumed in one branch"""
        run_error_goto_linear_branch_missing()
    
    def test_error_goto_linear_double_consume(self):
        """Error: double consume via goto"""
        run_error_goto_linear_double_consume()
    
    # Error tests - merge point inconsistency
    def test_error_goto_merge_inconsistent_1(self):
        """Error: two goto paths with different linear states"""
        run_error_goto_merge_inconsistent_1()
    
    def test_error_goto_merge_inconsistent_2(self):
        """Error: goto and fallthrough with different states"""
        run_error_goto_merge_inconsistent_2()
    
    def test_error_goto_merge_inconsistent_3(self):
        """Error: multiple gotos with inconsistent states"""
        run_error_goto_merge_inconsistent_3()
    
    def test_error_goto_merge_different_tokens(self):
        """Error: different tokens active at merge"""
        run_error_goto_merge_different_tokens()
    
    def test_error_goto_loop_merge_inconsistent(self):
        """Error: loop via goto with inconsistent state"""
        run_error_goto_loop_merge_inconsistent()


if __name__ == '__main__':
    unittest.main()
