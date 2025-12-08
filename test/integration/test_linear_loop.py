"""
Tests for linear token handling in loop statements

Key rules for linear tokens in loops:
1. for loop body: cannot consume token (would consume multiple times)
2. for-else: else branch runs when loop completes normally (no break)
3. while loop body: cannot consume token (would consume multiple times)
4. Loop with break+return: can consume token if all exit paths consume exactly once
5. Token created inside loop: must be consumed within same iteration
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import unittest

from pythoc.decorators.compile import compile
from pythoc.builtin_entities import linear, consume, void, i32, i8
from pythoc.std.utility import move
from pythoc.build.output_manager import flush_all_pending_outputs, clear_failed_group
from pythoc import seq


# =============================================================================
# Valid patterns - Loop with linear tokens
# =============================================================================

@compile
def test_token_before_loop_consume_after(n: i32) -> void:
    """Token created before loop, consumed after loop"""
    t = linear()
    sum: i32 = 0
    for i in seq(n):
        sum = sum + i
    consume(t)


@compile
def test_token_created_consumed_each_iteration(n: i32) -> i32:
    """Token created and consumed within each iteration"""
    sum: i32 = 0
    for i in seq(n):
        t = linear()
        sum = sum + i
        consume(t)
    return sum


@compile
def test_while_token_before_after(n: i32) -> void:
    """While loop: token before, consume after"""
    t = linear()
    i: i32 = 0
    while i < n:
        i = i + 1
    consume(t)


@compile
def test_while_token_each_iteration(n: i32) -> i32:
    """While loop: token created and consumed each iteration"""
    sum: i32 = 0
    i: i32 = 0
    while i < n:
        t = linear()
        sum = sum + i
        consume(t)
        i = i + 1
    return sum


@compile
def test_for_else_token_in_else(n: i32) -> void:
    """Token consumed in else branch (loop completes normally)"""
    t = linear()
    for i in seq(n):
        pass
    else:
        consume(t)


@compile
def test_for_else_no_break_consume_else(n: i32) -> i32:
    """For-else: no break, else executes and consumes"""
    t = linear()
    sum: i32 = 0
    for i in seq(n):
        sum = sum + i
    else:
        consume(t)
        sum = sum + 100
    return sum


@compile
def test_nested_loop_token_outer(a: i32, b: i32) -> void:
    """Nested loops: token created before, consumed after all"""
    t = linear()
    sum: i32 = 0
    for i in seq(a):
        for j in seq(b):
            sum = sum + i * j
    consume(t)


@compile
def test_nested_loop_token_each_inner(a: i32, b: i32) -> i32:
    """Nested loops: token created and consumed in inner loop"""
    sum: i32 = 0
    for i in seq(a):
        for j in seq(b):
            t = linear()
            sum = sum + i + j
            consume(t)
    return sum


@compile
def test_constant_for_token_before_after() -> void:
    """Constant for loop: token before, consume after"""
    t = linear()
    sum: i32 = 0
    for i in [1, 2, 3, 4, 5]:
        sum = sum + i
    consume(t)


@compile
def test_constant_for_token_each_iteration() -> i32:
    """Constant for loop: token each iteration"""
    sum: i32 = 0
    for i in [10, 20, 30]:
        t = linear()
        sum = sum + i
        consume(t)
    return sum


@compile
def consume_linear_void(t: linear) -> void:
    """Helper: consume token"""
    consume(t)


@compile
def test_loop_pass_token_to_function(n: i32) -> void:
    """Pass token to function after loop"""
    t = linear()
    sum: i32 = 0
    for i in seq(n):
        sum = sum + i
    consume_linear_void(move(t))


@compile
def test_multiple_tokens_loop(n: i32) -> void:
    """Multiple tokens: both consumed after loop"""
    t1 = linear()
    t2 = linear()
    for i in seq(n):
        pass
    consume(t1)
    consume(t2)


# =============================================================================
# Error cases - these should fail to compile
# =============================================================================

def test_for_consume_in_body_error():
    """Test error when consuming token in loop body"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_for_consume')
    try:
        @compile(suffix="bad_for_consume")
        def bad_for_consume(n: i32) -> void:
            t = linear()
            for i in seq(n):
                consume(t)  # ERROR: would consume multiple times

        flush_all_pending_outputs()
        return False, "should have raised TypeError"
    except TypeError as e:
        return True, str(e)
    finally:
        clear_failed_group(group_key)


def test_while_consume_in_body_error():
    """Test error when consuming token in while body"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_while_consume')
    try:
        @compile(suffix="bad_while_consume")
        def bad_while_consume(n: i32) -> void:
            t = linear()
            i: i32 = 0
            while i < n:
                consume(t)  # ERROR: would consume multiple times
                i = i + 1

        flush_all_pending_outputs()
        return False, "should have raised TypeError"
    except TypeError as e:
        return True, str(e)
    finally:
        clear_failed_group(group_key)


def test_for_token_not_consumed_error():
    """Test error when token created in loop but not consumed"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_for_no_consume')
    try:
        @compile(suffix="bad_for_no_consume")
        def bad_for_no_consume(n: i32) -> void:
            for i in seq(n):
                t = linear()
                # ERROR: t not consumed in this iteration

        flush_all_pending_outputs()
        return False, "should have raised TypeError"
    except TypeError as e:
        if "not consumed" in str(e).lower():
            return True, str(e)
        return False, f"wrong error: {e}"
    finally:
        clear_failed_group(group_key)


def test_for_else_inconsistent_error():
    """Test error when for-else branches handle token inconsistently"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_for_else')
    try:
        @compile(suffix="bad_for_else")
        def bad_for_else(n: i32) -> void:
            t = linear()
            for i in seq(n):
                if i == 5:
                    consume(t)  # ERROR: consumed in loop body
                    break
            else:
                consume(t)

        flush_all_pending_outputs()
        return False, "should have raised TypeError"
    except TypeError as e:
        return True, str(e)
    finally:
        clear_failed_group(group_key)


def test_nested_loop_outer_consume_inner_error():
    """Test error when outer token consumed in inner loop"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_nested_consume')
    try:
        @compile(suffix="bad_nested_consume")
        def bad_nested_consume(a: i32, b: i32) -> void:
            t = linear()
            for i in seq(a):
                for j in seq(b):
                    consume(t)  # ERROR: would consume multiple times

        flush_all_pending_outputs()
        return False, "should have raised TypeError"
    except TypeError as e:
        return True, str(e)
    finally:
        clear_failed_group(group_key)


def test_token_after_loop_not_consumed_error():
    """Test error when token not consumed after loop"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_no_consume_after')
    try:
        @compile(suffix="bad_no_consume_after")
        def bad_no_consume_after(n: i32) -> void:
            t = linear()
            for i in seq(n):
                pass
            # ERROR: t not consumed

        flush_all_pending_outputs()
        return False, "should have raised TypeError"
    except TypeError as e:
        if "not consumed" in str(e).lower():
            return True, str(e)
        return False, f"wrong error: {e}"
    finally:
        clear_failed_group(group_key)


# =============================================================================
# Test runner
# =============================================================================

class TestLinearLoop(unittest.TestCase):
    """Test linear token handling in loop statements"""

    def test_valid_token_before_after(self):
        """Test token before loop, consume after"""
        test_token_before_loop_consume_after(5)
        test_token_before_loop_consume_after(0)

    def test_valid_token_each_iteration(self):
        """Test token created and consumed each iteration"""
        result = test_token_created_consumed_each_iteration(5)
        self.assertEqual(result, 10)  # 0+1+2+3+4

    def test_valid_while_before_after(self):
        """Test while loop with token before/after"""
        test_while_token_before_after(5)
        test_while_token_before_after(0)

    def test_valid_while_each_iteration(self):
        """Test while loop with token each iteration"""
        result = test_while_token_each_iteration(5)
        self.assertEqual(result, 10)  # 0+1+2+3+4

    def test_valid_for_else_consume_else(self):
        """Test for-else with consume in else"""
        test_for_else_token_in_else(5)
        test_for_else_token_in_else(0)

    def test_valid_for_else_no_break(self):
        """Test for-else no break pattern"""
        result = test_for_else_no_break_consume_else(5)
        self.assertEqual(result, 110)  # 0+1+2+3+4+100

    def test_valid_nested_loop_outer(self):
        """Test nested loops with outer token"""
        test_nested_loop_token_outer(3, 3)
        test_nested_loop_token_outer(0, 5)

    def test_valid_nested_loop_inner(self):
        """Test nested loops with inner token"""
        result = test_nested_loop_token_each_inner(3, 2)
        # i=0: j=0,1 -> 0+1=1; i=1: j=0,1 -> 1+2=3; i=2: j=0,1 -> 2+3=5
        self.assertEqual(result, 9)

    def test_valid_constant_for_before_after(self):
        """Test constant for loop with token before/after"""
        test_constant_for_token_before_after()

    def test_valid_constant_for_each(self):
        """Test constant for loop with token each iteration"""
        result = test_constant_for_token_each_iteration()
        self.assertEqual(result, 60)  # 10+20+30

    def test_valid_pass_to_function(self):
        """Test passing token to function after loop"""
        test_loop_pass_token_to_function(5)

    def test_valid_multiple_tokens(self):
        """Test multiple tokens with loop"""
        test_multiple_tokens_loop(5)

    def test_error_for_consume_in_body(self):
        """Test error: consume in for body"""
        passed, msg = test_for_consume_in_body_error()
        self.assertTrue(passed, msg)

    def test_error_while_consume_in_body(self):
        """Test error: consume in while body"""
        passed, msg = test_while_consume_in_body_error()
        self.assertTrue(passed, msg)

    def test_error_for_token_not_consumed(self):
        """Test error: token in loop not consumed"""
        passed, msg = test_for_token_not_consumed_error()
        self.assertTrue(passed, msg)

    def test_error_for_else_inconsistent(self):
        """Test error: for-else inconsistent"""
        passed, msg = test_for_else_inconsistent_error()
        self.assertTrue(passed, msg)

    def test_error_nested_outer_in_inner(self):
        """Test error: outer token consumed in inner loop"""
        passed, msg = test_nested_loop_outer_consume_inner_error()
        self.assertTrue(passed, msg)

    def test_error_not_consumed_after_loop(self):
        """Test error: token not consumed after loop"""
        passed, msg = test_token_after_loop_not_consumed_error()
        self.assertTrue(passed, msg)


if __name__ == "__main__":
    unittest.main()
