"""
Tests for linear token handling in if-elif-else chains

Key rules for linear tokens in if statements:
1. if-else: both branches must handle token consistently (both consume or both don't)
2. if without else, no return: token cannot be modified in then branch
3. if without else, with return: then branch can consume token, code after if
   only executes when condition is false (token still active)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import unittest

from pythoc.decorators.compile import compile
from pythoc.builtin_entities import linear, consume, void, i32, i8
from pythoc.std.utility import move
from pythoc.build.output_manager import flush_all_pending_outputs, clear_failed_group


# =============================================================================
# Valid patterns - these should all compile and run successfully
# =============================================================================

@compile
def test_if_elif_else_all_consume(cond: i32) -> void:
    """Test if-elif-else where all branches consume the token"""
    t = linear()
    if cond == 1:
        consume(t)
    elif cond == 2:
        consume(t)
    else:
        consume(t)


@compile
def test_if_elif_elif_else_all_consume(cond: i32) -> void:
    """Test longer if-elif chain"""
    t = linear()
    if cond == 1:
        consume(t)
    elif cond == 2:
        consume(t)
    elif cond == 3:
        consume(t)
    elif cond == 4:
        consume(t)
    else:
        consume(t)


@compile
def consume_and_return(t: linear) -> i32:
    consume(t)
    return 1


@compile
def test_if_if_return_pattern(dispatch: i8) -> i32:
    """
    Test the if-if-return pattern.
    Each path consumes token exactly once.
    """
    t = linear()
    if dispatch == 2:
        return consume_and_return(move(t))
    if dispatch == 1:
        return consume_and_return(move(t))
    return consume_and_return(move(t))


@compile
def consume_and_return2(t: linear) -> i32:
    consume(t)
    return 1


@compile
def test_if_elif_return_pattern(dispatch: i8) -> i32:
    """Test the if-elif-else pattern with returns."""
    t = linear()
    if dispatch == 2:
        return consume_and_return2(move(t))
    elif dispatch == 1:
        return consume_and_return2(move(t))
    else:
        return consume_and_return2(move(t))


@compile
def test_nested_if_with_linear(a: i32, b: i32) -> void:
    """Test nested if statements with linear tokens"""
    t = linear()
    if a > 0:
        if b > 0:
            consume(t)
        else:
            consume(t)
    else:
        consume(t)


@compile
def _dispatch_impl(t: linear, code: i8) -> i32:
    """Dispatch helper that consumes token exactly once"""
    if code == 2:
        consume(t)
        return 2
    elif code == 1:
        consume(t)
        return 1
    else:
        consume(t)
        return 0


@compile
def test_dispatch_helper_pattern(cond: i8) -> i32:
    """Test the dispatch helper pattern."""
    t = linear()
    code: i8 = 0
    if cond > 10:
        code = 2
    elif cond > 5:
        code = 1
    return _dispatch_impl(move(t), code)


@compile
def test_multiple_linear_tokens_if_elif(cond: i32) -> void:
    """Test multiple linear tokens in if-elif-else"""
    t1 = linear()
    t2 = linear()
    if cond == 1:
        consume(t1)
        consume(t2)
    elif cond == 2:
        consume(t1)
        consume(t2)
    else:
        consume(t1)
        consume(t2)


# =============================================================================
# Error cases - these should fail to compile
# =============================================================================

def test_if_without_else_no_return_error():
    """Test that if without else (and no return) cannot consume token"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_no_else')
    try:
        @compile(suffix="bad_no_else")
        def bad_no_else(cond: i32) -> void:
            t = linear()
            if cond:
                consume(t)
            # ERROR: no else, no return - token might not be consumed
        
        flush_all_pending_outputs()
        return False, "should have raised RuntimeError"
    except RuntimeError as e:
        if "without else" in str(e).lower() or "consistently" in str(e).lower():
            return True, str(e)
        return False, f"wrong error: {e}"
    finally:
        clear_failed_group(group_key)


def test_if_elif_missing_else_error():
    """Test that if-elif without else cannot consume token"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_no_final_else')
    try:
        @compile(suffix="bad_no_final_else")
        def bad_no_final_else(cond: i32) -> void:
            t = linear()
            if cond == 1:
                consume(t)
            elif cond == 2:
                consume(t)
            # ERROR: no else branch
        
        flush_all_pending_outputs()
        return False, "should have raised RuntimeError"
    except RuntimeError as e:
        if "without else" in str(e).lower() or "consistently" in str(e).lower():
            return True, str(e)
        return False, f"wrong error: {e}"
    finally:
        clear_failed_group(group_key)


def test_if_elif_inconsistent_error():
    """Test error when if-elif-else branches handle token inconsistently"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_inconsistent')
    try:
        @compile(suffix="bad_inconsistent")
        def bad_inconsistent(cond: i32) -> void:
            t = linear()
            if cond == 1:
                consume(t)
            elif cond == 2:
                pass  # ERROR: doesn't consume
            else:
                consume(t)
        
        flush_all_pending_outputs()
        return False, "should have raised RuntimeError"
    except RuntimeError as e:
        if "consistently" in str(e).lower():
            return True, str(e)
        return False, f"wrong error: {e}"
    finally:
        clear_failed_group(group_key)


def test_if_return_missing_final_consume_error():
    """Test error when if-if-return pattern is missing final consume"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_missing_final')
    try:
        @compile(suffix="bad_missing_final")
        def bad_missing_final(cond: i32) -> void:
            t = linear()
            if cond == 1:
                consume(t)
                return
            if cond == 2:
                consume(t)
                return
            # ERROR: missing consume(t) here
        
        flush_all_pending_outputs()
        return False, "should have raised RuntimeError"
    except RuntimeError as e:
        if "not consumed" in str(e).lower():
            return True, str(e)
        return False, f"wrong error: {e}"
    finally:
        clear_failed_group(group_key)


# =============================================================================
# Test runner
# =============================================================================

class TestLinearIfChain(unittest.TestCase):
    """Test linear token handling in if-elif-else chains"""
    
    def test_valid_patterns(self):
        """Test all valid patterns compile and run"""
        # Call each function to trigger compilation and execution
        test_if_elif_else_all_consume(1)
        test_if_elif_elif_else_all_consume(1)
        result = test_if_if_return_pattern(2)
        self.assertEqual(result, 1)
        result = test_if_elif_return_pattern(1)
        self.assertEqual(result, 1)
        test_nested_if_with_linear(1, 1)
        result = test_dispatch_helper_pattern(15)
        self.assertEqual(result, 2)
        test_multiple_linear_tokens_if_elif(1)
    
    def test_error_if_without_else(self):
        """Test that if without else cannot consume token"""
        passed, msg = test_if_without_else_no_return_error()
        self.assertTrue(passed, msg)
    
    def test_error_if_elif_missing_else(self):
        """Test that if-elif without else cannot consume token"""
        passed, msg = test_if_elif_missing_else_error()
        self.assertTrue(passed, msg)
    
    def test_error_if_elif_inconsistent(self):
        """Test error when branches handle token inconsistently"""
        passed, msg = test_if_elif_inconsistent_error()
        self.assertTrue(passed, msg)
    
    def test_error_missing_final_consume(self):
        """Test error when if-if-return pattern is missing final consume"""
        passed, msg = test_if_return_missing_final_consume_error()
        self.assertTrue(passed, msg)


if __name__ == "__main__":
    unittest.main()
