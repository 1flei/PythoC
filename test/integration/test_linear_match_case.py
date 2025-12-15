"""
Tests for linear token handling in match-case statements

Key rules for linear tokens in match-case:
1. All cases must handle token consistently (all consume or all don't)
2. Wildcard case (_) counts as a branch that must be consistent
3. Match with return in each case: each case can consume and return
4. Match without wildcard: implicit fallthrough path must be considered
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import unittest

from pythoc.decorators.compile import compile
from pythoc.builtin_entities import linear, consume, void, i32, i8
from pythoc.std.utility import move
from pythoc.build.output_manager import flush_all_pending_outputs, clear_failed_group
from pythoc.logger import set_raise_on_error

# Enable exception raising for tests that expect to catch exceptions
set_raise_on_error(True)


# =============================================================================
# Valid patterns - Match-case with linear tokens
# =============================================================================

@compile
def test_match_all_cases_consume(code: i32) -> void:
    """Test match where all cases consume the token"""
    t = linear()
    match code:
        case 1:
            consume(t)
        case 2:
            consume(t)
        case _:
            consume(t)


@compile
def test_match_many_cases_consume(code: i32) -> void:
    """Test match with many cases all consuming"""
    t = linear()
    match code:
        case 1:
            consume(t)
        case 2:
            consume(t)
        case 3:
            consume(t)
        case 4:
            consume(t)
        case 5:
            consume(t)
        case _:
            consume(t)


@compile
def consume_linear_ret(t: linear) -> i32:
    """Helper: consume token and return value"""
    consume(t)
    return 1


@compile
def test_match_with_return(code: i32) -> i32:
    """Test match with return in each case"""
    t = linear()
    match code:
        case 1:
            return consume_linear_ret(move(t))
        case 2:
            return consume_linear_ret(move(t))
        case _:
            return consume_linear_ret(move(t))


@compile
def test_match_or_pattern_consume(code: i32) -> void:
    """Test match with OR patterns consuming token"""
    t = linear()
    match code:
        case 1 | 2 | 3:
            consume(t)
        case 4 | 5:
            consume(t)
        case _:
            consume(t)


@compile
def test_match_nested_consume(x: i32, y: i32) -> void:
    """Test nested match statements with linear token"""
    t = linear()
    match x:
        case 1:
            match y:
                case 1:
                    consume(t)
                case _:
                    consume(t)
        case 2:
            consume(t)
        case _:
            consume(t)


@compile
def test_match_multiple_tokens(code: i32) -> void:
    """Test match with multiple linear tokens"""
    t1 = linear()
    t2 = linear()
    match code:
        case 1:
            consume(t1)
            consume(t2)
        case 2:
            consume(t1)
            consume(t2)
        case _:
            consume(t1)
            consume(t2)


@compile
def test_match_token_not_consumed_in_any(code: i32) -> void:
    """Test match where token is consumed after (not in any case)"""
    t = linear()
    result: i32 = 0
    match code:
        case 1:
            result = 10
        case 2:
            result = 20
        case _:
            result = 0
    # Token consumed after match - all branches consistent (none consume)
    consume(t)


@compile
def _match_dispatch_impl(t: linear, code: i32) -> i32:
    """Helper function for dispatch pattern"""
    match code:
        case 1:
            consume(t)
            return 1
        case 2:
            consume(t)
            return 2
        case _:
            consume(t)
            return 0


@compile
def test_match_dispatch_pattern(cond: i32) -> i32:
    """Test dispatch pattern: create token, compute code, dispatch"""
    t = linear()
    code: i32 = 0
    if cond > 10:
        code = 2
    elif cond > 5:
        code = 1
    return _match_dispatch_impl(move(t), code)


# =============================================================================
# Error cases - these should fail to compile
# =============================================================================

def test_match_missing_wildcard_error():
    """Test error when match lacks wildcard and consumes token"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_match_no_wildcard')
    try:
        @compile(suffix="bad_match_no_wildcard")
        def bad_match_no_wildcard(code: i32) -> void:
            t = linear()
            match code:
                case 1:
                    consume(t)
                case 2:
                    consume(t)
                # ERROR: no wildcard case - what if code is 3?

        flush_all_pending_outputs()
        return False, "should have raised RuntimeError"
    except RuntimeError as e:
        return True, str(e)
    finally:
        clear_failed_group(group_key)


def test_match_inconsistent_cases_error():
    """Test error when match cases handle token inconsistently"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_match_inconsistent')
    try:
        @compile(suffix="bad_match_inconsistent")
        def bad_match_inconsistent(code: i32) -> void:
            t = linear()
            match code:
                case 1:
                    consume(t)
                case 2:
                    pass  # ERROR: doesn't consume
                case _:
                    consume(t)

        flush_all_pending_outputs()
        return False, "should have raised RuntimeError"
    except RuntimeError as e:
        if "consistently" in str(e).lower():
            return True, str(e)
        return False, f"wrong error: {e}"
    finally:
        clear_failed_group(group_key)


def test_match_wildcard_inconsistent_error():
    """Test error when wildcard case is inconsistent"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_match_wildcard')
    try:
        @compile(suffix="bad_match_wildcard")
        def bad_match_wildcard(code: i32) -> void:
            t = linear()
            match code:
                case 1:
                    consume(t)
                case 2:
                    consume(t)
                case _:
                    pass  # ERROR: wildcard doesn't consume

        flush_all_pending_outputs()
        return False, "should have raised RuntimeError"
    except RuntimeError as e:
        if "consistently" in str(e).lower():
            return True, str(e)
        return False, f"wrong error: {e}"
    finally:
        clear_failed_group(group_key)


def test_match_nested_inconsistent_error():
    """Test error when nested match is inconsistent"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_nested_match')
    try:
        @compile(suffix="bad_nested_match")
        def bad_nested_match(x: i32, y: i32) -> void:
            t = linear()
            match x:
                case 1:
                    match y:
                        case 1:
                            consume(t)
                        case _:
                            pass  # ERROR: inner match inconsistent
                case _:
                    consume(t)

        flush_all_pending_outputs()
        return False, "should have raised RuntimeError"
    except RuntimeError as e:
        if "consistently" in str(e).lower():
            return True, str(e)
        return False, f"wrong error: {e}"
    finally:
        clear_failed_group(group_key)


# =============================================================================
# Test runner
# =============================================================================

class TestLinearMatchCase(unittest.TestCase):
    """Test linear token handling in match-case statements"""

    def test_valid_all_cases_consume(self):
        """Test match where all cases consume"""
        test_match_all_cases_consume(1)
        test_match_all_cases_consume(2)
        test_match_all_cases_consume(99)

    def test_valid_many_cases(self):
        """Test match with many cases"""
        test_match_many_cases_consume(1)
        test_match_many_cases_consume(5)
        test_match_many_cases_consume(100)

    def test_valid_with_return(self):
        """Test match with return in each case"""
        self.assertEqual(test_match_with_return(1), 1)
        self.assertEqual(test_match_with_return(2), 1)
        self.assertEqual(test_match_with_return(99), 1)

    def test_valid_or_pattern(self):
        """Test match with OR patterns"""
        test_match_or_pattern_consume(1)
        test_match_or_pattern_consume(4)
        test_match_or_pattern_consume(99)

    def test_valid_nested(self):
        """Test nested match statements"""
        test_match_nested_consume(1, 1)
        test_match_nested_consume(1, 99)
        test_match_nested_consume(2, 1)
        test_match_nested_consume(99, 1)

    def test_valid_multiple_tokens(self):
        """Test multiple linear tokens in match"""
        test_match_multiple_tokens(1)
        test_match_multiple_tokens(2)
        test_match_multiple_tokens(99)

    def test_valid_token_after_match(self):
        """Test token consumed after match"""
        test_match_token_not_consumed_in_any(1)
        test_match_token_not_consumed_in_any(2)
        test_match_token_not_consumed_in_any(99)

    def test_valid_dispatch_pattern(self):
        """Test dispatch pattern"""
        self.assertEqual(test_match_dispatch_pattern(15), 2)
        self.assertEqual(test_match_dispatch_pattern(7), 1)
        self.assertEqual(test_match_dispatch_pattern(3), 0)

    def test_error_missing_wildcard(self):
        """Test error when match lacks wildcard"""
        passed, msg = test_match_missing_wildcard_error()
        self.assertTrue(passed, msg)

    def test_error_inconsistent_cases(self):
        """Test error when cases are inconsistent"""
        passed, msg = test_match_inconsistent_cases_error()
        self.assertTrue(passed, msg)

    def test_error_wildcard_inconsistent(self):
        """Test error when wildcard is inconsistent"""
        passed, msg = test_match_wildcard_inconsistent_error()
        self.assertTrue(passed, msg)

    def test_error_nested_inconsistent(self):
        """Test error when nested match is inconsistent"""
        passed, msg = test_match_nested_inconsistent_error()
        self.assertTrue(passed, msg)


if __name__ == "__main__":
    unittest.main()
