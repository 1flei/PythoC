"""
Tests for CFG linear checker edge cases

These tests are designed to expose potential issues in the CFG linear checker:

1. _get_effective_exit_snapshot "can't resurrect" assumption - may fail with reassignment
2. _check_merge_points and _merge_snapshots duplicate checking - may report errors twice
3. _simulate_block fallback to entry_snapshot - may miss state changes
4. Exit point detection for infinite loops and unreachable code
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import unittest
from pythoc.decorators.compile import compile
from pythoc.builtin_entities import linear, consume, void, i32
from pythoc.std.utility import move
from pythoc.build.output_manager import flush_all_pending_outputs, clear_failed_group
from pythoc.logger import set_raise_on_error

# Enable exception raising for tests
set_raise_on_error(True)


# =============================================================================
# Test 1: Reassignment after consume - tests "can't resurrect" assumption
# =============================================================================

@compile
def test_reassign_after_consume_simple(cond: i32) -> void:
    """Token reassigned after consume - should be valid"""
    t = linear()
    consume(t)
    t = linear()  # Reassign - token is now active again
    consume(t)


@compile
def test_reassign_in_branch(cond: i32) -> void:
    """Token consumed and reassigned in one branch, kept in other"""
    t = linear()
    if cond:
        consume(t)
        t = linear()  # Reassign in then branch
    # else: t is still active
    # At merge: both branches have t active
    consume(t)


@compile
def test_reassign_both_branches(cond: i32) -> void:
    """Token consumed and reassigned in both branches"""
    t = linear()
    if cond:
        consume(t)
        t = linear()
    else:
        consume(t)
        t = linear()
    # At merge: both branches have t active (reassigned)
    consume(t)


@compile
def test_reassign_nested_branches(a: i32, b: i32) -> void:
    """Nested branches with reassignment"""
    t = linear()
    if a:
        if b:
            consume(t)
            t = linear()
        else:
            consume(t)
            t = linear()
    else:
        consume(t)
        t = linear()
    # All paths: consume then reassign -> t is active
    consume(t)


# =============================================================================
# Test 2: Multiple return paths - tests exit point consistency
# =============================================================================

@compile
def test_multiple_returns_all_consume(cond: i32) -> i32:
    """Multiple return paths, all consume token"""
    t = linear()
    if cond == 1:
        consume(t)
        return 1
    elif cond == 2:
        consume(t)
        return 2
    else:
        consume(t)
        return 0


@compile
def test_early_return_after_consume(cond: i32) -> i32:
    """Early return after consuming token"""
    t = linear()
    if cond:
        consume(t)
        return 1
    consume(t)
    return 0


@compile
def test_return_in_nested_if(a: i32, b: i32) -> i32:
    """Return in nested if - multiple exit paths"""
    t = linear()
    if a:
        if b:
            consume(t)
            return 1
        consume(t)
        return 2
    consume(t)
    return 0


# =============================================================================
# Test 3: While True with break - tests infinite loop handling
# =============================================================================

@compile
def test_while_true_with_break(cond: i32) -> void:
    """While True with break - should consume token before break"""
    t = linear()
    while True:
        if cond:
            consume(t)
            break
        # Loop continues - but this is unreachable if cond is always true
        # For linear types, we need to ensure token is consumed on all exit paths


@compile
def test_while_true_consume_then_break() -> void:
    """While True: consume then break - single execution"""
    t = linear()
    while True:
        consume(t)
        break


@compile
def test_while_true_conditional_break(cond: i32) -> void:
    """While True with conditional break"""
    t = linear()
    while True:
        if cond:
            consume(t)
            break
        else:
            consume(t)
            break


# =============================================================================
# Test 4: Match case with multiple patterns - tests merge point handling
# =============================================================================

@compile
def test_match_all_cases_consume(code: i32) -> void:
    """Match where all cases consume token"""
    t = linear()
    match code:
        case 1:
            consume(t)
        case 2:
            consume(t)
        case 3:
            consume(t)
        case _:
            consume(t)


@compile
def test_match_nested_if(code: i32, cond: i32) -> void:
    """Match with nested if in cases"""
    t = linear()
    match code:
        case 1:
            if cond:
                consume(t)
            else:
                consume(t)
        case _:
            consume(t)


# =============================================================================
# Test 5: Complex control flow - tests dataflow propagation
# =============================================================================

@compile
def test_diamond_control_flow(cond: i32) -> void:
    """Diamond-shaped control flow (if-else merge)"""
    t = linear()
    x: i32 = 0
    if cond:
        x = 1
    else:
        x = 2
    # Merge point - t should still be active
    consume(t)


@compile
def test_sequential_ifs(a: i32, b: i32) -> void:
    """Sequential if statements"""
    t = linear()
    if a:
        pass
    if b:
        pass
    consume(t)


@compile
def test_if_chain_all_paths_consume(a: i32, b: i32, c: i32) -> void:
    """If-elif-else chain where all paths consume"""
    t = linear()
    if a:
        consume(t)
    elif b:
        consume(t)
    elif c:
        consume(t)
    else:
        consume(t)


# =============================================================================
# Error cases - should fail to compile
# =============================================================================

def run_error_test_reassign_inconsistent():
    """Error: reassign in one branch only, inconsistent at merge"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_reassign_inconsistent')
    try:
        @compile(suffix="bad_reassign_inconsistent")
        def bad_reassign_inconsistent(cond: i32) -> void:
            t = linear()
            if cond:
                consume(t)
                t = linear()  # t is active
            else:
                pass  # t is still active (original)
            # At merge: both have t active, but...
            consume(t)
            # ERROR: in then branch, we consumed original t
            # This should actually be valid! Both branches end with t active

        flush_all_pending_outputs()
        # This should actually succeed - both branches have t active at merge
        return True, "correctly compiled"
    except RuntimeError as e:
        return False, f"unexpected error: {e}"
    finally:
        clear_failed_group(group_key)


def run_error_test_consume_only_in_then():
    """Error: consume only in then branch, not in else"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_consume_only_then')
    try:
        @compile(suffix="bad_consume_only_then")
        def bad_consume_only_then(cond: i32) -> void:
            t = linear()
            if cond:
                consume(t)  # consumed
            else:
                pass  # still active
            # ERROR: inconsistent at merge

        flush_all_pending_outputs()
        return False, "should have raised RuntimeError"
    except RuntimeError as e:
        err_str = str(e).lower()
        if "inconsistent" in err_str or "not consumed" in err_str:
            return True, str(e)
        return False, f"wrong error: {e}"
    finally:
        clear_failed_group(group_key)


def run_error_test_return_without_consume():
    """Error: return without consuming token"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_return_no_consume')
    try:
        @compile(suffix="bad_return_no_consume")
        def bad_return_no_consume(cond: i32) -> i32:
            t = linear()
            if cond:
                consume(t)
                return 1
            # ERROR: return without consuming t
            return 0

        flush_all_pending_outputs()
        return False, "should have raised RuntimeError"
    except RuntimeError as e:
        err_str = str(e).lower()
        if "not consumed" in err_str or "unconsumed" in err_str:
            return True, str(e)
        return False, f"wrong error: {e}"
    finally:
        clear_failed_group(group_key)


def run_error_test_while_true_no_consume():
    """Error: while True breaks without consuming token"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_while_true_no_consume')
    try:
        @compile(suffix="bad_while_true_no_consume")
        def bad_while_true_no_consume() -> void:
            t = linear()
            while True:
                break  # ERROR: didn't consume t

        flush_all_pending_outputs()
        return False, "should have raised RuntimeError"
    except RuntimeError as e:
        err_str = str(e).lower()
        if "not consumed" in err_str or "unconsumed" in err_str:
            return True, str(e)
        return False, f"wrong error: {e}"
    finally:
        clear_failed_group(group_key)


def run_error_test_match_missing_consume():
    """Error: match case missing consume"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_match_missing')
    try:
        @compile(suffix="bad_match_missing")
        def bad_match_missing(code: i32) -> void:
            t = linear()
            match code:
                case 1:
                    consume(t)
                case _:
                    pass  # ERROR: doesn't consume

        flush_all_pending_outputs()
        return False, "should have raised RuntimeError"
    except RuntimeError as e:
        err_str = str(e).lower()
        if "inconsistent" in err_str or "not consumed" in err_str:
            return True, str(e)
        return False, f"wrong error: {e}"
    finally:
        clear_failed_group(group_key)


# =============================================================================
# Test class using unittest
# =============================================================================

class TestLinearMoreEdgeCases(unittest.TestCase):
    """Test CFG linear checker edge cases"""
    
    # Error tests
    def test_error_reassign_inconsistent(self):
        result, msg = run_error_test_reassign_inconsistent()
        self.assertTrue(result, msg)
    
    def test_error_consume_only_in_then(self):
        result, msg = run_error_test_consume_only_in_then()
        self.assertTrue(result, msg)
    
    def test_error_return_without_consume(self):
        result, msg = run_error_test_return_without_consume()
        self.assertTrue(result, msg)
    
    def test_error_while_true_no_consume(self):
        result, msg = run_error_test_while_true_no_consume()
        self.assertTrue(result, msg)
    
    def test_error_match_missing_consume(self):
        result, msg = run_error_test_match_missing_consume()
        self.assertTrue(result, msg)
    
    # Reassignment tests
    def test_reassign_after_consume_simple(self):
        test_reassign_after_consume_simple(1)
    
    def test_reassign_in_branch(self):
        test_reassign_in_branch(1)
        test_reassign_in_branch(0)
    
    def test_reassign_both_branches(self):
        test_reassign_both_branches(1)
        test_reassign_both_branches(0)
    
    def test_reassign_nested_branches(self):
        test_reassign_nested_branches(1, 1)
        test_reassign_nested_branches(1, 0)
        test_reassign_nested_branches(0, 1)
    
    # Multiple return tests
    def test_multiple_returns_all_consume(self):
        self.assertEqual(test_multiple_returns_all_consume(1), 1)
        self.assertEqual(test_multiple_returns_all_consume(2), 2)
        self.assertEqual(test_multiple_returns_all_consume(0), 0)
    
    def test_early_return_after_consume(self):
        self.assertEqual(test_early_return_after_consume(1), 1)
        self.assertEqual(test_early_return_after_consume(0), 0)
    
    def test_return_in_nested_if(self):
        self.assertEqual(test_return_in_nested_if(1, 1), 1)
        self.assertEqual(test_return_in_nested_if(1, 0), 2)
        self.assertEqual(test_return_in_nested_if(0, 1), 0)
    
    # While True tests
    def test_while_true_with_break(self):
        test_while_true_with_break(1)
    
    def test_while_true_consume_then_break(self):
        test_while_true_consume_then_break()
    
    def test_while_true_conditional_break(self):
        test_while_true_conditional_break(1)
        test_while_true_conditional_break(0)
    
    # Match tests
    def test_match_all_cases_consume(self):
        test_match_all_cases_consume(1)
        test_match_all_cases_consume(2)
        test_match_all_cases_consume(3)
        test_match_all_cases_consume(99)
    
    def test_match_nested_if(self):
        test_match_nested_if(1, 1)
        test_match_nested_if(1, 0)
        test_match_nested_if(99, 1)
    
    # Complex control flow tests
    def test_diamond_control_flow(self):
        test_diamond_control_flow(1)
        test_diamond_control_flow(0)
    
    def test_sequential_ifs(self):
        test_sequential_ifs(1, 1)
        test_sequential_ifs(1, 0)
        test_sequential_ifs(0, 1)
        test_sequential_ifs(0, 0)
    
    def test_if_chain_all_paths_consume(self):
        test_if_chain_all_paths_consume(1, 0, 0)
        test_if_chain_all_paths_consume(0, 1, 0)
        test_if_chain_all_paths_consume(0, 0, 1)
        test_if_chain_all_paths_consume(0, 0, 0)


if __name__ == "__main__":
    unittest.main()
