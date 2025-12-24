"""
Comprehensive tests for CFG linear checker with complex control flow

This module tests:
1. Deep nesting (if/match)
2. Multiple tokens
3. Reassignment patterns
4. if + match combinations
5. if + loop combinations
6. match + loop combinations
7. All three combined (if + match + loop)
8. While True patterns
9. Early return patterns
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import unittest

from pythoc.decorators.compile import compile
from pythoc.builtin_entities import linear, consume, void, i32
from pythoc.std.utility import move
from pythoc import seq

from test.utils.test_utils import DeferredTestCase, expect_error


# =============================================================================
# Deep nesting tests
# =============================================================================

@compile
def test_deeply_nested_if(a: i32, b: i32, c: i32, d: i32) -> void:
    """4-level nested if, all paths consume"""
    t = linear()
    if a:
        if b:
            if c:
                if d:
                    consume(t)
                else:
                    consume(t)
            else:
                consume(t)
        else:
            consume(t)
    else:
        consume(t)


@compile
def test_deeply_nested_consume_at_end(a: i32, b: i32, c: i32, d: i32) -> void:
    """4-level nested if, consume at the end"""
    t = linear()
    x: i32 = 0
    if a:
        x = x + 1
        if b:
            x = x + 2
            if c:
                x = x + 4
                if d:
                    x = x + 8
    consume(t)


# =============================================================================
# Multiple tokens tests
# =============================================================================

@compile
def test_two_tokens_parallel(cond: i32) -> void:
    """Two tokens, both consumed in both branches"""
    t1 = linear()
    t2 = linear()
    if cond:
        consume(t1)
        consume(t2)
    else:
        consume(t1)
        consume(t2)


@compile
def test_two_tokens_interleaved(cond: i32) -> void:
    """Two tokens, interleaved consumption"""
    t1 = linear()
    t2 = linear()
    if cond:
        consume(t1)
    else:
        consume(t1)
    if cond:
        consume(t2)
    else:
        consume(t2)


@compile
def test_three_tokens_complex(a: i32, b: i32) -> void:
    """Three tokens with complex control flow"""
    t1 = linear()
    t2 = linear()
    t3 = linear()
    if a:
        consume(t1)
        if b:
            consume(t2)
            consume(t3)
        else:
            consume(t2)
            consume(t3)
    else:
        consume(t1)
        consume(t2)
        consume(t3)


@compile
def test_multiple_tokens_if_match(cond: i32, code: i32) -> void:
    """Multiple tokens with if-match combination"""
    t1 = linear()
    t2 = linear()
    if cond > 0:
        match code:
            case 1:
                consume(t1)
                consume(t2)
            case _:
                consume(t1)
                consume(t2)
    else:
        consume(t1)
        consume(t2)


# =============================================================================
# Reassignment tests
# =============================================================================

@compile
def test_reassign_multiple_times(cond: i32) -> void:
    """Reassign token multiple times"""
    t = linear()
    consume(t)
    t = linear()
    consume(t)
    t = linear()
    consume(t)


@compile
def test_reassign_in_sequence_of_ifs(a: i32, b: i32, c: i32) -> void:
    """Reassign through sequence of ifs"""
    t = linear()
    if a:
        consume(t)
        t = linear()
    if b:
        consume(t)
        t = linear()
    if c:
        consume(t)
        t = linear()
    consume(t)


@compile
def test_reassign_alternating(cond: i32) -> void:
    """Alternating consume and reassign"""
    t = linear()
    if cond:
        consume(t)
        t = linear()
        consume(t)
        t = linear()
    else:
        consume(t)
        t = linear()
    consume(t)


@compile
def test_diamond_with_reassign(cond: i32) -> void:
    """Diamond control flow with reassignment"""
    t = linear()
    x: i32 = 0
    if cond:
        x = 1
        consume(t)
        t = linear()
    else:
        x = 2
        consume(t)
        t = linear()
    # Both branches reassign t to active
    consume(t)


# =============================================================================
# if + match combinations
# =============================================================================

@compile
def test_if_then_match(cond: i32, code: i32) -> void:
    """If condition, then match in branches"""
    t = linear()
    if cond > 0:
        match code:
            case 1:
                consume(t)
            case _:
                consume(t)
    else:
        consume(t)


@compile
def test_match_then_if(code: i32, cond: i32) -> void:
    """Match first, then if in cases"""
    t = linear()
    match code:
        case 1:
            if cond > 0:
                consume(t)
            else:
                consume(t)
        case 2:
            consume(t)
        case _:
            consume(t)


@compile
def test_if_elif_with_match(cond: i32, code: i32) -> void:
    """If-elif-else with match in one branch"""
    t = linear()
    if cond == 1:
        consume(t)
    elif cond == 2:
        match code:
            case 1:
                consume(t)
            case _:
                consume(t)
    else:
        consume(t)


@compile
def test_match_with_nested_if(code: i32, a: i32, b: i32) -> void:
    """Match with nested if-else in cases"""
    t = linear()
    match code:
        case 1:
            if a > 0:
                if b > 0:
                    consume(t)
                else:
                    consume(t)
            else:
                consume(t)
        case _:
            consume(t)


@compile
def test_match_many_cases(code: i32) -> void:
    """Match with many cases"""
    t = linear()
    match code:
        case 0:
            consume(t)
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
def test_match_nested_match(outer: i32, inner: i32) -> void:
    """Nested match statements"""
    t = linear()
    match outer:
        case 0:
            match inner:
                case 0:
                    consume(t)
                case _:
                    consume(t)
        case 1:
            match inner:
                case 0:
                    consume(t)
                case 1:
                    consume(t)
                case _:
                    consume(t)
        case _:
            consume(t)


# =============================================================================
# if + loop combinations
# =============================================================================

@compile
def test_if_then_loop_consume_after(cond: i32, n: i32) -> void:
    """If-else with loop, token consumed after"""
    t = linear()
    if cond > 0:
        for i in seq(n):
            pass
    else:
        pass
    consume(t)


@compile
def test_loop_then_if_consume(n: i32, cond: i32) -> void:
    """Loop first, then if-else to consume"""
    t = linear()
    sum: i32 = 0
    for i in seq(n):
        sum = sum + i
    if cond > 0:
        consume(t)
    else:
        consume(t)


@compile
def test_if_with_loop_token_each_iteration(cond: i32, n: i32) -> i32:
    """If-else with loop creating token each iteration"""
    sum: i32 = 0
    if cond > 0:
        for i in seq(n):
            t = linear()
            sum = sum + i
            consume(t)
    else:
        for i in seq(n):
            t = linear()
            sum = sum + i * 2
            consume(t)
    return sum


@compile
def test_loop_with_if_token_each_iteration(n: i32) -> i32:
    """Loop with if-else, token created and consumed each iteration"""
    sum: i32 = 0
    for i in seq(n):
        t = linear()
        if i % 2 == 0:
            sum = sum + i
            consume(t)
        else:
            sum = sum + i * 2
            consume(t)
    return sum


# =============================================================================
# match + loop combinations
# =============================================================================

@compile
def test_match_then_loop_consume_after(code: i32, n: i32) -> void:
    """Match with loop in case, token consumed after"""
    t = linear()
    match code:
        case 1:
            for i in seq(n):
                pass
        case _:
            pass
    consume(t)


@compile
def test_loop_then_match_consume(n: i32, code: i32) -> void:
    """Loop first, then match to consume"""
    t = linear()
    sum: i32 = 0
    for i in seq(n):
        sum = sum + i
    match code:
        case 1:
            consume(t)
        case 2:
            consume(t)
        case _:
            consume(t)


@compile
def test_match_with_loop_token_each_iteration(code: i32, n: i32) -> i32:
    """Match with loop creating token each iteration"""
    sum: i32 = 0
    match code:
        case 1:
            for i in seq(n):
                t = linear()
                sum = sum + i
                consume(t)
        case _:
            for i in seq(n):
                t = linear()
                sum = sum + i * 2
                consume(t)
    return sum


@compile
def test_loop_with_match_token_each_iteration(n: i32) -> i32:
    """Loop with match, token created and consumed each iteration"""
    sum: i32 = 0
    for i in seq(n):
        t = linear()
        match i:
            case 0:
                sum = sum + 100
                consume(t)
            case 1:
                sum = sum + 200
                consume(t)
            case _:
                sum = sum + i
                consume(t)
    return sum


# =============================================================================
# if + match + loop combinations
# =============================================================================

@compile
def test_if_match_loop_all_combined(cond: i32, code: i32, n: i32) -> i32:
    """All three combined: if -> match -> loop"""
    t = linear()
    sum: i32 = 0
    if cond > 0:
        match code:
            case 1:
                for i in seq(n):
                    sum = sum + i
            case _:
                for i in seq(n):
                    sum = sum + i * 2
    else:
        for i in seq(n):
            sum = sum + i * 3
    consume(t)
    return sum


@compile
def test_loop_if_match_token_each_iteration(n: i32) -> i32:
    """Loop -> if -> match, token each iteration"""
    sum: i32 = 0
    for i in seq(n):
        t = linear()
        if i < 3:
            match i:
                case 0:
                    sum = sum + 100
                    consume(t)
                case 1:
                    sum = sum + 200
                    consume(t)
                case _:
                    sum = sum + 300
                    consume(t)
        else:
            consume(t)
            sum = sum + i
    return sum


@compile
def consume_ret_i32(t: linear, val: i32) -> i32:
    """Helper: consume token and return value"""
    consume(t)
    return val


@compile
def test_complex_dispatch_pattern(mode: i32, submode: i32) -> i32:
    """Complex dispatch: if-elif determines mode, match handles submode"""
    t = linear()
    code: i32 = 0
    if mode == 1:
        code = 10
    elif mode == 2:
        code = 20
    else:
        code = 0

    match submode:
        case 1:
            return consume_ret_i32(move(t), code + 1)
        case 2:
            return consume_ret_i32(move(t), code + 2)
        case _:
            return consume_ret_i32(move(t), code)


@compile
def test_nested_all_three(a: i32, b: i32, c: i32) -> void:
    """Deeply nested: if -> match -> if -> loop"""
    t = linear()
    if a > 0:
        match b:
            case 1:
                if c > 0:
                    for i in seq(3):
                        pass
                    consume(t)
                else:
                    consume(t)
            case _:
                consume(t)
    else:
        consume(t)


# =============================================================================
# While True tests
# =============================================================================

@compile
def test_while_true_nested_if_break(a: i32, b: i32) -> void:
    """While True with nested if and break"""
    t = linear()
    while True:
        if a:
            if b:
                consume(t)
                break
            else:
                consume(t)
                break
        else:
            consume(t)
            break


@compile
def test_while_true_match_break(code: i32) -> void:
    """While True with match and break"""
    t = linear()
    while True:
        match code:
            case 0:
                consume(t)
                break
            case 1:
                consume(t)
                break
            case _:
                consume(t)
                break


@compile
def test_if_match_while_combination(cond: i32, code: i32) -> void:
    """Combination of if, match, and while True"""
    t = linear()
    if cond:
        match code:
            case 0:
                while True:
                    consume(t)
                    break
            case _:
                consume(t)
    else:
        consume(t)


# =============================================================================
# Early return tests
# =============================================================================

@compile
def test_early_return_many_paths(a: i32, b: i32, c: i32) -> i32:
    """Many early return paths"""
    t = linear()
    if a == 1:
        consume(t)
        return 1
    if b == 1:
        consume(t)
        return 2
    if c == 1:
        consume(t)
        return 3
    consume(t)
    return 0


# =============================================================================
# Error test helpers (using expect_error decorator)
# =============================================================================

@expect_error(["inconsistent", "not consumed"], suffix="bad_deep_missing")
def run_error_test_deeply_nested_missing():
    """Error: deeply nested missing consume"""
    @compile(suffix="bad_deep_missing")
    def bad_deep_missing(a: i32, b: i32, c: i32) -> void:
        t = linear()
        if a:
            if b:
                if c:
                    consume(t)
                else:
                    pass  # ERROR: missing consume
            else:
                consume(t)
        else:
            consume(t)


@expect_error(["inconsistent", "not consumed"], suffix="bad_two_tokens")
def run_error_test_two_tokens_one_missing():
    """Error: two tokens, one missing consume"""
    @compile(suffix="bad_two_tokens")
    def bad_two_tokens(cond: i32) -> void:
        t1 = linear()
        t2 = linear()
        if cond:
            consume(t1)
            consume(t2)
        else:
            consume(t1)
            # ERROR: t2 not consumed


@expect_error(["inconsistent", "not consumed"], suffix="bad_reassign_one")
def run_error_test_reassign_only_one_branch():
    """Error: reassign only in one branch, other path has different state"""
    @compile(suffix="bad_reassign_one")
    def bad_reassign_one(cond: i32) -> void:
        t = linear()
        if cond:
            consume(t)
            t = linear()
            consume(t)
        else:
            # t still active, not consumed
            pass
        # ERROR: inconsistent - then has t consumed, else has t active


@expect_error(["consistent", "inconsistent"], suffix="bad_if_match")
def run_error_test_if_match_inconsistent():
    """Error: if-match combination is inconsistent"""
    @compile(suffix="bad_if_match")
    def bad_if_match(cond: i32, code: i32) -> void:
        t = linear()
        if cond > 0:
            match code:
                case 1:
                    consume(t)
                case _:
                    pass  # ERROR: match inconsistent
        else:
            consume(t)


@expect_error(["inconsistent", "not consumed", "error"], suffix="bad_match_if")
def run_error_test_match_if_inconsistent():
    """Error: match-if combination is inconsistent"""
    @compile(suffix="bad_match_if")
    def bad_match_if(code: i32, cond: i32) -> void:
        t = linear()
        match code:
            case 1:
                if cond > 0:
                    consume(t)
                # ERROR: if without else
            case _:
                consume(t)


@expect_error(["inconsistent", "not consumed", "error"], suffix="bad_loop_if")
def run_error_test_loop_if_consume():
    """Error: consuming in loop with if"""
    @compile(suffix="bad_loop_if")
    def bad_loop_if(n: i32) -> void:
        t = linear()
        for i in seq(n):
            if i == 0:
                consume(t)  # ERROR: consumed in loop


@expect_error(["inconsistent", "not consumed", "error"], suffix="bad_loop_match")
def run_error_test_loop_match_consume():
    """Error: consuming in loop with match"""
    @compile(suffix="bad_loop_match")
    def bad_loop_match(n: i32) -> void:
        t = linear()
        for i in seq(n):
            match i:
                case 0:
                    consume(t)  # ERROR: consumed in loop
                case _:
                    pass


@expect_error(["inconsistent", "not consumed"], suffix="bad_complex")
def run_error_test_complex_missing_consume():
    """Error: complex flow missing consume"""
    @compile(suffix="bad_complex")
    def bad_complex(cond: i32, code: i32) -> void:
        t = linear()
        if cond > 0:
            match code:
                case 1:
                    consume(t)
                case _:
                    consume(t)
        # ERROR: else branch missing, token not consumed


# =============================================================================
# Test class
# =============================================================================

class TestLinearComplex(DeferredTestCase):
    """Comprehensive tests for CFG linear checker with complex control flow"""

    # --- Deep nesting tests ---
    def test_deeply_nested_if(self):
        test_deeply_nested_if(1, 1, 1, 1)
        test_deeply_nested_if(1, 1, 1, 0)
        test_deeply_nested_if(1, 1, 0, 0)
        test_deeply_nested_if(1, 0, 0, 0)
        test_deeply_nested_if(0, 0, 0, 0)

    def test_deeply_nested_consume_at_end(self):
        test_deeply_nested_consume_at_end(1, 1, 1, 1)
        test_deeply_nested_consume_at_end(0, 0, 0, 0)

    # --- Multiple tokens tests ---
    def test_two_tokens_parallel(self):
        test_two_tokens_parallel(1)
        test_two_tokens_parallel(0)

    def test_two_tokens_interleaved(self):
        test_two_tokens_interleaved(1)
        test_two_tokens_interleaved(0)

    def test_three_tokens_complex(self):
        test_three_tokens_complex(1, 1)
        test_three_tokens_complex(1, 0)
        test_three_tokens_complex(0, 1)

    def test_multiple_tokens_if_match(self):
        test_multiple_tokens_if_match(1, 1)
        test_multiple_tokens_if_match(1, 99)
        test_multiple_tokens_if_match(0, 1)

    # --- Reassignment tests ---
    def test_reassign_multiple_times(self):
        test_reassign_multiple_times(1)

    def test_reassign_in_sequence_of_ifs(self):
        test_reassign_in_sequence_of_ifs(1, 1, 1)
        test_reassign_in_sequence_of_ifs(0, 0, 0)
        test_reassign_in_sequence_of_ifs(1, 0, 1)

    def test_reassign_alternating(self):
        test_reassign_alternating(1)
        test_reassign_alternating(0)

    def test_diamond_with_reassign(self):
        test_diamond_with_reassign(1)
        test_diamond_with_reassign(0)

    # --- if + match tests ---
    def test_if_then_match(self):
        test_if_then_match(1, 1)
        test_if_then_match(1, 99)
        test_if_then_match(0, 1)

    def test_match_then_if(self):
        test_match_then_if(1, 1)
        test_match_then_if(1, 0)
        test_match_then_if(2, 1)
        test_match_then_if(99, 1)

    def test_if_elif_with_match(self):
        test_if_elif_with_match(1, 1)
        test_if_elif_with_match(2, 1)
        test_if_elif_with_match(2, 99)
        test_if_elif_with_match(99, 1)

    def test_match_with_nested_if(self):
        test_match_with_nested_if(1, 1, 1)
        test_match_with_nested_if(1, 1, 0)
        test_match_with_nested_if(1, 0, 1)
        test_match_with_nested_if(99, 1, 1)

    def test_match_many_cases(self):
        for i in range(8):
            test_match_many_cases(i)

    def test_match_nested_match(self):
        test_match_nested_match(0, 0)
        test_match_nested_match(0, 1)
        test_match_nested_match(1, 0)
        test_match_nested_match(1, 1)
        test_match_nested_match(1, 99)
        test_match_nested_match(99, 0)

    # --- if + loop tests ---
    def test_if_then_loop_consume_after(self):
        test_if_then_loop_consume_after(1, 5)
        test_if_then_loop_consume_after(0, 5)

    def test_loop_then_if_consume(self):
        test_loop_then_if_consume(5, 1)
        test_loop_then_if_consume(5, 0)

    def test_if_with_loop_token_each_iteration(self):
        result = test_if_with_loop_token_each_iteration(1, 5)
        self.assertEqual(result, 10)  # 0+1+2+3+4
        result = test_if_with_loop_token_each_iteration(0, 5)
        self.assertEqual(result, 20)  # (0+1+2+3+4)*2

    def test_loop_with_if_token_each_iteration(self):
        result = test_loop_with_if_token_each_iteration(5)
        # i=0: 0, i=1: 2, i=2: 2, i=3: 6, i=4: 4 -> 0+2+2+6+4=14
        self.assertEqual(result, 14)

    # --- match + loop tests ---
    def test_match_then_loop_consume_after(self):
        test_match_then_loop_consume_after(1, 5)
        test_match_then_loop_consume_after(99, 5)

    def test_loop_then_match_consume(self):
        test_loop_then_match_consume(5, 1)
        test_loop_then_match_consume(5, 2)
        test_loop_then_match_consume(5, 99)

    def test_match_with_loop_token_each_iteration(self):
        result = test_match_with_loop_token_each_iteration(1, 5)
        self.assertEqual(result, 10)  # 0+1+2+3+4
        result = test_match_with_loop_token_each_iteration(99, 5)
        self.assertEqual(result, 20)  # (0+1+2+3+4)*2

    def test_loop_with_match_token_each_iteration(self):
        result = test_loop_with_match_token_each_iteration(5)
        # i=0: 100, i=1: 200, i=2: 2, i=3: 3, i=4: 4 -> 100+200+2+3+4=309
        self.assertEqual(result, 309)

    # --- All three combined tests ---
    def test_if_match_loop_all_combined(self):
        result = test_if_match_loop_all_combined(1, 1, 5)
        self.assertEqual(result, 10)  # 0+1+2+3+4
        result = test_if_match_loop_all_combined(1, 99, 5)
        self.assertEqual(result, 20)  # (0+1+2+3+4)*2
        result = test_if_match_loop_all_combined(0, 1, 5)
        self.assertEqual(result, 30)  # (0+1+2+3+4)*3

    def test_loop_if_match_token_each_iteration(self):
        result = test_loop_if_match_token_each_iteration(5)
        # i=0: 100, i=1: 200, i=2: 300, i=3: 3, i=4: 4 -> 607
        self.assertEqual(result, 607)

    def test_complex_dispatch_pattern(self):
        self.assertEqual(test_complex_dispatch_pattern(1, 1), 11)
        self.assertEqual(test_complex_dispatch_pattern(1, 2), 12)
        self.assertEqual(test_complex_dispatch_pattern(2, 1), 21)
        self.assertEqual(test_complex_dispatch_pattern(99, 99), 0)

    def test_nested_all_three(self):
        test_nested_all_three(1, 1, 1)
        test_nested_all_three(1, 1, 0)
        test_nested_all_three(1, 99, 1)
        test_nested_all_three(0, 1, 1)

    # --- While True tests ---
    def test_while_true_nested_if_break(self):
        test_while_true_nested_if_break(1, 1)
        test_while_true_nested_if_break(1, 0)
        test_while_true_nested_if_break(0, 1)

    def test_while_true_match_break(self):
        test_while_true_match_break(0)
        test_while_true_match_break(1)
        test_while_true_match_break(99)

    def test_if_match_while_combination(self):
        test_if_match_while_combination(1, 0)
        test_if_match_while_combination(1, 1)
        test_if_match_while_combination(0, 0)

    # --- Early return tests ---
    def test_early_return_many_paths(self):
        self.assertEqual(test_early_return_many_paths(1, 0, 0), 1)
        self.assertEqual(test_early_return_many_paths(0, 1, 0), 2)
        self.assertEqual(test_early_return_many_paths(0, 0, 1), 3)
        self.assertEqual(test_early_return_many_paths(0, 0, 0), 0)

    # --- Error tests ---
    def test_error_deeply_nested_missing(self):
        passed, msg = run_error_test_deeply_nested_missing()
        self.assertTrue(passed, msg)

    def test_error_two_tokens_one_missing(self):
        passed, msg = run_error_test_two_tokens_one_missing()
        self.assertTrue(passed, msg)

    def test_error_reassign_only_one_branch(self):
        passed, msg = run_error_test_reassign_only_one_branch()
        self.assertTrue(passed, msg)

    def test_error_if_match_inconsistent(self):
        passed, msg = run_error_test_if_match_inconsistent()
        self.assertTrue(passed, msg)

    def test_error_match_if_inconsistent(self):
        passed, msg = run_error_test_match_if_inconsistent()
        self.assertTrue(passed, msg)

    def test_error_loop_if_consume(self):
        passed, msg = run_error_test_loop_if_consume()
        self.assertTrue(passed, msg)

    def test_error_loop_match_consume(self):
        passed, msg = run_error_test_loop_match_consume()
        self.assertTrue(passed, msg)

    def test_error_complex_missing_consume(self):
        passed, msg = run_error_test_complex_missing_consume()
        self.assertTrue(passed, msg)


if __name__ == "__main__":
    unittest.main()
