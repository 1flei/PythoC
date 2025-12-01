#!/usr/bin/env python3
"""
Comprehensive tests for control flow including complex nesting,
edge cases, and various control flow patterns.
"""

import unittest
from pythoc import i32, i64, bool, compile, seq


# =============================================================================
# Complex If-Else Nesting
# =============================================================================

@compile
def test_deeply_nested_if() -> i32:
    """Test deeply nested if statements (5 levels)"""
    a: i32 = 1
    b: i32 = 2
    c: i32 = 3
    d: i32 = 4
    e: i32 = 5
    result: i32 = 0
    
    if a == 1:
        if b == 2:
            if c == 3:
                if d == 4:
                    if e == 5:
                        result = 100
                    else:
                        result = 50
                else:
                    result = 40
            else:
                result = 30
        else:
            result = 20
    else:
        result = 10
    
    return result  # 100


@compile
def test_if_elif_chain() -> i32:
    """Test long if-elif chain"""
    x: i32 = 7
    result: i32 = 0
    
    if x == 1:
        result = 10
    elif x == 2:
        result = 20
    elif x == 3:
        result = 30
    elif x == 4:
        result = 40
    elif x == 5:
        result = 50
    elif x == 6:
        result = 60
    elif x == 7:
        result = 70
    elif x == 8:
        result = 80
    elif x == 9:
        result = 90
    else:
        result = 0
    
    return result  # 70


@compile
def test_if_with_complex_conditions() -> i32:
    """Test if with complex boolean conditions"""
    a: i32 = 10
    b: i32 = 20
    c: i32 = 30
    result: i32 = 0
    
    if a < b and b < c:
        result = result + 1
    
    if a < b or c < b:
        result = result + 2
    
    if not (a > b):
        result = result + 4
    
    if (a < b and b < c) or (c < a):
        result = result + 8
    
    if a < b < c:  # Chained comparison
        result = result + 16
    
    return result  # 1 + 2 + 4 + 8 + 16 = 31


@compile
def test_if_all_paths() -> i32:
    """Test all paths through if-else"""
    result: i32 = 0
    
    # Path 1: true branch
    if True:
        result = result + 1
    else:
        result = result + 100
    
    # Path 2: false branch
    if False:
        result = result + 100
    else:
        result = result + 2
    
    # Path 3: no else
    if True:
        result = result + 4
    
    return result  # 1 + 2 + 4 = 7


@compile
def test_if_empty_branches() -> i32:
    """Test if with minimal work in branches"""
    x: i32 = 10
    result: i32 = 0
    
    if x > 5:
        result = 1
    
    if x < 5:
        result = 2
    
    return result  # 1


# =============================================================================
# While Loop Edge Cases
# =============================================================================

@compile
def test_while_zero_iterations() -> i32:
    """Test while loop with zero iterations"""
    count: i32 = 0
    i: i32 = 10
    
    while i < 10:
        count = count + 1
        i = i + 1
    
    return count  # 0


@compile
def test_while_one_iteration() -> i32:
    """Test while loop with exactly one iteration"""
    count: i32 = 0
    i: i32 = 0
    
    while i < 1:
        count = count + 1
        i = i + 1
    
    return count  # 1


@compile
def test_while_large_iterations() -> i32:
    """Test while loop with many iterations"""
    sum: i32 = 0
    i: i32 = 0
    
    while i < 1000:
        sum = sum + i
        i = i + 1
    
    return sum  # 499500


@compile
def test_while_nested_deep() -> i32:
    """Test deeply nested while loops (4 levels)"""
    count: i32 = 0
    i: i32 = 0
    
    while i < 3:
        j: i32 = 0
        while j < 3:
            k: i32 = 0
            while k < 3:
                l: i32 = 0
                while l < 3:
                    count = count + 1
                    l = l + 1
                k = k + 1
            j = j + 1
        i = i + 1
    
    return count  # 3^4 = 81


@compile
def test_while_with_multiple_conditions() -> i32:
    """Test while with complex condition"""
    x: i32 = 0
    y: i32 = 100
    count: i32 = 0
    
    while x < 10 and y > 50:
        x = x + 1
        y = y - 5
        count = count + 1
    
    return count  # Stops when y <= 50 (after 10 iterations)


@compile
def test_while_condition_change() -> i32:
    """Test while where condition variable changes in complex ways"""
    x: i32 = 0
    count: i32 = 0
    
    while x < 100:
        if x < 50:
            x = x + 1
        else:
            x = x + 10
        count = count + 1
    
    return count  # 50 + 5 = 55


# =============================================================================
# Break Statement Edge Cases
# =============================================================================

@compile
def test_break_first_iteration() -> i32:
    """Test break on first iteration"""
    count: i32 = 0
    i: i32 = 0
    
    while i < 100:
        count = count + 1
        break
        i = i + 1
    
    return count  # 1


@compile
def test_break_conditional() -> i32:
    """Test conditional break"""
    sum: i32 = 0
    i: i32 = 0
    
    while i < 100:
        sum = sum + i
        if sum > 50:
            break
        i = i + 1
    
    return sum  # 55 (0+1+...+10)


@compile
def test_break_nested_inner() -> i32:
    """Test break only breaks inner loop"""
    outer_count: i32 = 0
    inner_count: i32 = 0
    i: i32 = 0
    
    while i < 5:
        outer_count = outer_count + 1
        j: i32 = 0
        while j < 100:
            inner_count = inner_count + 1
            if j >= 2:
                break
            j = j + 1
        i = i + 1
    
    return outer_count * 100 + inner_count  # 5*100 + 15 = 515


@compile
def test_break_in_deeply_nested() -> i32:
    """Test break in deeply nested structure"""
    result: i32 = 0
    i: i32 = 0
    
    while i < 10:
        j: i32 = 0
        while j < 10:
            k: i32 = 0
            while k < 10:
                result = result + 1
                if k == 2:
                    break
                k = k + 1
            if j == 3:
                break
            j = j + 1
        if i == 4:
            break
        i = i + 1
    
    # Each outer iteration: 4 inner j iterations, each with 3 k iterations
    # Total: 5 * 4 * 3 = 60
    return result


@compile
def test_break_after_work() -> i32:
    """Test that work before break is done"""
    result: i32 = 0
    i: i32 = 0
    
    while i < 10:
        result = result + 10
        if i == 5:
            result = result + 100  # This should execute
            break
        i = i + 1
    
    return result  # 6*10 + 100 = 160


# =============================================================================
# Continue Statement Edge Cases
# =============================================================================

@compile
def test_continue_skip_all() -> i32:
    """Test continue that skips all remaining work"""
    sum: i32 = 0
    i: i32 = 0
    
    while i < 10:
        i = i + 1
        if i % 2 == 0:
            continue
        sum = sum + i  # Only odd numbers
    
    return sum  # 1+3+5+7+9 = 25


@compile
def test_continue_in_nested() -> i32:
    """Test continue in nested loop"""
    sum: i32 = 0
    i: i32 = 0
    
    while i < 5:
        j: i32 = 0
        while j < 5:
            j = j + 1
            if j == 3:
                continue
            sum = sum + 1
        i = i + 1
    
    return sum  # 5 * 4 = 20 (skip j=3 each time)


@compile
def test_continue_with_complex_condition() -> i32:
    """Test continue with complex skip condition"""
    sum: i32 = 0
    i: i32 = 0
    
    while i < 20:
        i = i + 1
        # Skip multiples of 3 or 5
        if i % 3 == 0 or i % 5 == 0:
            continue
        sum = sum + i
    
    # Numbers not divisible by 3 or 5: 1,2,4,7,8,11,13,14,16,17,19
    return sum  # 112


@compile
def test_break_and_continue_together() -> i32:
    """Test break and continue in same loop"""
    sum: i32 = 0
    i: i32 = 0
    
    while i < 100:
        i = i + 1
        if i % 2 == 0:
            continue  # Skip even numbers
        if i > 10:
            break  # Stop after 10
        sum = sum + i  # Only odd numbers <= 10
    
    return sum  # 1+3+5+7+9 = 25


# =============================================================================
# For Loop with Range
# =============================================================================

@compile
def test_for_basic() -> i32:
    """Test basic for loop with seq"""
    sum: i32 = 0
    for i in seq(10):
        sum = sum + i
    return sum  # 0+1+...+9 = 45


@compile
def test_for_with_start() -> i32:
    """Test for loop with start value"""
    sum: i32 = 0
    for i in seq(5, 10):
        sum = sum + i
    return sum  # 5+6+7+8+9 = 35


@compile
def test_for_with_step() -> i32:
    """Test for loop with step"""
    sum: i32 = 0
    for i in seq(0, 20, 3):
        sum = sum + i
    return sum  # 0+3+6+9+12+15+18 = 63


@compile
def test_for_negative_step() -> i32:
    """Test for loop with negative step"""
    sum: i32 = 0
    for i in seq(10, 0, -1):
        sum = sum + i
    return sum  # 10+9+8+7+6+5+4+3+2+1 = 55


@compile
def test_for_empty() -> i32:
    """Test for loop with empty range"""
    count: i32 = 0
    for i in seq(10, 5):  # Start > end with positive step
        count = count + 1
    return count  # 0


@compile
def test_for_nested() -> i32:
    """Test nested for loops"""
    sum: i32 = 0
    for i in seq(5):
        for j in seq(5):
            sum = sum + i * 10 + j
    return sum


@compile
def test_for_with_break() -> i32:
    """Test for loop with break"""
    sum: i32 = 0
    for i in seq(100):
        sum = sum + i
        if sum > 50:
            break
    return sum  # 55


@compile
def test_for_with_continue() -> i32:
    """Test for loop with continue"""
    sum: i32 = 0
    for i in seq(10):
        if i % 2 == 0:
            continue
        sum = sum + i
    return sum  # 1+3+5+7+9 = 25


# =============================================================================
# Complex Control Flow Patterns
# =============================================================================

@compile
def test_loop_with_if_else() -> i32:
    """Test loop containing if-else"""
    result: i32 = 0
    i: i32 = 0
    
    while i < 10:
        if i < 5:
            result = result + 1
        else:
            result = result + 10
        i = i + 1
    
    return result  # 5*1 + 5*10 = 55


@compile
def test_if_containing_loop() -> i32:
    """Test if statement containing loop"""
    x: i32 = 5
    result: i32 = 0
    
    if x > 0:
        i: i32 = 0
        while i < x:
            result = result + i
            i = i + 1
    else:
        result = -1
    
    return result  # 0+1+2+3+4 = 10


@compile
def test_multiple_loops_sequential() -> i32:
    """Test multiple sequential loops"""
    result: i32 = 0
    
    i: i32 = 0
    while i < 5:
        result = result + 1
        i = i + 1
    
    j: i32 = 0
    while j < 5:
        result = result + 10
        j = j + 1
    
    k: i32 = 0
    while k < 5:
        result = result + 100
        k = k + 1
    
    return result  # 5 + 50 + 500 = 555


@compile
def test_alternating_control_flow() -> i32:
    """Test alternating if and loop structures"""
    x: i32 = 10
    result: i32 = 0
    
    if x > 5:
        i: i32 = 0
        while i < 3:
            if i == 1:
                result = result + 100
            else:
                result = result + 10
            i = i + 1
    
    if result > 50:
        j: i32 = 0
        while j < 2:
            result = result + 1
            j = j + 1
    
    return result  # 10 + 100 + 10 + 1 + 1 = 122


@compile
def test_early_return_in_loop() -> i32:
    """Test early return from within a loop"""
    i: i32 = 0
    while i < 100:
        if i == 7:
            return i * 10
        i = i + 1
    return -1


@compile
def test_multiple_returns_in_if() -> i32:
    """Test multiple return statements in if branches"""
    x: i32 = 5
    
    if x < 0:
        return -1
    elif x == 0:
        return 0
    elif x < 10:
        return 1
    else:
        return 2


@compile
def test_return_in_nested_structure() -> i32:
    """Test return in deeply nested structure"""
    i: i32 = 0
    while i < 10:
        j: i32 = 0
        while j < 10:
            if i == 3 and j == 5:
                return i * 100 + j
            j = j + 1
        i = i + 1
    return -1  # 305


# =============================================================================
# Edge Cases with Boolean Conditions
# =============================================================================

@compile
def test_while_true_with_break() -> i32:
    """Test while True with break"""
    count: i32 = 0
    while True:
        count = count + 1
        if count >= 10:
            break
    return count  # 10


@compile
def test_while_false() -> i32:
    """Test while False (never executes)"""
    count: i32 = 0
    while False:
        count = count + 1
    return count  # 0


@compile
def test_if_true() -> i32:
    """Test if True (always executes)"""
    result: i32 = 0
    if True:
        result = 42
    return result  # 42


@compile
def test_if_false() -> i32:
    """Test if False (never executes)"""
    result: i32 = 42
    if False:
        result = 0
    return result  # 42


# =============================================================================
# State Machine Pattern
# =============================================================================

@compile
def test_state_machine() -> i32:
    """Test state machine pattern with control flow"""
    state: i32 = 0
    result: i32 = 0
    iterations: i32 = 0
    
    while state != 4:
        iterations = iterations + 1
        if iterations > 100:
            break  # Safety limit
        
        if state == 0:
            result = result + 1
            state = 1
        elif state == 1:
            result = result + 10
            state = 2
        elif state == 2:
            result = result + 100
            state = 3
        elif state == 3:
            result = result + 1000
            state = 4
    
    return result  # 1 + 10 + 100 + 1000 = 1111


@compile
def test_accumulator_pattern() -> i32:
    """Test accumulator pattern"""
    acc: i32 = 0
    i: i32 = 1
    
    while i <= 10:
        acc = acc + i * i  # Sum of squares
        i = i + 1
    
    return acc  # 1+4+9+16+25+36+49+64+81+100 = 385


@compile
def test_find_first_pattern() -> i32:
    """Test find first matching element pattern"""
    target: i32 = 7
    i: i32 = 0
    found_index: i32 = -1
    
    while i < 20:
        if i * i > 50:  # First i where i^2 > 50
            found_index = i
            break
        i = i + 1
    
    return found_index  # 8 (8^2 = 64 > 50)


@compile
def test_count_matching_pattern() -> i32:
    """Test count matching elements pattern"""
    count: i32 = 0
    i: i32 = 1
    
    while i <= 100:
        # Count numbers divisible by both 3 and 5
        if i % 3 == 0 and i % 5 == 0:
            count = count + 1
        i = i + 1
    
    return count  # 6 (15, 30, 45, 60, 75, 90)


# =============================================================================
# Test Runner
# =============================================================================

class TestIfElse(unittest.TestCase):
    def test_deep_nesting(self):
        self.assertEqual(test_deeply_nested_if(), 100)
    
    def test_elif_chain(self):
        self.assertEqual(test_if_elif_chain(), 70)
    
    def test_complex_conditions(self):
        self.assertEqual(test_if_with_complex_conditions(), 31)
    
    def test_all_paths(self):
        self.assertEqual(test_if_all_paths(), 7)
    
    def test_empty_branches(self):
        self.assertEqual(test_if_empty_branches(), 1)


class TestWhileLoop(unittest.TestCase):
    def test_zero_iterations(self):
        self.assertEqual(test_while_zero_iterations(), 0)
    
    def test_one_iteration(self):
        self.assertEqual(test_while_one_iteration(), 1)
    
    def test_large_iterations(self):
        self.assertEqual(test_while_large_iterations(), 499500)
    
    def test_nested_deep(self):
        self.assertEqual(test_while_nested_deep(), 81)
    
    def test_multiple_conditions(self):
        self.assertEqual(test_while_with_multiple_conditions(), 10)


class TestBreak(unittest.TestCase):
    def test_first_iteration(self):
        self.assertEqual(test_break_first_iteration(), 1)
    
    def test_conditional(self):
        self.assertEqual(test_break_conditional(), 55)
    
    def test_nested_inner(self):
        self.assertEqual(test_break_nested_inner(), 515)
    
    def test_after_work(self):
        self.assertEqual(test_break_after_work(), 160)


class TestContinue(unittest.TestCase):
    def test_skip_all(self):
        self.assertEqual(test_continue_skip_all(), 25)
    
    def test_in_nested(self):
        self.assertEqual(test_continue_in_nested(), 20)
    
    def test_complex_condition(self):
        self.assertEqual(test_continue_with_complex_condition(), 112)
    
    def test_with_break(self):
        self.assertEqual(test_break_and_continue_together(), 25)


class TestForLoop(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(test_for_basic(), 45)
    
    def test_with_start(self):
        self.assertEqual(test_for_with_start(), 35)
    
    def test_with_step(self):
        self.assertEqual(test_for_with_step(), 63)
    
    def test_negative_step(self):
        self.assertEqual(test_for_negative_step(), 55)
    
    def test_empty(self):
        self.assertEqual(test_for_empty(), 0)
    
    def test_with_break(self):
        self.assertEqual(test_for_with_break(), 55)
    
    def test_with_continue(self):
        self.assertEqual(test_for_with_continue(), 25)


class TestComplexPatterns(unittest.TestCase):
    def test_loop_with_if(self):
        self.assertEqual(test_loop_with_if_else(), 55)
    
    def test_if_with_loop(self):
        self.assertEqual(test_if_containing_loop(), 10)
    
    def test_sequential_loops(self):
        self.assertEqual(test_multiple_loops_sequential(), 555)
    
    def test_early_return(self):
        self.assertEqual(test_early_return_in_loop(), 70)
    
    def test_multiple_returns(self):
        self.assertEqual(test_multiple_returns_in_if(), 1)
    
    def test_return_in_nested(self):
        self.assertEqual(test_return_in_nested_structure(), 305)


class TestBooleanConditions(unittest.TestCase):
    def test_while_true(self):
        self.assertEqual(test_while_true_with_break(), 10)
    
    def test_while_false(self):
        self.assertEqual(test_while_false(), 0)
    
    def test_if_true(self):
        self.assertEqual(test_if_true(), 42)
    
    def test_if_false(self):
        self.assertEqual(test_if_false(), 42)


class TestPatterns(unittest.TestCase):
    def test_state_machine(self):
        self.assertEqual(test_state_machine(), 1111)
    
    def test_accumulator(self):
        self.assertEqual(test_accumulator_pattern(), 385)
    
    def test_find_first(self):
        self.assertEqual(test_find_first_pattern(), 8)
    
    def test_count_matching(self):
        self.assertEqual(test_count_matching_pattern(), 6)


if __name__ == '__main__':
    unittest.main()
