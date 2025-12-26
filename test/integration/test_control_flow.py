#!/usr/bin/env python3
"""
Control flow tests (if/else, while, break, continue)
Note: for loop tests are in test_for_loop.py
"""

from pythoc import i32, bool, compile

@compile
def test_if_else() -> i32:
    """Test if-else statements"""
    x: i32 = 10
    result: i32 = 0
    
    if x > 5:
        result = 1
    else:
        result = 2
    
    return result

@compile
def test_nested_if() -> i32:
    """Test nested if statements"""
    x: i32 = 10
    y: i32 = 20
    result: i32 = 0
    
    if x > 5:
        if y > 15:
            result = 1
        else:
            result = 2
    else:
        result = 3
    
    return result

@compile
def test_while_loop() -> i32:
    """Test while loop"""
    sum: i32 = 0
    i: i32 = 0
    
    while i < 10:
        sum = sum + i
        i = i + 1
    
    return sum

@compile
def test_break_statement() -> i32:
    """Test break statement"""
    sum: i32 = 0
    i: i32 = 0
    
    while i < 100:
        if i >= 10:
            break
        sum = sum + i
        i = i + 1
    
    return sum

@compile
def test_continue_statement() -> i32:
    """Test continue statement"""
    sum: i32 = 0
    i: i32 = 0
    
    while i < 10:
        i = i + 1
        if i % 2 == 0:
            continue
        sum = sum + i
    
    return sum

@compile
def test_nested_loops() -> i32:
    """Test nested loops"""
    sum: i32 = 0
    i: i32 = 0
    
    while i < 5:
        j: i32 = 0
        while j < 5:
            sum = sum + i * j
            j = j + 1
        i = i + 1
    
    return sum

@compile
def test_nested_break() -> i32:
    """Test break in nested loops"""
    sum: i32 = 0
    i: i32 = 0
    
    while i < 10:
        j: i32 = 0
        while j < 10:
            if j >= 5:
                break
            sum = sum + 1
            j = j + 1
        i = i + 1
    
    return sum

import unittest


class TestControlFlow(unittest.TestCase):
    """Test control flow statements"""

    def test_if_else(self):
        """Test if-else statements"""
        self.assertEqual(test_if_else(), 1)

    def test_nested_if(self):
        """Test nested if statements"""
        self.assertEqual(test_nested_if(), 1)

    def test_while_loop(self):
        """Test while loop: sum of 0..9 = 45"""
        self.assertEqual(test_while_loop(), 45)

    def test_break_statement(self):
        """Test break statement: sum of 0..9 = 45"""
        self.assertEqual(test_break_statement(), 45)

    def test_continue_statement(self):
        """Test continue statement: sum of odd 1..9 = 25"""
        self.assertEqual(test_continue_statement(), 25)

    def test_nested_loops(self):
        """Test nested loops"""
        self.assertEqual(test_nested_loops(), 100)

    def test_nested_break(self):
        """Test break in nested loops: 10 * 5 = 50"""
        self.assertEqual(test_nested_break(), 50)


if __name__ == "__main__":
    unittest.main()
