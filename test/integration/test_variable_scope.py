#!/usr/bin/env python3
"""
Variable scope tests

Test variable scope and shadowing behavior without inline functions.
Tests include:
- Local variable scope within a function
- Variable shadowing in nested blocks (if/while)
- Variable usage after scope exit
- Name reuse in different scopes
"""

from pythoc import i32, bool, compile

@compile
def test_basic_scope() -> i32:
    """Test basic variable scope - variables defined in function scope"""
    x: i32 = 10
    y: i32 = 20
    z: i32 = x + y
    return z

@compile
def test_same_name_in_if_blocks() -> i32:
    """Test variable shadowing in if blocks"""
    x: i32 = 1
    result: i32 = x  # result = 1
    
    if True:
        y: i32 = 2
        result = result + y  # result = 3
    
    # y should not be accessible here, but x should be
    result = result + x  # result = 4
    return result

@compile
def test_same_name_in_while_blocks() -> i32:
    """Test variable shadowing in while blocks"""
    x: i32 = 5
    result: i32 = 0
    i: i32 = 0
    
    while i < 3:
        y: i32 = x * 2  # y = 10
        result = result + y
        i = i + 1
    
    # y should not be accessible here
    result = result + x  # Add original x
    return result

@compile
def test_shadowing_in_if() -> i32:
    """Test variable shadowing - inner scope shadows outer scope"""
    x: i32 = 10
    
    if True:
        x: i32 = 20  # This should shadow the outer x
        return x  # Should return 20, not 10
    
    return x

@compile
def test_shadowing_in_if2() -> i32:
    """Test variable shadowing - inner scope shadows outer scope"""
    x: i32 = 10
    if True:
        x: i32 = 20  # This should shadow the outer x
    return x  # Should return 10

@compile
def test_shadowing_in_while() -> i32:
    """Test variable shadowing in while loop"""
    x: i32 = 100
    i: i32 = 0
    
    while i < 1:
        x: i32 = 200  # This should shadow the outer x
        i = i + 1
        return x  # Should return 200
    
    return x  # This should not be reached

@compile
def test_shadowing_in_while2() -> i32:
    """Test variable shadowing in while loop"""
    x: i32 = 100
    i: i32 = 0
    
    while i < 1:
        x: i32 = 200  # This should shadow the outer x
        i = i + 1
    
    return x  # Should return 100

@compile
def test_nested_if_shadowing() -> i32:
    """Test nested if with multiple levels of shadowing"""
    x: i32 = 1
    
    if True:
        x: i32 = 2  # First level shadowing
        if True:
            x: i32 = 3  # Second level shadowing
            return x  # Should return 3
        return x  # Should not reach here
    
    return x  # Should not reach here

@compile
def test_nested_while_shadowing() -> i32:
    """Test nested while with shadowing"""
    x: i32 = 1
    i: i32 = 0
    
    while i < 1:
        x: i32 = 2  # Shadow outer x
        j: i32 = 0
        while j < 1:
            x: i32 = 3  # Shadow middle x
            j = j + 1
            return x  # Should return 3
        i = i + 1
    
    return x  # Should not reach here

@compile
def test_no_shadowing_simple() -> i32:
    """Test that variables in same scope are reused, not shadowed"""
    x: i32 = 10
    y: i32 = x + 5  # y = 15
    
    # Reassign x (not redeclare)
    x = 20
    z: i32 = x + y  # z = 20 + 15 = 35
    
    return z

@compile
def test_loop_variable_scope() -> i32:
    """Test that loop variables maintain proper scope"""
    sum: i32 = 0
    i: i32 = 0
    
    while i < 5:
        temp: i32 = i * 2
        sum = sum + temp
        i = i + 1
    
    # temp should not be accessible here
    # but i should still be accessible
    result: i32 = sum + i  # i should be 5 now
    return result

@compile
def test_separate_if_blocks() -> i32:
    """Test that variables in separate if blocks don't interfere"""
    x: i32 = 10
    result: i32 = 0
    
    if x > 5:
        y: i32 = 1
        result = result + y
    
    if x > 0:
        # This y is different from the y in the previous if block
        y: i32 = 2
        result = result + y
    
    return result  # Should be 1 + 2 = 3

@compile
def test_parameter_shadowing() -> i32:
    """Test that local variables can shadow function parameters"""
    x: i32 = 10
    y: i32 = 5
    
    if True:
        x: i32 = 100  # Shadow parameter x
        y: i32 = 50   # Shadow parameter y
        return x + y  # Should return 150
    
    return x + y  # Should not reach here

@compile
def test_multiple_assignment_same_scope() -> i32:
    """Test multiple assignments to same variable in same scope"""
    x: i32 = 1
    x = 2
    x = 3
    x = 4
    return x  # Should return 4

@compile
def test_mixed_shadowing_and_assignment() -> i32:
    """Test mixing shadowing and assignment"""
    x: i32 = 10
    
    if True:
        # Use outer x
        temp: i32 = x  # temp = 10
        # Shadow outer x
        x: i32 = 20
        # Now x refers to inner variable
        result: i32 = x + temp  # result = 20 + 10 = 30
        return result
    
    return x

@compile
def test_sequential_blocks_no_interference() -> i32:
    """Test that sequential blocks don't interfere with each other"""
    result: i32 = 0
    
    # First block
    if True:
        x: i32 = 10
        result = result + x
    
    # Second block - x from first block should not be accessible
    if True:
        x: i32 = 20  # This is a new x, not related to previous x
        result = result + x
    
    return result  # Should be 10 + 20 = 30

import unittest


class TestVariableScope(unittest.TestCase):
    """Test variable scope and shadowing behavior"""

    def test_basic_scope(self):
        """Test basic variable scope"""
        self.assertEqual(test_basic_scope(), 30)

    def test_same_name_in_if_blocks(self):
        """Test variable shadowing in if blocks"""
        self.assertEqual(test_same_name_in_if_blocks(), 4)

    def test_same_name_in_while_blocks(self):
        """Test variable shadowing in while blocks"""
        self.assertEqual(test_same_name_in_while_blocks(), 35)

    def test_shadowing_in_if(self):
        """Test variable shadowing - inner scope shadows outer scope"""
        self.assertEqual(test_shadowing_in_if(), 20)

    def test_shadowing_in_if2(self):
        """Test variable shadowing - inner scope shadows outer scope"""
        self.assertEqual(test_shadowing_in_if2(), 10)

    def test_shadowing_in_while(self):
        """Test variable shadowing in while loop"""
        self.assertEqual(test_shadowing_in_while(), 200)

    def test_shadowing_in_while2(self):
        """Test variable shadowing in while loop"""
        self.assertEqual(test_shadowing_in_while2(), 100)

    def test_nested_if_shadowing(self):
        """Test nested if with multiple levels of shadowing"""
        self.assertEqual(test_nested_if_shadowing(), 3)

    def test_nested_while_shadowing(self):
        """Test nested while with shadowing"""
        self.assertEqual(test_nested_while_shadowing(), 3)

    def test_no_shadowing_simple(self):
        """Test that variables in same scope are reused"""
        self.assertEqual(test_no_shadowing_simple(), 35)

    def test_loop_variable_scope(self):
        """Test that loop variables maintain proper scope"""
        self.assertEqual(test_loop_variable_scope(), 25)

    def test_separate_if_blocks(self):
        """Test that variables in separate if blocks don't interfere"""
        self.assertEqual(test_separate_if_blocks(), 3)

    def test_parameter_shadowing(self):
        """Test that local variables can shadow function parameters"""
        self.assertEqual(test_parameter_shadowing(), 150)

    def test_multiple_assignment_same_scope(self):
        """Test multiple assignments to same variable in same scope"""
        self.assertEqual(test_multiple_assignment_same_scope(), 4)

    def test_mixed_shadowing_and_assignment(self):
        """Test mixing shadowing and assignment"""
        self.assertEqual(test_mixed_shadowing_and_assignment(), 30)

    def test_sequential_blocks_no_interference(self):
        """Test that sequential blocks don't interfere with each other"""
        self.assertEqual(test_sequential_blocks_no_interference(), 30)


if __name__ == "__main__":
    unittest.main()
