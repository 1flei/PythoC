#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function Pointer Demo

Demonstrates various uses of function pointers:
1. Passing functions as arguments
2. Function pointer arrays (function tables)
3. Callbacks
"""

from pythoc import compile
from pythoc.builtin_entities import i32, func, void
from pythoc.libc.stdlib import malloc
from pythoc.libc.stdio import printf


@compile
def add(x: i32, y: i32) -> i32:
    return x + y


@compile
def subtract(x: i32, y: i32) -> i32:
    return x - y


@compile
def multiply(x: i32, y: i32) -> i32:
    return x * y


@compile
def divide(x: i32, y: i32) -> i32:
    if y == 0:
        return 0
    return x / y


@compile
def apply_binary_op(op: func[[i32, i32], i32], x: i32, y: i32) -> i32:
    """Apply a binary operation to two numbers"""
    return op(x, y)


@compile
def square(x: i32) -> i32:
    return x * x


@compile
def double(x: i32) -> i32:
    return x + x


@compile
def apply_unary_op(op: func[[i32], i32], x: i32) -> i32:
    """Apply a unary operation to a number"""
    return op(x)


@compile
def select_operation(op_code: i32) -> func[[i32, i32], i32]:
    """Select an operation based on code (function table)"""
    if op_code == 0:
        return add
    elif op_code == 1:
        return subtract
    elif op_code == 2:
        return multiply
    else:
        return divide


@compile
def calculator(op_code: i32, x: i32, y: i32) -> i32:
    """Simple calculator using function pointers"""
    operation: func[[i32, i32], i32] = select_operation(op_code)
    return operation(x, y)


@compile
def main() -> i32:
    """Test various function pointer operations"""
    # Test 1: Direct function pointer call
    result1: i32 = apply_binary_op(add, 10, 5)  # 15
    
    # Test 2: Different operations
    result2: i32 = apply_binary_op(multiply, 3, 4)  # 12
    
    # Test 3: Unary operations
    result3: i32 = apply_unary_op(square, 4)  # 16
    result4: i32 = apply_unary_op(double, 5)  # 10
    
    # Test 4: Calculator with function table
    calc1: i32 = calculator(0, 20, 10)  # add: 30
    calc2: i32 = calculator(1, 30, 10)  # subtract: 20
    calc3: i32 = calculator(2, 5, 3)    # multiply: 15
    calc4: i32 = calculator(3, 40, 4)   # divide: 10
    
    # Sum all results
    total: i32 = result1 + result2 + result3 + result4
    total = total + calc1 + calc2 + calc3 + calc4
    
    return total  # 15 + 12 + 16 + 10 + 30 + 20 + 15 + 10 = 128


if __name__ == '__main__':
    main()
