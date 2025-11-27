#!/usr/bin/env python3
"""
Operator tests (arithmetic, bitwise, logical, comparison)
"""

from pythoc import i32, bool, compile

@compile
def test_arithmetic_operators() -> i32:
    """Test arithmetic operators"""
    a: i32 = 10
    b: i32 = 3
    
    add: i32 = a + b
    sub: i32 = a - b
    mul: i32 = a * b
    div: i32 = a / b
    mod: i32 = a % b
    
    return add + sub + mul + div + mod

@compile
def test_bitwise_operators() -> i32:
    """Test bitwise operators"""
    a: i32 = 12
    b: i32 = 10
    
    and_result: i32 = a & b
    or_result: i32 = a | b
    xor_result: i32 = a ^ b
    not_result: i32 = ~a
    shl_result: i32 = a << 2
    shr_result: i32 = a >> 1
    
    return and_result + or_result + xor_result + not_result + shl_result + shr_result

@compile
def test_comparison_operators() -> i32:
    """Test comparison operators"""
    a: i32 = 10
    b: i32 = 20
    c: i32 = 10
    
    result: i32 = 0
    
    if a < b:
        result = result + 1
    if a <= c:
        result = result + 1
    if b > a:
        result = result + 1
    if c >= a:
        result = result + 1
    if a == c:
        result = result + 1
    if a != b:
        result = result + 1
    
    return result

@compile
def test_logical_operators() -> i32:
    """Test logical operators"""
    t: bool = True
    f: bool = False
    
    result: i32 = 0
    
    if t and t:
        result = result + 1
    if t or f:
        result = result + 1
    if not f:
        result = result + 1
    if not (f and t):
        result = result + 1
    
    return result

@compile
def test_unary_operators() -> i32:
    """Test unary operators"""
    a: i32 = 10
    b: i32 = -a
    c: i32 = +a
    
    return b + c

if __name__ == "__main__":
    print(f"test_arithmetic_operators: {test_arithmetic_operators()}")
    print(f"test_bitwise_operators: {test_bitwise_operators()}")
    print(f"test_comparison_operators: {test_comparison_operators()}")
    print(f"test_logical_operators: {test_logical_operators()}")
    print(f"test_unary_operators: {test_unary_operators()}")
