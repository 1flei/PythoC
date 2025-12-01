#!/usr/bin/env python3
"""
Comprehensive tests for operators including compound operators,
edge cases, operator precedence, and type mixing.
"""

import unittest
from pythoc import (
    i8, i16, i32, i64,
    u8, u16, u32, u64,
    f32, f64, bool, ptr, compile
)


# =============================================================================
# Compound Assignment Operators
# =============================================================================

@compile
def test_add_assign() -> i32:
    """Test += operator"""
    x: i32 = 10
    x += 5
    return x  # 15


@compile
def test_sub_assign() -> i32:
    """Test -= operator"""
    x: i32 = 10
    x -= 3
    return x  # 7


@compile
def test_mul_assign() -> i32:
    """Test *= operator"""
    x: i32 = 10
    x *= 4
    return x  # 40


@compile
def test_div_assign() -> i32:
    """Test /= operator"""
    x: i32 = 20
    x /= 4
    return x  # 5


@compile
def test_mod_assign() -> i32:
    """Test %= operator"""
    x: i32 = 17
    x %= 5
    return x  # 2


@compile
def test_and_assign() -> i32:
    """Test &= operator"""
    x: i32 = 0xFF
    x &= 0x0F
    return x  # 15


@compile
def test_or_assign() -> i32:
    """Test |= operator"""
    x: i32 = 0x0F
    x |= 0xF0
    return x  # 255


@compile
def test_xor_assign() -> i32:
    """Test ^= operator"""
    x: i32 = 0xFF
    x ^= 0x0F
    return x  # 240


@compile
def test_lshift_assign() -> i32:
    """Test <<= operator"""
    x: i32 = 1
    x <<= 4
    return x  # 16


@compile
def test_rshift_assign() -> i32:
    """Test >>= operator"""
    x: i32 = 64
    x >>= 2
    return x  # 16


@compile
def test_compound_chain() -> i32:
    """Test chained compound assignments"""
    x: i32 = 10
    x += 5
    x *= 2
    x -= 10
    x /= 2
    return x  # ((10+5)*2-10)/2 = 10


# =============================================================================
# Operator Precedence Tests
# =============================================================================

@compile
def test_precedence_mul_add() -> i32:
    """Test * has higher precedence than +"""
    return 2 + 3 * 4  # 14, not 20


@compile
def test_precedence_div_sub() -> i32:
    """Test / has higher precedence than -"""
    return 10 - 6 / 2  # 7, not 2


@compile
def test_precedence_parentheses() -> i32:
    """Test parentheses override precedence"""
    return (2 + 3) * 4  # 20


@compile
def test_precedence_bitwise_comparison() -> i32:
    """Test bitwise vs comparison precedence"""
    # In C: & has lower precedence than ==
    # 5 & 3 = 1, then 1 == 1 is true
    x: i32 = 5 & 3
    if x == 1:
        return 1
    return 0


@compile
def test_precedence_shift_add() -> i32:
    """Test shift vs addition precedence"""
    # << has lower precedence than +
    return 1 << 2 + 1  # 1 << 3 = 8


@compile
def test_precedence_and_or() -> i32:
    """Test && vs || precedence"""
    # && has higher precedence than ||
    t: bool = True
    f: bool = False
    if f and f or t:  # (f && f) || t = t
        return 1
    return 0


@compile
def test_precedence_not() -> i32:
    """Test unary not precedence"""
    t: bool = True
    f: bool = False
    if not f and t:  # (not f) and t = t
        return 1
    return 0


@compile
def test_precedence_complex() -> i32:
    """Test complex precedence expression"""
    # 2 + 3 * 4 - 8 / 2 + 1 << 2
    # = 2 + 12 - 4 + (1 << 2)
    # = 2 + 12 - 4 + 4 = 14
    return 2 + 3 * 4 - 8 / 2 + 1 << 2


# =============================================================================
# Comparison Operator Edge Cases
# =============================================================================

@compile
def test_compare_equal_values() -> i32:
    """Test comparing equal values"""
    a: i32 = 42
    b: i32 = 42
    result: i32 = 0
    if a == b:
        result = result + 1
    if a <= b:
        result = result + 1
    if a >= b:
        result = result + 1
    if not (a < b):
        result = result + 1
    if not (a > b):
        result = result + 1
    if not (a != b):
        result = result + 1
    return result  # Should be 6


@compile
def test_compare_negative() -> i32:
    """Test comparing negative numbers"""
    a: i32 = -10
    b: i32 = -5
    result: i32 = 0
    if a < b:
        result = result + 1
    if a <= b:
        result = result + 1
    if b > a:
        result = result + 1
    if b >= a:
        result = result + 1
    return result  # Should be 4


@compile
def test_compare_signed_boundary() -> i32:
    """Test comparing at signed boundaries"""
    min_val: i32 = -2147483648
    max_val: i32 = 2147483647
    result: i32 = 0
    if min_val < max_val:
        result = result + 1
    if min_val < 0:
        result = result + 1
    if max_val > 0:
        result = result + 1
    return result  # Should be 3


@compile
def test_compare_unsigned() -> i32:
    """Test unsigned comparisons"""
    a: u32 = 0xFFFFFFFF  # Max u32
    b: u32 = 0
    result: i32 = 0
    if a > b:
        result = result + 1
    if b < a:
        result = result + 1
    return result  # Should be 2


@compile
def test_compare_float() -> i32:
    """Test floating point comparisons"""
    a: f64 = 3.14
    b: f64 = 3.14
    c: f64 = 3.15
    result: i32 = 0
    if a == b:
        result = result + 1
    if a < c:
        result = result + 1
    if c > a:
        result = result + 1
    return result  # Should be 3


@compile
def test_compare_float_near_zero() -> i32:
    """Test floating point comparisons near zero"""
    a: f64 = 0.0
    b: f64 = -0.0
    c: f64 = 0.0000001
    result: i32 = 0
    if a == b:  # -0.0 == 0.0
        result = result + 1
    if c > a:
        result = result + 1
    return result  # Should be 2


# =============================================================================
# Logical Operator Edge Cases
# =============================================================================

@compile
def test_logical_short_circuit_and() -> i32:
    """Test && short-circuit evaluation"""
    x: i32 = 0
    f: bool = False
    # Second expression should not be evaluated
    if f and True:
        x = 1
    return x  # Should be 0


@compile
def test_logical_short_circuit_or() -> i32:
    """Test || short-circuit evaluation"""
    x: i32 = 0
    t: bool = True
    # Second expression should not be evaluated
    if t or False:
        x = 1
    return x  # Should be 1


@compile
def test_logical_complex() -> i32:
    """Test complex logical expressions"""
    a: bool = True
    b: bool = False
    c: bool = True
    d: bool = False
    result: i32 = 0
    
    if a and c:
        result = result + 1
    if a or b:
        result = result + 1
    if not b:
        result = result + 1
    if (a and b) or (c and not d):
        result = result + 1
    if not (b or d):
        result = result + 1
    
    return result  # Should be 5


@compile
def test_logical_with_comparison() -> i32:
    """Test logical operators with comparisons"""
    x: i32 = 10
    y: i32 = 20
    z: i32 = 15
    result: i32 = 0
    
    if x < y and y > z:
        result = result + 1
    if x < z or z > y:
        result = result + 1
    if not (x > y):
        result = result + 1
    if x < z and z < y:
        result = result + 1
    
    return result  # Should be 4


# =============================================================================
# Bitwise Operator Edge Cases
# =============================================================================

@compile
def test_bitwise_all_ones() -> i32:
    """Test bitwise operations with all ones"""
    x: i32 = -1  # All bits set
    y: i32 = 0x0F0F0F0F
    return x & y  # Should be 0x0F0F0F0F


@compile
def test_bitwise_all_zeros() -> i32:
    """Test bitwise operations with all zeros"""
    x: i32 = 0
    y: i32 = 0x12345678
    return x | y  # Should be 0x12345678


@compile
def test_bitwise_xor_self() -> i32:
    """Test XOR with self gives zero"""
    x: i32 = 0x12345678
    return x ^ x  # Should be 0


@compile
def test_bitwise_not() -> i32:
    """Test bitwise NOT"""
    x: i32 = 0
    return ~x  # Should be -1 (all bits set)


@compile
def test_bitwise_not_pattern() -> i32:
    """Test bitwise NOT pattern"""
    x: i32 = 0x55555555  # 01010101...
    y: i32 = ~x  # Should be 0xAAAAAAAA = -1431655766
    return y


@compile
def test_shift_by_zero() -> i32:
    """Test shift by zero"""
    x: i32 = 42
    return (x << 0) + (x >> 0)  # Should be 84


@compile
def test_shift_all_bits() -> i32:
    """Test shift by type width - 1"""
    x: i32 = 1
    y: i32 = x << 31  # MSB set
    z: i32 = y >> 31  # Arithmetic shift fills with sign
    return z  # Should be -1


@compile
def test_bitwise_extract_byte() -> i32:
    """Test extracting a byte using bitwise ops"""
    x: i32 = 0x12345678
    byte0: i32 = x & 0xFF  # 0x78 = 120
    byte1: i32 = (x >> 8) & 0xFF  # 0x56 = 86
    byte2: i32 = (x >> 16) & 0xFF  # 0x34 = 52
    byte3: i32 = (x >> 24) & 0xFF  # 0x12 = 18
    return byte0 + byte1 + byte2 + byte3  # 276


@compile
def test_bitwise_set_bit() -> i32:
    """Test setting a specific bit"""
    x: i32 = 0
    x = x | (1 << 5)  # Set bit 5
    x = x | (1 << 10)  # Set bit 10
    return x  # 32 + 1024 = 1056


@compile
def test_bitwise_clear_bit() -> i32:
    """Test clearing a specific bit"""
    x: i32 = 0xFF
    x = x & ~(1 << 3)  # Clear bit 3
    return x  # 255 - 8 = 247


@compile
def test_bitwise_toggle_bit() -> i32:
    """Test toggling a specific bit"""
    x: i32 = 0x0F
    x = x ^ (1 << 2)  # Toggle bit 2 (was 1, becomes 0)
    x = x ^ (1 << 4)  # Toggle bit 4 (was 0, becomes 1)
    return x  # 0x0F ^ 0x04 ^ 0x10 = 0x1B = 27


# =============================================================================
# Arithmetic Edge Cases
# =============================================================================

@compile
def test_division_rounding() -> i32:
    """Test integer division rounding toward zero"""
    a: i32 = 7 / 3  # 2
    b: i32 = -7 / 3  # -2 (toward zero)
    return a + b  # Should be 0


@compile
def test_modulo_sign() -> i32:
    """Test modulo sign follows dividend (C behavior)"""
    a: i32 = 7 % 3  # 1
    b: i32 = -7 % 3  # -1
    c: i32 = 7 % -3  # 1
    d: i32 = -7 % -3  # -1
    return a + b + c + d  # 0


@compile
def test_unary_minus_min() -> i32:
    """Test unary minus on minimum value (undefined behavior in C)"""
    # Note: This is technically UB, but we test the behavior
    x: i32 = -2147483648
    y: i32 = -x  # Wraps to -2147483648
    return y


@compile
def test_multiplication_overflow() -> i32:
    """Test multiplication that overflows"""
    x: i32 = 100000
    y: i32 = 100000
    z: i32 = x * y  # Overflows i32
    return z  # Implementation-defined result


@compile
def test_chained_arithmetic() -> i32:
    """Test chained arithmetic operations"""
    result: i32 = 100
    result = result + 50
    result = result - 30
    result = result * 2
    result = result / 4
    result = result % 7
    return result  # ((100+50-30)*2/4) % 7 = 60 % 7 = 4


# =============================================================================
# Unary Operators
# =============================================================================

@compile
def test_unary_plus() -> i32:
    """Test unary plus operator"""
    x: i32 = 42
    y: i32 = +x
    return y  # 42


@compile
def test_unary_minus() -> i32:
    """Test unary minus operator"""
    x: i32 = 42
    y: i32 = -x
    return y  # -42


@compile
def test_unary_not_bool() -> i32:
    """Test unary not on bool"""
    t: bool = True
    f: bool = False
    result: i32 = 0
    if not t:
        result = result + 1
    if not f:
        result = result + 10
    return result  # 10


@compile
def test_unary_bitwise_not() -> i32:
    """Test unary bitwise not"""
    x: i32 = 0
    y: i32 = ~x  # -1
    z: i32 = ~y  # 0
    return z


@compile
def test_double_unary() -> i32:
    """Test double unary operators"""
    x: i32 = 42
    a: i32 = --x  # Double minus = positive
    b: i32 = ~~x  # Double bitwise not = original
    return a + b  # 84


# =============================================================================
# Pointer Operators
# =============================================================================

@compile
def test_ptr_subscript() -> i32:
    """Test pointer subscript operator"""
    x: i32 = 42
    p: ptr[i32] = ptr(x)
    return p[0]


@compile
def test_ptr_add() -> i64:
    """Test pointer addition"""
    x: i32 = 42
    p: ptr[i32] = ptr(x)
    p2: ptr[i32] = p + 1
    diff: i64 = i64(p2) - i64(p)
    return diff  # Should be 4 (sizeof i32)


@compile
def test_ptr_sub() -> i64:
    """Test pointer subtraction"""
    x: i32 = 42
    p: ptr[i32] = ptr(x)
    p2: ptr[i32] = p + 5
    p3: ptr[i32] = p2 - 3
    diff: i64 = i64(p3) - i64(p)
    return diff  # Should be 8 (2 * sizeof i32)


@compile
def test_ptr_comparison() -> i32:
    """Test pointer comparison"""
    x: i32 = 42
    p1: ptr[i32] = ptr(x)
    p2: ptr[i32] = p1
    p3: ptr[i32] = p1 + 1
    result: i32 = 0
    if p1 == p2:
        result = result + 1
    if p1 != p3:
        result = result + 1
    if p1 < p3:
        result = result + 1
    return result  # Should be 3


# =============================================================================
# Mixed Type Operators
# =============================================================================

@compile
def test_mixed_int_sizes_add() -> i64:
    """Test addition with mixed integer sizes"""
    a: i8 = 10
    b: i16 = 20
    c: i32 = 30
    d: i64 = 40
    return i64(a) + i64(b) + i64(c) + d


@compile
def test_mixed_signed_unsigned_compare() -> i32:
    """Test comparison between signed and unsigned"""
    s: i32 = -1
    u: u32 = 1
    # When comparing, -1 as u32 is very large
    result: i32 = 0
    if u32(s) > u:  # -1 as unsigned > 1
        result = 1
    return result


@compile
def test_float_int_operators() -> i32:
    """Test operators between float and int"""
    a: i32 = 10
    b: f64 = 3.5
    c: f64 = f64(a) + b  # 13.5
    d: f64 = f64(a) - b  # 6.5
    e: f64 = f64(a) * b  # 35.0
    f: f64 = f64(a) / b  # ~2.857
    result: i32 = i32(c) + i32(d) + i32(e) + i32(f)
    return result  # 13 + 6 + 35 + 2 = 56


# =============================================================================
# Test Runner
# =============================================================================

class TestCompoundAssignment(unittest.TestCase):
    def test_arithmetic_compound(self):
        self.assertEqual(test_add_assign(), 15)
        self.assertEqual(test_sub_assign(), 7)
        self.assertEqual(test_mul_assign(), 40)
        self.assertEqual(test_div_assign(), 5)
        self.assertEqual(test_mod_assign(), 2)
    
    def test_bitwise_compound(self):
        self.assertEqual(test_and_assign(), 15)
        self.assertEqual(test_or_assign(), 255)
        self.assertEqual(test_xor_assign(), 240)
        self.assertEqual(test_lshift_assign(), 16)
        self.assertEqual(test_rshift_assign(), 16)
    
    def test_compound_chain(self):
        self.assertEqual(test_compound_chain(), 10)


class TestPrecedence(unittest.TestCase):
    def test_basic_precedence(self):
        self.assertEqual(test_precedence_mul_add(), 14)
        self.assertEqual(test_precedence_div_sub(), 7)
        self.assertEqual(test_precedence_parentheses(), 20)
    
    def test_bitwise_precedence(self):
        self.assertEqual(test_precedence_bitwise_comparison(), 1)
        self.assertEqual(test_precedence_shift_add(), 8)
    
    def test_logical_precedence(self):
        self.assertEqual(test_precedence_and_or(), 1)
        self.assertEqual(test_precedence_not(), 1)


class TestComparison(unittest.TestCase):
    def test_equal_values(self):
        self.assertEqual(test_compare_equal_values(), 6)
    
    def test_negative_compare(self):
        self.assertEqual(test_compare_negative(), 4)
    
    def test_boundary_compare(self):
        self.assertEqual(test_compare_signed_boundary(), 3)
    
    def test_unsigned_compare(self):
        self.assertEqual(test_compare_unsigned(), 2)
    
    def test_float_compare(self):
        self.assertEqual(test_compare_float(), 3)
        self.assertEqual(test_compare_float_near_zero(), 2)


class TestLogical(unittest.TestCase):
    def test_short_circuit(self):
        self.assertEqual(test_logical_short_circuit_and(), 0)
        self.assertEqual(test_logical_short_circuit_or(), 1)
    
    def test_complex_logical(self):
        self.assertEqual(test_logical_complex(), 5)
        self.assertEqual(test_logical_with_comparison(), 4)


class TestBitwise(unittest.TestCase):
    def test_basic_bitwise(self):
        self.assertEqual(test_bitwise_all_ones(), 0x0F0F0F0F)
        self.assertEqual(test_bitwise_all_zeros(), 0x12345678)
        self.assertEqual(test_bitwise_xor_self(), 0)
        self.assertEqual(test_bitwise_not(), -1)
    
    def test_shift(self):
        self.assertEqual(test_shift_by_zero(), 84)
        self.assertEqual(test_shift_all_bits(), -1)
    
    def test_bit_manipulation(self):
        self.assertEqual(test_bitwise_extract_byte(), 276)
        self.assertEqual(test_bitwise_set_bit(), 1056)
        self.assertEqual(test_bitwise_clear_bit(), 247)
        self.assertEqual(test_bitwise_toggle_bit(), 27)


class TestArithmetic(unittest.TestCase):
    def test_division_modulo(self):
        self.assertEqual(test_division_rounding(), 0)
        self.assertEqual(test_modulo_sign(), 0)
    
    def test_chained(self):
        self.assertEqual(test_chained_arithmetic(), 4)


class TestUnary(unittest.TestCase):
    def test_plus_minus(self):
        self.assertEqual(test_unary_plus(), 42)
        self.assertEqual(test_unary_minus(), -42)
    
    def test_not(self):
        self.assertEqual(test_unary_not_bool(), 10)
        self.assertEqual(test_unary_bitwise_not(), 0)
    
    def test_double_unary(self):
        self.assertEqual(test_double_unary(), 84)


class TestPointerOperators(unittest.TestCase):
    def test_subscript(self):
        self.assertEqual(test_ptr_subscript(), 42)
    
    def test_arithmetic(self):
        self.assertEqual(test_ptr_add(), 4)
        self.assertEqual(test_ptr_sub(), 8)
    
    def test_comparison(self):
        self.assertEqual(test_ptr_comparison(), 3)


class TestMixedTypes(unittest.TestCase):
    def test_mixed_sizes(self):
        self.assertEqual(test_mixed_int_sizes_add(), 100)
    
    def test_signed_unsigned(self):
        self.assertEqual(test_mixed_signed_unsigned_compare(), 1)
    
    def test_float_int(self):
        self.assertEqual(test_float_int_operators(), 56)


if __name__ == '__main__':
    unittest.main()
