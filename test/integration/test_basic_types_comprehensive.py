#!/usr/bin/env python3
"""
Comprehensive tests for basic types including edge cases, boundary values,
overflow/underflow behavior, and type conversions.
"""

import unittest
from pythoc import (
    i8, i16, i32, i64,
    u8, u16, u32, u64,
    f32, f64, bool, ptr, compile
)


# =============================================================================
# Integer Boundary Tests
# =============================================================================

@compile
def test_i8_max() -> i8:
    """Test i8 maximum value"""
    x: i8 = 127
    return x


@compile
def test_i8_min() -> i8:
    """Test i8 minimum value"""
    x: i8 = -128
    return x


@compile
def test_i16_max() -> i16:
    """Test i16 maximum value"""
    x: i16 = 32767
    return x


@compile
def test_i16_min() -> i16:
    """Test i16 minimum value"""
    x: i16 = -32768
    return x


@compile
def test_i32_max() -> i32:
    """Test i32 maximum value"""
    x: i32 = 2147483647
    return x


@compile
def test_i32_min() -> i32:
    """Test i32 minimum value"""
    x: i32 = -2147483648
    return x


@compile
def test_i64_max() -> i64:
    """Test i64 maximum value"""
    x: i64 = 9223372036854775807
    return x


@compile
def test_i64_min() -> i64:
    """Test i64 minimum value"""
    x: i64 = -9223372036854775808
    return x


@compile
def test_u8_max() -> u8:
    """Test u8 maximum value"""
    x: u8 = 255
    return x


@compile
def test_u8_zero() -> u8:
    """Test u8 zero"""
    x: u8 = 0
    return x


@compile
def test_u16_max() -> u16:
    """Test u16 maximum value"""
    x: u16 = 65535
    return x


@compile
def test_u32_max() -> u32:
    """Test u32 maximum value"""
    x: u32 = 4294967295
    return x


@compile
def test_u64_max() -> u64:
    """Test u64 maximum value"""
    x: u64 = 18446744073709551615
    return x


# =============================================================================
# Integer Overflow/Underflow Tests (Wrap-around behavior)
# =============================================================================

@compile
def test_i8_overflow() -> i8:
    """Test i8 overflow wraps around"""
    x: i8 = 127
    y: i8 = x + 1  # Should wrap to -128
    return y


@compile
def test_i8_underflow() -> i8:
    """Test i8 underflow wraps around"""
    x: i8 = -128
    y: i8 = x - 1  # Should wrap to 127
    return y


@compile
def test_u8_overflow() -> u8:
    """Test u8 overflow wraps around"""
    x: u8 = 255
    y: u8 = x + 1  # Should wrap to 0
    return y


@compile
def test_u8_underflow() -> u8:
    """Test u8 underflow wraps around"""
    x: u8 = 0
    y: u8 = x - 1  # Should wrap to 255
    return y


@compile
def test_i32_overflow() -> i32:
    """Test i32 overflow"""
    x: i32 = 2147483647
    y: i32 = x + 1  # Should wrap to -2147483648
    return y


@compile
def test_u32_overflow() -> u32:
    """Test u32 overflow"""
    x: u32 = 4294967295
    y: u32 = x + 1  # Should wrap to 0
    return y


# =============================================================================
# Type Conversion Edge Cases
# =============================================================================

@compile
def test_signed_to_unsigned_negative() -> u32:
    """Test converting negative signed to unsigned"""
    x: i32 = -1
    y: u32 = u32(x)  # Should be 4294967295
    return y


@compile
def test_unsigned_to_signed_large() -> i32:
    """Test converting large unsigned to signed"""
    x: u32 = 4294967295
    y: i32 = i32(x)  # Should be -1
    return y


@compile
def test_truncation_i64_to_i32() -> i32:
    """Test truncation from i64 to i32"""
    x: i64 = 0x123456789ABCDEF0
    y: i32 = i32(x)  # Should keep lower 32 bits
    return y


@compile
def test_truncation_i32_to_i8() -> i8:
    """Test truncation from i32 to i8"""
    x: i32 = 0x12345678
    y: i8 = i8(x)  # Should keep lower 8 bits (0x78 = 120)
    return y


@compile
def test_sign_extension_i8_to_i32() -> i32:
    """Test sign extension from i8 to i32"""
    x: i8 = -1  # 0xFF
    y: i32 = i32(x)  # Should be -1 (0xFFFFFFFF)
    return y


@compile
def test_zero_extension_u8_to_u32() -> u32:
    """Test zero extension from u8 to u32"""
    x: u8 = 255  # 0xFF
    y: u32 = u32(x)  # Should be 255 (0x000000FF)
    return y


@compile
def test_mixed_sign_conversion() -> i32:
    """Test mixed sign conversions"""
    a: u8 = 200
    b: i8 = i8(a)  # Interpret as signed: -56
    c: i32 = i32(b)  # Sign extend: -56
    return c


# =============================================================================
# Floating Point Edge Cases
# =============================================================================

@compile
def test_f32_small_positive() -> f32:
    """Test small positive f32"""
    x: f32 = f32(0.0000001)
    return x


@compile
def test_f32_large_positive() -> f32:
    """Test large positive f32"""
    x: f32 = f32(3.4e38)
    return x


@compile
def test_f64_precision() -> f64:
    """Test f64 precision"""
    x: f64 = 1.0000000000000001
    return x


@compile
def test_f64_negative_zero() -> f64:
    """Test negative zero"""
    x: f64 = -0.0
    return x


@compile
def test_float_int_roundtrip() -> i32:
    """Test float to int conversion truncates"""
    x: f64 = 3.9
    y: i32 = i32(x)  # Should be 3
    return y


@compile
def test_float_negative_truncation() -> i32:
    """Test negative float truncation"""
    x: f64 = -3.9
    y: i32 = i32(x)  # Should be -3
    return y


@compile
def test_f32_f64_conversion() -> f64:
    """Test f32 to f64 conversion"""
    x: f32 = f32(3.14159)
    y: f64 = f64(x)
    return y


@compile
def test_f64_f32_precision_loss() -> f32:
    """Test f64 to f32 with precision loss"""
    x: f64 = 3.141592653589793
    y: f32 = f32(x)  # Loses precision
    return y


# =============================================================================
# Boolean Edge Cases
# =============================================================================

@compile
def test_bool_from_zero() -> bool:
    """Test bool from zero integer"""
    x: i32 = 0
    b: bool = bool(x)
    if b:
        return True
    return False


@compile
def test_bool_from_nonzero() -> bool:
    """Test bool from non-zero integer"""
    x: i32 = 42
    b: bool = bool(x)
    return b


@compile
def test_bool_from_negative() -> bool:
    """Test bool from negative integer"""
    x: i32 = -1
    b: bool = bool(x)
    return b


@compile
def test_bool_to_int() -> i32:
    """Test bool to int conversion"""
    t: bool = True
    f: bool = False
    return i32(t) + i32(f)  # Should be 1 + 0 = 1


@compile
def test_bool_arithmetic() -> i32:
    """Test bool in arithmetic context"""
    t: bool = True
    f: bool = False
    x: i32 = i32(t) * 10 + i32(f) * 5
    return x  # Should be 10


# =============================================================================
# Pointer Conversion Edge Cases
# =============================================================================

@compile
def test_ptr_to_int_and_back() -> i32:
    """Test pointer to integer and back"""
    x: i32 = 42
    p: ptr[i32] = ptr(x)
    addr: i64 = i64(p)
    p2: ptr[i32] = ptr[i32](addr)
    return p2[0]


@compile
def test_ptr_arithmetic_with_cast() -> i32:
    """Test pointer arithmetic through integer cast"""
    x: i32 = 100
    p: ptr[i32] = ptr(x)
    addr: i64 = i64(p)
    addr2: i64 = addr + 4  # Move by sizeof(i32)
    p2: ptr[i32] = ptr[i32](addr2)
    # Note: p2 points to invalid memory, just testing cast works
    return i32(addr2 - addr)  # Should be 4


@compile
def test_different_ptr_types() -> i32:
    """Test casting between different pointer types"""
    x: i64 = 0x0102030405060708
    p64: ptr[i64] = ptr(x)
    p8: ptr[i8] = ptr[i8](p64)
    # Access first byte
    first_byte: i8 = p8[0]
    return i32(first_byte)


# =============================================================================
# Mixed Type Arithmetic
# =============================================================================

@compile
def test_mixed_int_arithmetic() -> i64:
    """Test arithmetic with mixed integer types"""
    a: i8 = 10
    b: i16 = 20
    c: i32 = 30
    d: i64 = 40
    result: i64 = i64(a) + i64(b) + i64(c) + d
    return result


@compile
def test_mixed_signed_unsigned() -> i64:
    """Test arithmetic with mixed signed/unsigned"""
    a: i32 = -10
    b: u32 = 20
    result: i64 = i64(a) + i64(b)
    return result


@compile
def test_float_int_mixed() -> f64:
    """Test arithmetic with mixed float and int"""
    a: i32 = 10
    b: f64 = 3.5
    result: f64 = f64(a) + b
    return result


# =============================================================================
# Zero and One Special Cases
# =============================================================================

@compile
def test_multiply_by_zero() -> i32:
    """Test multiplication by zero"""
    x: i32 = 12345
    y: i32 = 0
    return x * y


@compile
def test_divide_by_one() -> i32:
    """Test division by one"""
    x: i32 = 12345
    return x / 1


@compile
def test_modulo_by_one() -> i32:
    """Test modulo by one"""
    x: i32 = 12345
    return x % 1


@compile
def test_add_zero() -> i32:
    """Test addition of zero"""
    x: i32 = 12345
    return x + 0


@compile
def test_subtract_zero() -> i32:
    """Test subtraction of zero"""
    x: i32 = 12345
    return x - 0


# =============================================================================
# Negative Number Operations
# =============================================================================

@compile
def test_negative_multiplication() -> i32:
    """Test negative number multiplication"""
    a: i32 = -5
    b: i32 = -3
    return a * b  # Should be 15


@compile
def test_negative_division() -> i32:
    """Test negative number division"""
    a: i32 = -15
    b: i32 = 3
    return a / b  # Should be -5


@compile
def test_negative_modulo() -> i32:
    """Test negative number modulo"""
    a: i32 = -17
    b: i32 = 5
    return a % b  # C behavior: -2


@compile
def test_double_negative() -> i32:
    """Test double negation"""
    x: i32 = -42
    y: i32 = -x
    return y  # Should be 42


# =============================================================================
# Bitwise Operations on Different Types
# =============================================================================

@compile
def test_bitwise_on_i8() -> i8:
    """Test bitwise operations on i8"""
    a: i8 = 0x55  # 01010101
    b: i8 = 0x33  # 00110011
    return a ^ b  # 01100110 = 0x66 = 102


@compile
def test_bitwise_on_u8() -> u8:
    """Test bitwise operations on u8"""
    a: u8 = 0xFF
    b: u8 = 0x0F
    return a & b  # Should be 0x0F = 15


@compile
def test_shift_i32() -> i32:
    """Test shift operations on i32"""
    x: i32 = 1
    y: i32 = x << 31  # Shift to sign bit
    return y  # Should be -2147483648 (signed interpretation)


@compile
def test_shift_u32() -> u32:
    """Test shift operations on u32"""
    x: u32 = 1
    y: u32 = x << 31  # Shift to MSB
    return y  # Should be 2147483648


@compile
def test_right_shift_signed() -> i32:
    """Test arithmetic right shift on signed"""
    x: i32 = -8
    y: i32 = x >> 2  # Arithmetic shift preserves sign
    return y  # Should be -2


@compile
def test_right_shift_unsigned() -> u32:
    """Test logical right shift on unsigned"""
    x: u32 = 0x80000000
    y: u32 = x >> 2  # Logical shift fills with zeros
    return y  # Should be 0x20000000


# =============================================================================
# Test Runner
# =============================================================================

class TestIntegerBoundaries(unittest.TestCase):
    def test_i8_boundaries(self):
        self.assertEqual(test_i8_max(), 127)
        self.assertEqual(test_i8_min(), -128)
    
    def test_i16_boundaries(self):
        self.assertEqual(test_i16_max(), 32767)
        self.assertEqual(test_i16_min(), -32768)
    
    def test_i32_boundaries(self):
        self.assertEqual(test_i32_max(), 2147483647)
        self.assertEqual(test_i32_min(), -2147483648)
    
    def test_i64_boundaries(self):
        self.assertEqual(test_i64_max(), 9223372036854775807)
        self.assertEqual(test_i64_min(), -9223372036854775808)
    
    def test_unsigned_boundaries(self):
        self.assertEqual(test_u8_max(), 255)
        self.assertEqual(test_u8_zero(), 0)
        self.assertEqual(test_u16_max(), 65535)
        self.assertEqual(test_u32_max(), 4294967295)
        self.assertEqual(test_u64_max(), 18446744073709551615)


class TestOverflowUnderflow(unittest.TestCase):
    def test_i8_wrap(self):
        self.assertEqual(test_i8_overflow(), -128)
        self.assertEqual(test_i8_underflow(), 127)
    
    def test_u8_wrap(self):
        self.assertEqual(test_u8_overflow(), 0)
        self.assertEqual(test_u8_underflow(), 255)
    
    def test_i32_wrap(self):
        self.assertEqual(test_i32_overflow(), -2147483648)
    
    def test_u32_wrap(self):
        self.assertEqual(test_u32_overflow(), 0)


class TestTypeConversions(unittest.TestCase):
    def test_signed_unsigned_conversions(self):
        self.assertEqual(test_signed_to_unsigned_negative(), 4294967295)
        self.assertEqual(test_unsigned_to_signed_large(), -1)
    
    def test_truncation(self):
        # 0x9ABCDEF0 in lower 32 bits
        self.assertEqual(test_truncation_i64_to_i32(), -1698898192)
        self.assertEqual(test_truncation_i32_to_i8(), 120)
    
    def test_extension(self):
        self.assertEqual(test_sign_extension_i8_to_i32(), -1)
        self.assertEqual(test_zero_extension_u8_to_u32(), 255)
    
    def test_mixed_sign(self):
        self.assertEqual(test_mixed_sign_conversion(), -56)


class TestFloatingPoint(unittest.TestCase):
    def test_float_truncation(self):
        self.assertEqual(test_float_int_roundtrip(), 3)
        self.assertEqual(test_float_negative_truncation(), -3)
    
    def test_bool_conversions(self):
        self.assertEqual(test_bool_from_zero(), False)
        self.assertEqual(test_bool_from_nonzero(), True)
        self.assertEqual(test_bool_from_negative(), True)
        self.assertEqual(test_bool_to_int(), 1)
        self.assertEqual(test_bool_arithmetic(), 10)


class TestPointerConversions(unittest.TestCase):
    def test_ptr_int_roundtrip(self):
        self.assertEqual(test_ptr_to_int_and_back(), 42)
    
    def test_ptr_arithmetic(self):
        self.assertEqual(test_ptr_arithmetic_with_cast(), 4)


class TestMixedArithmetic(unittest.TestCase):
    def test_mixed_int(self):
        self.assertEqual(test_mixed_int_arithmetic(), 100)
    
    def test_mixed_signed_unsigned(self):
        self.assertEqual(test_mixed_signed_unsigned(), 10)
    
    def test_float_int(self):
        self.assertAlmostEqual(test_float_int_mixed(), 13.5, places=5)


class TestSpecialCases(unittest.TestCase):
    def test_zero_operations(self):
        self.assertEqual(test_multiply_by_zero(), 0)
        self.assertEqual(test_divide_by_one(), 12345)
        self.assertEqual(test_modulo_by_one(), 0)
        self.assertEqual(test_add_zero(), 12345)
        self.assertEqual(test_subtract_zero(), 12345)
    
    def test_negative_operations(self):
        self.assertEqual(test_negative_multiplication(), 15)
        self.assertEqual(test_negative_division(), -5)
        self.assertEqual(test_negative_modulo(), -2)
        self.assertEqual(test_double_negative(), 42)


class TestBitwiseOperations(unittest.TestCase):
    def test_bitwise_types(self):
        self.assertEqual(test_bitwise_on_i8(), 102)
        self.assertEqual(test_bitwise_on_u8(), 15)
    
    def test_shifts(self):
        self.assertEqual(test_shift_i32(), -2147483648)
        self.assertEqual(test_shift_u32(), 2147483648)
        self.assertEqual(test_right_shift_signed(), -2)
        self.assertEqual(test_right_shift_unsigned(), 0x20000000)


if __name__ == '__main__':
    unittest.main()
