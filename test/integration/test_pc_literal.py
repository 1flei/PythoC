#!/usr/bin/env python3
"""
Integration tests for pc_literal: Python-level tagged value carrier.

Tests the full pipeline:
- pc_literal creation outside @compile
- Python-level arithmetic preserving type tags
- Correct lowering when pc_literal values are referenced inside @compile
- Native execution round-trip (pass pc_literal as arg, get pc_literal back)
- Struct/composite pc_literal construction
"""

import unittest
from pythoc import (
    i8, i16, i32, i64,
    u8, u16, u32, u64,
    f32, f64, compile, struct,
    pc_literal,
)


# =============================================================================
# Module-level pc_literal constants (the primary use case)
# =============================================================================

X_U64 = u64(42)
Y_U64 = u64(8)
Z_U64 = X_U64 + Y_U64     # pc_literal arithmetic -> u64(50)

PI_F64 = f64(3.14159265)
HALF_F32 = f32(0.5)

MASK_U8 = u8(0xFF)
SHIFT_AMT = u32(4)

SIGNED_NEG = i32(-100)
UNSIGNED_BIG = u64(0xDEADBEEF)


# =============================================================================
# @compile functions that reference pc_literal globals
# =============================================================================

@compile
def use_u64_constant() -> u64:
    """Reference a u64 pc_literal global inside @compile."""
    return X_U64


@compile
def use_u64_arithmetic_result() -> u64:
    """Reference a u64 pc_literal that was computed via Python-level add."""
    return Z_U64


@compile
def use_f64_constant() -> f64:
    """Reference an f64 pc_literal global."""
    return PI_F64


@compile
def add_pc_literal_and_local(offset: u64) -> u64:
    """Mix a pc_literal global with a @compile local."""
    return X_U64 + offset


@compile
def pc_literal_in_branch() -> i32:
    """Use pc_literal in a conditional branch."""
    if SIGNED_NEG < 0:
        return i32(1)
    return i32(0)


@compile
def pc_literal_in_loop() -> u64:
    """Accumulate using a pc_literal start value in a loop."""
    acc: u64 = X_U64
    i: u64 = u64(0)
    while i < u64(5):
        acc = acc + u64(1)
        i = i + u64(1)
    return acc


@compile
def pc_literal_mixed_widths() -> i64:
    """Use pc_literals of different widths in one function."""
    a: i8 = i8(10)
    b: i16 = i16(20)
    c: i32 = i32(30)
    d: i64 = i64(40)
    return i64(a) + i64(b) + i64(c) + d


# =============================================================================
# Parameterised @compile functions for native-execution round-trip
# =============================================================================

@compile
def identity_i32(x: i32) -> i32:
    return x


@compile
def identity_u64(x: u64) -> u64:
    return x


@compile
def identity_f64(x: f64) -> f64:
    return x


@compile
def add_i32(a: i32, b: i32) -> i32:
    return a + b


@compile
def mul_u64(a: u64, b: u64) -> u64:
    return a * b


@compile
def mixed_arith(a: i32, b: u64) -> u64:
    """Promote i32 arg to u64 inside compiled code."""
    return u64(a) + b


@compile
def float_arith(a: f64, b: f64) -> f64:
    return a * b


# =============================================================================
# Multi-step pipeline: Python arithmetic -> @compile -> Python check
# =============================================================================

TEN = i32(10)
TWENTY = i32(20)
SUM_PY = TEN + TWENTY       # i32(30) at Python level

@compile
def pipeline_step(x: i32) -> i32:
    """Double the value."""
    return x + x


# =============================================================================
# Tests
# =============================================================================

class TestPcLiteralLowering(unittest.TestCase):
    """pc_literal values correctly lower to LLVM IR inside @compile."""

    def test_u64_constant(self):
        self.assertEqual(use_u64_constant(), 42)

    def test_u64_arithmetic_result(self):
        self.assertEqual(use_u64_arithmetic_result(), 50)

    def test_f64_constant(self):
        self.assertAlmostEqual(float(use_f64_constant()), 3.14159265, places=6)

    def test_add_pc_literal_and_local(self):
        result = add_pc_literal_and_local(u64(8))
        self.assertEqual(result, 50)

    def test_branch_on_negative(self):
        self.assertEqual(pc_literal_in_branch(), 1)

    def test_loop_accumulation(self):
        self.assertEqual(pc_literal_in_loop(), 47)  # 42 + 5

    def test_mixed_widths(self):
        self.assertEqual(pc_literal_mixed_widths(), 100)  # 10+20+30+40


class TestPcLiteralNativeRoundTrip(unittest.TestCase):
    """Pass pc_literal as arg to @compile, get pc_literal back."""

    def test_identity_i32(self):
        result = identity_i32(i32(99))
        self.assertIsInstance(result, pc_literal)
        self.assertEqual(result, 99)
        self.assertIs(result._pc_type, i32)

    def test_identity_u64(self):
        result = identity_u64(u64(12345))
        self.assertIsInstance(result, pc_literal)
        self.assertEqual(result, 12345)

    def test_identity_f64(self):
        result = identity_f64(f64(2.718))
        self.assertIsInstance(result, pc_literal)
        self.assertAlmostEqual(float(result), 2.718, places=10)

    def test_add_i32(self):
        result = add_i32(i32(100), i32(200))
        self.assertEqual(result, 300)
        self.assertIs(result._pc_type, i32)

    def test_mul_u64(self):
        result = mul_u64(u64(1000), u64(1000))
        self.assertEqual(result, 1_000_000)

    def test_float_arith(self):
        result = float_arith(f64(3.0), f64(4.0))
        self.assertAlmostEqual(float(result), 12.0)

    def test_plain_int_arg_still_works(self):
        """Plain Python int should still work as argument."""
        result = identity_i32(42)
        self.assertEqual(result, 42)

    def test_plain_float_arg_still_works(self):
        result = identity_f64(3.14)
        self.assertAlmostEqual(float(result), 3.14, places=10)


class TestPcLiteralPipeline(unittest.TestCase):
    """Multi-step: Python arithmetic -> @compile -> Python check."""

    def test_python_sum_then_compile(self):
        """SUM_PY = i32(10) + i32(20) = i32(30), pass to @compile."""
        result = pipeline_step(SUM_PY)
        self.assertEqual(result, 60)  # 30 * 2

    def test_compile_result_then_python_arithmetic(self):
        """Get result from @compile, do Python-level arithmetic."""
        r = identity_i32(i32(25))
        doubled = r + r
        self.assertIsInstance(doubled, pc_literal)
        self.assertEqual(doubled, 50)

    def test_chained_compile_calls(self):
        """Chain @compile calls with pc_literal flowing between them."""
        r1 = add_i32(i32(10), i32(20))
        r2 = add_i32(r1, i32(5))
        self.assertEqual(r2, 35)


class TestPcLiteralTypePreservation(unittest.TestCase):
    """Type tags survive round-trip through native execution."""

    def test_i32_type_preserved(self):
        result = identity_i32(i32(42))
        self.assertIs(result._pc_type, i32)

    def test_u64_type_preserved(self):
        result = identity_u64(u64(100))
        self.assertIs(result._pc_type, u64)

    def test_f64_type_preserved(self):
        result = identity_f64(f64(1.5))
        self.assertIs(result._pc_type, f64)

    def test_repr_after_round_trip(self):
        result = identity_i32(i32(42))
        self.assertEqual(repr(result), 'i32(42)')


class TestPcLiteralPythonArithmetic(unittest.TestCase):
    """Complex Python-level arithmetic chains."""

    def test_chain_of_operations(self):
        a = u64(100)
        b = u64(200)
        c = (a + b) * u64(3) - u64(100)
        self.assertEqual(c, 800)
        self.assertIs(c._pc_type, u64)

    def test_mixed_type_promotion(self):
        a = i8(10)
        b = i32(20)
        c = a + b
        self.assertIs(c._pc_type, i32)
        self.assertEqual(c, 30)

    def test_float_int_promotion(self):
        a = i32(5)
        b = f64(2.5)
        c = a + b
        self.assertIs(c._pc_type, f64)
        self.assertAlmostEqual(float(c), 7.5)

    def test_overflow_wrapping_u8(self):
        a = u8(250)
        b = u8(10)
        c = a + b
        self.assertEqual(c, 4)  # 260 & 0xFF = 4

    def test_overflow_wrapping_i8(self):
        a = i8(120)
        b = i8(20)
        c = a + b
        self.assertEqual(c, -116)  # 140 wraps to -116 in signed 8-bit

    def test_bitwise_chain(self):
        x = u32(0xABCD1234)
        high = (x >> u32(16)) & u32(0xFFFF)
        low = x & u32(0xFFFF)
        self.assertEqual(high, 0xABCD)
        self.assertEqual(low, 0x1234)

    def test_comparison_chain(self):
        a = i32(10)
        b = i32(20)
        c = i32(10)
        self.assertTrue(a < b)
        self.assertTrue(a <= c)
        self.assertTrue(b > a)
        self.assertTrue(a == c)
        self.assertTrue(a != b)

    def test_use_as_dict_key(self):
        """pc_literal should be hashable and usable as dict key."""
        d = {i32(1): 'one', i32(2): 'two'}
        self.assertEqual(d[i32(1)], 'one')
        self.assertEqual(d[i32(2)], 'two')

    def test_use_in_python_control_flow(self):
        """pc_literal should work in standard Python if/while."""
        total = i32(0)
        i = i32(0)
        while i < 10:
            total = total + i32(1)
            i = i + 1
        self.assertEqual(total, 10)

    def test_use_in_list_comprehension(self):
        """pc_literal in Python list comprehension."""
        vals = [i32(x) for x in range(5)]
        total = i32(0)
        for v in vals:
            total = total + v
        self.assertEqual(total, 10)  # 0+1+2+3+4


class TestPcLiteralEdgeCases(unittest.TestCase):
    """Edge cases and corner cases."""

    def test_zero(self):
        z = i32(0)
        self.assertEqual(z, 0)
        self.assertFalse(bool(z))

    def test_max_u64(self):
        m = u64(0xFFFFFFFFFFFFFFFF)
        self.assertEqual(m._value, 0xFFFFFFFFFFFFFFFF)

    def test_negative_i32(self):
        n = i32(-1)
        self.assertEqual(n, -1)

    def test_add_then_pass_to_compile(self):
        """Compute at Python level, then use in @compile."""
        x = u64(100)
        y = u64(200)
        z = x + y
        result = identity_u64(z)
        self.assertEqual(result, 300)

    def test_multiply_then_pass(self):
        a = i32(7)
        b = i32(6)
        product = a * b
        result = identity_i32(product)
        self.assertEqual(result, 42)


if __name__ == "__main__":
    unittest.main()
