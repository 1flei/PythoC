"""Unit tests for pc_literal: Python-level tagged value carrier."""

import unittest
from pythoc.builtin_entities import (
    i8, i16, i32, i64,
    u8, u16, u32, u64,
    f32, f64,
)
from pythoc.builtin_entities.pc_literal import pc_literal


class TestPcLiteralCreation(unittest.TestCase):
    """Test pc_literal construction via type calls."""

    def test_integer_types(self):
        x = i32(42)
        self.assertIsInstance(x, pc_literal)
        self.assertEqual(x._value, 42)
        self.assertIs(x._pc_type, i32)

    def test_unsigned_integer(self):
        x = u64(100)
        self.assertIsInstance(x, pc_literal)
        self.assertEqual(x._value, 100)
        self.assertIs(x._pc_type, u64)

    def test_float_types(self):
        x = f32(3.14)
        self.assertIsInstance(x, pc_literal)
        self.assertAlmostEqual(x._value, 3.14, places=5)
        self.assertIs(x._pc_type, f32)

    def test_f64_creation(self):
        x = f64(2.718)
        self.assertIsInstance(x, pc_literal)
        self.assertAlmostEqual(x._value, 2.718, places=10)
        self.assertIs(x._pc_type, f64)

    def test_integer_overflow_wraps(self):
        x = i8(200)
        self.assertEqual(x._value, -56)  # 200 wraps in signed 8-bit

    def test_unsigned_overflow_wraps(self):
        x = u8(256)
        self.assertEqual(x._value, 0)  # 256 wraps in unsigned 8-bit

    def test_from_pc_literal(self):
        x = i32(42)
        y = i64(x)
        self.assertEqual(y._value, 42)
        self.assertIs(y._pc_type, i64)


class TestPcLiteralRepr(unittest.TestCase):
    """Test repr/str output."""

    def test_integer_repr(self):
        self.assertEqual(repr(i32(42)), 'i32(42)')

    def test_unsigned_repr(self):
        self.assertEqual(repr(u64(100)), 'u64(100)')

    def test_float_repr(self):
        r = repr(f64(3.14))
        self.assertTrue(r.startswith('f64('))


class TestPcLiteralArithmetic(unittest.TestCase):
    """Test Python-level arithmetic on pc_literal."""

    def test_add_same_type(self):
        r = i32(10) + i32(20)
        self.assertIsInstance(r, pc_literal)
        self.assertEqual(r._value, 30)
        self.assertIs(r._pc_type, i32)

    def test_sub(self):
        r = u64(50) - u64(20)
        self.assertEqual(r._value, 30)

    def test_mul(self):
        r = i32(6) * i32(7)
        self.assertEqual(r._value, 42)

    def test_floordiv(self):
        r = i32(10) // i32(3)
        self.assertEqual(r._value, 3)

    def test_mod(self):
        r = i32(10) % i32(3)
        self.assertEqual(r._value, 1)

    def test_bitwise_and(self):
        r = u8(0xFF) & u8(0x0F)
        self.assertEqual(r._value, 0x0F)

    def test_bitwise_or(self):
        r = u8(0xF0) | u8(0x0F)
        self.assertEqual(r._value, 0xFF)

    def test_bitwise_xor(self):
        r = u8(0xFF) ^ u8(0x0F)
        self.assertEqual(r._value, 0xF0)

    def test_lshift(self):
        r = u32(1) << u32(4)
        self.assertEqual(r._value, 16)

    def test_rshift(self):
        r = u32(16) >> u32(2)
        self.assertEqual(r._value, 4)

    def test_neg(self):
        r = -i32(42)
        self.assertEqual(r._value, -42)

    def test_invert(self):
        r = ~u8(0)
        self.assertEqual(r._value, 255)

    def test_add_with_plain_int(self):
        r = i32(10) + 5
        self.assertIsInstance(r, pc_literal)
        self.assertEqual(r._value, 15)
        self.assertIs(r._pc_type, i32)

    def test_radd_with_plain_int(self):
        r = 5 + i32(10)
        self.assertIsInstance(r, pc_literal)
        self.assertEqual(r._value, 15)
        self.assertIs(r._pc_type, i32)


class TestPcLiteralTypeCoercion(unittest.TestCase):
    """Test type coercion in mixed-type operations."""

    def test_wider_type_wins(self):
        r = i8(1) + i32(2)
        self.assertIs(r._pc_type, i32)
        self.assertEqual(r._value, 3)

    def test_float_wins_over_int(self):
        r = i32(1) + f64(2.0)
        self.assertIs(r._pc_type, f64)
        self.assertAlmostEqual(r._value, 3.0)

    def test_signed_wins_same_width(self):
        r = u32(1) + i32(2)
        self.assertIs(r._pc_type, i32)


class TestPcLiteralComparisons(unittest.TestCase):
    """Test comparison operators."""

    def test_eq(self):
        self.assertTrue(i32(42) == i32(42))
        self.assertFalse(i32(42) == i32(43))

    def test_ne(self):
        self.assertTrue(i32(42) != i32(43))

    def test_lt(self):
        self.assertTrue(i32(1) < i32(2))
        self.assertFalse(i32(2) < i32(1))

    def test_le(self):
        self.assertTrue(i32(1) <= i32(1))
        self.assertTrue(i32(1) <= i32(2))

    def test_gt(self):
        self.assertTrue(i32(2) > i32(1))

    def test_ge(self):
        self.assertTrue(i32(2) >= i32(2))

    def test_eq_with_plain_int(self):
        self.assertTrue(i32(42) == 42)

    def test_lt_with_plain_int(self):
        self.assertTrue(i32(1) < 2)


class TestPcLiteralNumericProtocol(unittest.TestCase):
    """Test __int__, __float__, __bool__ for ctypes compatibility."""

    def test_int(self):
        self.assertEqual(int(i32(42)), 42)

    def test_float(self):
        self.assertAlmostEqual(float(f64(3.14)), 3.14)

    def test_float_from_int_type(self):
        self.assertEqual(float(i32(42)), 42.0)

    def test_bool_true(self):
        self.assertTrue(bool(i32(1)))

    def test_bool_false(self):
        self.assertFalse(bool(i32(0)))


class TestPcLiteralGetValue(unittest.TestCase):
    """Test the get_value() protocol for compile-time integration."""

    def test_get_value_returns_valueref(self):
        from pythoc.valueref import ValueRef
        x = i32(42)
        vr = x.get_value()
        self.assertIsInstance(vr, ValueRef)

    def test_get_value_carries_preferred_type(self):
        x = u64(100)
        vr = x.get_value()
        self.assertIsNotNone(vr.type_hint)
        self.assertEqual(vr.type_hint._preferred_pc_type, u64)

    def test_get_value_python_value(self):
        x = i32(42)
        vr = x.get_value()
        self.assertTrue(vr.is_python_value())
        self.assertEqual(vr.get_python_value(), 42)


class TestPcLiteralHash(unittest.TestCase):
    """Test hashing for use in sets/dicts."""

    def test_hashable(self):
        x = i32(42)
        s = {x}
        self.assertIn(x, s)

    def test_different_values_different_hash(self):
        self.assertNotEqual(hash(i32(1)), hash(i32(2)))


class TestPcLiteralFactory(unittest.TestCase):
    """Test factory methods."""

    def test_from_ctypes_result_int(self):
        r = pc_literal._from_ctypes_result(42, i32)
        self.assertIsInstance(r, pc_literal)
        self.assertEqual(r._value, 42)
        self.assertIs(r._pc_type, i32)

    def test_from_ctypes_result_float(self):
        r = pc_literal._from_ctypes_result(3.14, f64)
        self.assertIsInstance(r, pc_literal)
        self.assertAlmostEqual(r._value, 3.14)

    def test_from_ctypes_result_none_type(self):
        r = pc_literal._from_ctypes_result(42, None)
        self.assertEqual(r, 42)  # passthrough


if __name__ == '__main__':
    unittest.main()
