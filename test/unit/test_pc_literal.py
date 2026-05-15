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


class TestPcLiteralCtypesOwner(unittest.TestCase):
    """ctypes-backed pc_literal: pointer + live struct semantics."""

    def _make_ptr_pc_type(self):
        # Minimal duck-typed PC type with _is_pointer flag; pc_literal
        # only consults attribute presence, not ctypes details.
        class _PtrTy:
            _is_pointer = True
            _is_integer = False
            _is_float = False
            _is_bool = False
            _size_bytes = 8

            @classmethod
            def get_name(cls):
                return 'ptr[u8]'
        return _PtrTy

    def test_pointer_keeps_owner(self):
        import ctypes as ct
        ptr_ty = self._make_ptr_pc_type()
        buf = (ct.c_uint8 * 4)(1, 2, 3, 4)
        addr = ct.addressof(buf)
        owner = ct.c_void_p(addr)
        r = pc_literal._from_ctypes_result(owner, ptr_ty)
        self.assertIsInstance(r, pc_literal)
        self.assertEqual(r._value, addr)
        self.assertIs(r._ctypes_owner, owner)

    def test_pointer_to_ctypes_zero_copy(self):
        import ctypes as ct
        ptr_ty = self._make_ptr_pc_type()
        owner = ct.c_void_p(0xCAFEBABE)
        r = pc_literal._from_ctypes_result(owner, ptr_ty)
        out = r._to_ctypes(ct.c_void_p)
        # Owner is itself a c_void_p so zero-copy returns it directly.
        self.assertIs(out, owner)
        self.assertEqual(out.value, 0xCAFEBABE)

    def test_pointer_to_ctypes_void_p_from_typed_owner(self):
        import ctypes as ct
        ptr_ty = self._make_ptr_pc_type()
        # Owner is a typed pointer rather than c_void_p; bridging path
        # extracts the integer address while self still roots the buffer.
        buf = (ct.c_uint8 * 4)(1, 2, 3, 4)
        typed_owner = ct.cast(buf, ct.POINTER(ct.c_uint8))
        r = pc_literal._from_ctypes_result(typed_owner, ptr_ty)
        out = r._to_ctypes(ct.c_void_p)
        self.assertIsInstance(out, int)
        self.assertEqual(out, ct.addressof(buf))

    def test_live_struct_lazy_field_read(self):
        import ctypes as ct

        class _CT(ct.Structure):
            _fields_ = [('a', ct.c_int32), ('b', ct.c_int64)]

        class _PCStruct:
            _field_names = ['a', 'b']
            _field_types = [i32, i64]

            @classmethod
            def get_name(cls):
                return 'PCStruct'

        ct_inst = _CT(7, 99)
        r = pc_literal._from_ctypes_result(ct_inst, _PCStruct)
        self.assertIs(r._ctypes_owner, ct_inst)
        # Lazy read sees current buffer.
        self.assertEqual(int(r.a), 7)
        self.assertEqual(int(r.b), 99)
        # Mutating the live owner is visible through pc_literal.
        ct_inst.a = 11
        self.assertEqual(int(r.a), 11)

    def test_live_struct_writeback(self):
        import ctypes as ct

        class _CT(ct.Structure):
            _fields_ = [('a', ct.c_int32)]

        class _PCStruct:
            _field_names = ['a']
            _field_types = [i32]

            @classmethod
            def get_name(cls):
                return 'PCStruct'

        ct_inst = _CT(0)
        r = pc_literal._from_ctypes_result(ct_inst, _PCStruct)
        r.a = 123
        self.assertEqual(ct_inst.a, 123)

    def test_live_struct_to_ctypes_zero_copy(self):
        import ctypes as ct

        class _CT(ct.Structure):
            _fields_ = [('a', ct.c_int32)]

        class _PCStruct:
            _field_names = ['a']
            _field_types = [i32]

            @classmethod
            def get_name(cls):
                return 'PCStruct'

        ct_inst = _CT(5)
        r = pc_literal._from_ctypes_result(ct_inst, _PCStruct)
        out = r._to_ctypes(_CT)
        self.assertIs(out, ct_inst)

    def test_live_struct_same_layout_different_class(self):
        import ctypes as ct

        class _CT_A(ct.Structure):
            _fields_ = [('x', ct.c_int32), ('y', ct.c_int32)]

        class _CT_B(ct.Structure):
            _fields_ = [('x', ct.c_int32), ('y', ct.c_int32)]

        class _PCStruct:
            _field_names = ['x', 'y']
            _field_types = [i32, i32]

            @classmethod
            def get_name(cls):
                return 'PCStruct'

        a = _CT_A(11, 22)
        r = pc_literal._from_ctypes_result(a, _PCStruct)
        out = r._to_ctypes(_CT_B)
        self.assertIsInstance(out, _CT_B)
        self.assertEqual(out.x, 11)
        self.assertEqual(out.y, 22)

    def test_get_value_pointer_owner_rejected(self):
        """Capturing a runtime pointer is rejected: a heap address has
        no stable identity to bake into a cacheable IR artefact.  Users
        are expected to pass the pointer as an explicit argument
        instead (which goes through ``_to_ctypes`` zero-copy)."""
        import ctypes as ct
        ptr_ty = self._make_ptr_pc_type()
        owner = ct.c_void_p(0xDEADBEEF)
        r = pc_literal._from_ctypes_result(owner, ptr_ty)
        with self.assertRaises((TypeError, SystemExit)):
            r.get_value()

    def test_get_value_live_struct_owner(self):
        """ctypes-backed by-value struct pc_literal forwards itself as the
        PythonType payload; PythonType's hasattr-handle_attribute hook
        then routes field access through pc_literal.handle_attribute at
        IR build time (which has access to module.context)."""
        import ctypes as ct

        class _CT(ct.Structure):
            _fields_ = [('a', ct.c_int32), ('b', ct.c_int64)]

        class _PCStruct:
            _field_names = ['a', 'b']
            _field_types = [i32, i64]

            @classmethod
            def get_name(cls):
                return 'PCStruct'

        ct_inst = _CT(11, 22)
        r = pc_literal._from_ctypes_result(ct_inst, _PCStruct)
        vr = r.get_value()
        # Carrier is the pc_literal itself; downstream hits
        # pc_literal.handle_attribute on field access.
        self.assertIs(vr.value, r)
        self.assertTrue(callable(getattr(vr.value, 'handle_attribute', None)))
        # Lazy field reads still see the live owner.
        self.assertEqual(int(r.a), 11)
        self.assertEqual(int(r.b), 22)
        # Mutation through Python remains visible.
        ct_inst.a = 99
        self.assertEqual(int(r.a), 99)


if __name__ == '__main__':
    unittest.main()
