"""Bit placement, storage projection, and qualifier regression tests."""

import ctypes
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from llvmlite import binding, ir

from pythoc import const, f32, struct, u3, u5, u6, u8, u33, union, void, volatile
from pythoc.builtin_entities.bitfield_access import access_bit_field
from pythoc.target_layout import make_aggregate_layout_config
from pythoc.valueref import ValueRef
from pythoc.logger import logger


@struct(storage_unit=16)
class AllShort:
    a: u3
    b: u5
    c: u6


@struct(storage_unit=16, storage_unit_overrides={'b': 32})
class MixedUnits:
    a: u3
    b: u5
    c: u6


@struct(storage_unit=16, unit_break_before={'b': 32})
class UnitBoundary:
    a: u3
    b: u3


@union(storage_unit=32, cross_unit=True, abi_alignment=8)
class WideUnion:
    value: u33


def make_visitor():
    module = ir.Module()
    function = ir.Function(module, ir.FunctionType(ir.VoidType(), []), 'test')
    builder = ir.IRBuilder(function.append_basic_block('entry'))
    return SimpleNamespace(module=module, builder=builder,
                           _create_alloca_in_entry=lambda ty, name:
                           builder.alloca(ty, name=name))


class TestBitLayout(unittest.TestCase):
    def test_c_layout_shapes(self):
        for cls, offsets, size, alignment in (
            (AllShort, (0, 3, 8), 2, 2),
            (MixedUnits, (0, 3, 8), 4, 4),
            (UnitBoundary, (0, 32), 6, 2),
        ):
            with self.subTest(cls=cls):
                layout = cls._struct_type.get_layout()
                self.assertEqual(layout.field_offsets_bits, offsets)
                self.assertEqual(layout.allocation_size, size)
                self.assertEqual(layout.alignment, alignment)
                ctype = cls._struct_type.get_ctypes_type()
                self.assertEqual(ctypes.sizeof(ctype), size)
                self.assertEqual(ctypes.alignment(ctype), alignment)
        with self.assertRaisesRegex(ValueError, 'field offset'):
            _ = AllShort._struct_type.get_layout().field_offsets

    def test_crossing_union_reserves_complete_access_units(self):
        layout = WideUnion._union_type.get_layout()
        self.assertEqual(layout.storage_bits, 64)
        self.assertEqual(layout.allocation_size, 8)
        self.assertEqual(layout.alignment, 1)

    def test_mixed_msb_units_cannot_silently_alias_struct_fields(self):
        @struct(storage_unit=8, bit_order='msb0',
                storage_unit_overrides={'b': 16})
        class Overlapping:
            a: u3
            b: u5
            c: u8

        with self.assertRaisesRegex(ValueError, 'overlaps another field'):
            Overlapping._struct_type.get_layout()

    def test_vector_carrier_constant_initializer(self):
        from pythoc.builtin_entities.bitfield_access import assemble_bit_fields

        @struct(storage_unit=8, abi_alignment=256)
        class Overaligned:
            a: u3

        visitor = make_visitor()
        constant = assemble_bit_fields(
            visitor, Overaligned._struct_type, [ir.Constant(ir.IntType(3), 5)])
        global_value = ir.GlobalVariable(visitor.module, constant.type, 'value')
        global_value.initializer = constant
        visitor.builder.ret_void()
        binding.parse_assembly(str(visitor.module)).verify()
        self.assertEqual(Overaligned.get_size_bytes(), 32)

    def test_union_initializers_do_not_silently_merge_active_members(self):
        from pythoc.builtin_entities.bitfield_access import assemble_bit_fields
        visitor = make_visitor()
        with self.assertRaisesRegex(TypeError, 'selected field'):
            assemble_bit_fields(visitor, WideUnion._union_type,
                                [ir.Constant(ir.IntType(33), 1)])

    def test_invalid_options(self):
        for options, error in (
            ({'unknown': 8}, TypeError),
            ({'alignment': True}, TypeError),
            ({'storage_unit': 3}, ValueError),
            ({'storage_unit': 4}, ValueError),
            ({'abi_alignment': 1}, ValueError),
            ({'cross_unit': 1}, TypeError),
            ({'bit_order': 'native'}, ValueError),
            ({'unit_break_before': {'x': 1.5}}, TypeError),
        ):
            with self.subTest(options=options), self.assertRaises(error):
                make_aggregate_layout_config(**options)

    def test_invalid_field_configurations(self):
        for options, field_type, error in (
            ({'storage_unit': 8}, f32, TypeError),
            ({'storage_unit': 8}, void, TypeError),
            ({'storage_unit': 8}, u33, ValueError),
            ({'storage_unit_overrides': {'missing': 16}}, u3, ValueError),
            ({'unit_break_before': {'x': 4}}, u3, None),
        ):
            with self.subTest(options=options):
                cls = struct(**options)(type('Invalid', (), {
                    '__annotations__': {'x': field_type},
                    '__module__': __name__,
                }))
                if error is None:
                    self.assertEqual(cls._struct_type.get_layout()
                                     .field_offsets_bits, (0,))
                else:
                    with self.assertRaises(error):
                        cls._struct_type.get_layout()

    def test_layout_identity_is_normalized_and_distinct(self):
        a = make_aggregate_layout_config(
            storage_unit_overrides={'b': 16, 'a': 8})
        b = make_aggregate_layout_config(
            storage_unit_overrides={'a': 8, 'b': 16})
        self.assertEqual(a.identity(), b.identity())
        natural = struct(type('Natural', (), {
            '__annotations__': {'x': u8}, '__module__': __name__}))
        packed = struct(storage_unit=8)(type('Packed', (), {
            '__annotations__': {'x': u8}, '__module__': __name__}))
        self.assertNotEqual(natural.get_type_id(), packed.get_type_id())
        empty_byte = union(abi_alignment=8)(type('EmptyByte', (), {
            '__annotations__': {}, '__module__': __name__}))
        empty_word = union(abi_alignment=32)(type('EmptyWord', (), {
            '__annotations__': {}, '__module__': __name__}))
        self.assertNotEqual(empty_byte.get_type_id(), empty_word.get_type_id())
        self.assertEqual(empty_byte.get_size_bytes(), 1)
        self.assertEqual(empty_word.get_size_bytes(), 4)

    def test_lazy_volatile_projection_and_single_rmw(self):
        visitor = make_visitor()
        ty = AllShort._struct_type
        pointer = visitor.builder.alloca(ty.get_llvm_type(visitor.module.context))
        base = ValueRef('address', pointer, ty, address=pointer)
        field = access_bit_field(visitor, base, ty, 0, volatile[u3])
        self.assertFalse(field.has_place())
        self.assertTrue(field.has_storage())
        self.assertFalse(field.is_pointer_typed())
        self.assertIsNone(field.pointee())
        self.assertNotIn('load ', str(visitor.module))
        field.store(visitor.builder, ir.Constant(ir.IntType(3), 5))
        text = str(visitor.module)
        self.assertEqual(text.count('load volatile'), 1)
        self.assertEqual(text.count('store volatile'), 1)
        self.assertIn('align 1', text)
        field.ir_value
        self.assertEqual(str(visitor.module).count('load volatile'), 2)
        visitor.builder.ret_void()
        binding.parse_assembly(str(visitor.module)).verify()

    def test_full_unit_assignment_has_no_read(self):
        visitor = make_visitor()
        ty = union(storage_unit=8)(type('Byte', (), {
            '__annotations__': {'x': u8}, '__module__': __name__}))._union_type
        pointer = visitor.builder.alloca(ty.get_llvm_type(visitor.module.context))
        base = ValueRef('address', pointer, ty, address=pointer)
        field = access_bit_field(visitor, base, ty, 0, volatile[u8])
        field.store(visitor.builder, ir.Constant(ir.IntType(8), 42))
        self.assertNotIn('load ', str(visitor.module))
        self.assertEqual(str(visitor.module).count('store volatile'), 1)

    def test_const_and_temporary_are_not_mutable(self):
        visitor = make_visitor()
        ty = AllShort._struct_type
        llvm_type = ty.get_llvm_type(visitor.module.context)
        pointer = visitor.builder.alloca(llvm_type)
        base = ValueRef('address', pointer, ty, address=pointer)
        field = access_bit_field(visitor, base, ty, 0, const[u3])
        with patch.object(logger, 'raise_on_error', True), self.assertRaisesRegex(
                RuntimeError, 'Cannot reassign to const'):
            field.store(visitor.builder, ir.Constant(ir.IntType(3), 1))
        base = ValueRef('value', ir.Constant(llvm_type, None), ty)
        field = access_bit_field(visitor, base, ty, 0, u3)
        with self.assertRaisesRegex(ValueError, 'temporary'):
            field.as_lvalue()

    def test_invalid_programs_are_rejected_for_the_right_reason(self):
        root = Path(__file__).parents[2]
        source = root / 'test/integration/fixtures/bitfield_invalid.py'
        for mode, message in (
            ('address_attribute', 'requires an addressable value'),
            ('address_subscript', 'requires an addressable value'),
            ('offsetof_struct', 'cannot be applied to a bit field'),
            ('offsetof_union', 'cannot be applied to a bit field'),
            ('const_attribute', 'Cannot reassign to const'),
            ('const_subscript', 'Cannot reassign to const'),
        ):
            with self.subTest(mode=mode):
                result = subprocess.run([sys.executable, str(source), mode],
                                        cwd=root, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stdout + result.stderr)

    def test_debug_type_describes_storage_not_fake_byte_fields(self):
        from pythoc.debug_info import DebugInfoBuilder
        module = ir.Module()
        debug = DebugInfoBuilder(module, __file__)
        debug.get_debug_type(AllShort._struct_type)
        text = str(module)
        self.assertIn('name: "_storage"', text)
        self.assertNotIn('name: "b"', text)


if __name__ == '__main__':
    unittest.main()
