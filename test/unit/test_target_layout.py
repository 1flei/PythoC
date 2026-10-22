"""Target-aware semantic layout tests."""

import ctypes
import unittest
from unittest.mock import patch

from llvmlite import ir

from pythoc import (
    array,
    const,
    f64,
    i8,
    i24,
    i32,
    i64,
    i128,
    linear,
    ptr,
    pyconst,
    simd,
    struct,
    union,
)
from pythoc.config import config
from pythoc.session import CompileSession
from pythoc.target_layout import (
    TargetLayout, TypeLayout, get_target_layout, get_target_machine,
)


class _CStandardRecord(ctypes.Structure):
    _fields_ = [
        ("tag", ctypes.c_uint8),
        ("value", ctypes.c_int64),
    ]


class _CStandardUnion(ctypes.Union):
    _fields_ = [
        ("integer", ctypes.c_int32),
        ("floating", ctypes.c_double),
    ]


class TestTargetLayout(unittest.TestCase):
    def test_unusual_integer_distinguishes_storage_and_allocation(self):
        layout = i24.get_layout()

        self.assertEqual(layout.bit_width, 24)
        self.assertEqual(layout.storage_size, 3)
        self.assertEqual(layout.allocation_size, 4)
        self.assertEqual(layout.alignment, 4)
        self.assertEqual(layout.stride, 4)

    def test_array_uses_element_stride(self):
        layout = array[i24, 3].get_layout()

        self.assertEqual(layout.allocation_size, 12)
        self.assertEqual(layout.alignment, 4)
        self.assertEqual(layout.stride, 12)

    def test_struct_exposes_offsets_and_tail_padding(self):
        pc_type = struct[i8, i128]
        layout = pc_type.get_layout()
        llvm_type = pc_type.get_llvm_type()
        target_data = get_target_layout().target_data
        offset = llvm_type.get_element_offset(target_data, 1)
        self.assertEqual(layout.field_offsets, (0, offset))
        self.assertEqual(layout.storage_size, offset + i128.get_size_bytes())
        self.assertEqual(layout.allocation_size, llvm_type.get_abi_size(target_data))
        self.assertEqual(layout.alignment, llvm_type.get_abi_alignment(target_data))

    def test_union_storage_matches_largest_member(self):
        pc_type = union[array[i24, 3], i64]
        layout = pc_type.get_layout()
        llvm_type = pc_type.get_llvm_type()
        target_data = get_target_layout().target_data

        self.assertEqual(layout.field_offsets, (0, 0))
        self.assertEqual(layout.allocation_size, 16)
        self.assertEqual(layout.alignment, 8)
        self.assertEqual(llvm_type.get_abi_size(target_data), 16)
        self.assertEqual(llvm_type.get_abi_alignment(target_data), 8)

    def test_standard_layout_matches_c_oracle(self):
        layout = struct[i8, i64].get_layout()

        self.assertEqual(layout.allocation_size, ctypes.sizeof(_CStandardRecord))
        self.assertEqual(layout.alignment, ctypes.alignment(_CStandardRecord))
        self.assertEqual(
            layout.field_offsets,
            (_CStandardRecord.tag.offset, _CStandardRecord.value.offset),
        )

    def test_standard_union_matches_c_oracle(self):
        pc_type = union[i32, f64]
        layout = pc_type.get_layout()

        self.assertEqual(layout.allocation_size, ctypes.sizeof(_CStandardUnion))
        self.assertEqual(layout.alignment, ctypes.alignment(_CStandardUnion))
        self.assertEqual(
            ctypes.sizeof(pc_type.get_ctypes_type()),
            ctypes.sizeof(_CStandardUnion),
        )
        self.assertEqual(
            ctypes.alignment(pc_type.get_ctypes_type()),
            ctypes.alignment(_CStandardUnion),
        )

    def test_semantic_struct_layout_matches_llvm_oracle(self):
        pc_type = struct[i8, array[i24, 3], union[i64, i128]]
        layout = pc_type.get_layout()
        llvm_type = pc_type.get_llvm_type()
        target_data = get_target_layout().target_data

        self.assertEqual(
            layout.allocation_size,
            llvm_type.get_abi_size(target_data),
        )
        self.assertEqual(
            layout.alignment,
            llvm_type.get_abi_alignment(target_data),
        )

    def test_all_integer_widths_use_target_allocation(self):
        from pythoc.builtin_entities import types
        target_data = get_target_layout().target_data
        for width in range(1, 129):
            for prefix in ('i', 'u'):
                pc_type = getattr(types, f'{prefix}{width}')
                with self.subTest(type=pc_type):
                    llvm_type = pc_type.get_llvm_type()
                    layout = pc_type.get_layout()
                    self.assertEqual(layout.storage_size, (width + 7) // 8)
                    self.assertEqual(layout.bit_width, width)
                    self.assertEqual(pc_type.get_size_bytes(),
                                     llvm_type.get_abi_size(target_data))
                    self.assertEqual(pc_type.get_alignment(),
                                     llvm_type.get_abi_alignment(target_data))

    def test_multidimensional_and_zero_length_arrays(self):
        target_data = get_target_layout().target_data
        # Exercise the internal empty layout without enabling array[T, 0].
        empty = type('EmptyArray', (array,), {
            'element_type': i64, 'dimensions': (0,),
        })
        for pc_type in (array[i24, 2, 3], empty,
                        union[empty], struct[empty, i8]):
            with self.subTest(type=pc_type):
                llvm_type = pc_type.get_llvm_type()
                self.assertEqual(pc_type.get_size_bytes(),
                                 llvm_type.get_abi_size(target_data))
                self.assertEqual(pc_type.get_alignment(),
                                 llvm_type.get_abi_alignment(target_data))

    def test_vectors_and_union_carriers_match_target(self):
        target_data = get_target_layout().target_data
        for lanes in (3, 4, 8, 16):
            vector = simd[i32, lanes]
            for pc_type in (vector, array[vector, 2], union[vector, i8],
                            struct[i8, union[vector], i64]):
                with self.subTest(type=pc_type):
                    llvm_type = pc_type.get_llvm_type()
                    self.assertEqual(pc_type.get_size_bytes(),
                                     llvm_type.get_abi_size(target_data))
                    self.assertEqual(pc_type.get_alignment(),
                                     llvm_type.get_abi_alignment(target_data))

    def test_qualifiers_and_zero_sized_fields(self):
        self.assertEqual(const[i24].get_layout(), i24.get_layout())
        pc_type = struct[pyconst[17], i8, linear, i64]
        self.assertEqual(pc_type.get_layout().field_offsets, (0, 0, 1, 8))
        self.assertEqual(pc_type.get_size_bytes(), 16)

    def test_decorated_recursive_and_forward_array_layout(self):
        with CompileSession():
            @struct
            class NaturalNode:
                value: i64
                next: ptr['NaturalNode']

            @union
            class NaturalUnion:
                nodes: array['NaturalNode', 2]
                value: i64

            self.assertEqual(NaturalNode.get_size_bytes(), 16)
            self.assertEqual(NaturalUnion.get_size_bytes(), 32)
            self.assertEqual(NaturalUnion.get_alignment(), 8)
            self.assertEqual(ctypes.sizeof(NaturalUnion.get_ctypes_type()), 32)

    def test_abi_queries_preserve_custom_identified_context(self):
        from pythoc.builder.abi.aarch64 import AArch64ABI
        context = ir.Context()
        named = context.get_identified_type('NaturalABIRecord')
        named.set_body(ir.IntType(8), ir.IntType(128))
        outer = ir.LiteralStructType([ir.IntType(8), named])
        target_data = get_target_layout().target_data
        abi = AArch64ABI()
        self.assertEqual(abi.get_type_size(outer),
                         outer.get_abi_size(target_data, context))
        self.assertEqual(abi.get_type_alignment(outer),
                         outer.get_abi_alignment(target_data, context))

    def test_target_configuration_is_shared_with_compiler(self):
        from pythoc.compiler import LLVMCompiler
        layouts = []
        for mode in ('generic', 'native'):
            with config.override(target_cpu=mode, target_features=mode):
                layout = get_target_layout()
                layouts.append(layout)
                module = LLVMCompiler().module
                self.assertEqual(str(module.data_layout), str(layout.target_data))
                self.assertEqual(str(get_target_machine().target_data),
                                 str(layout.target_data))
        self.assertIsNot(layouts[0], layouts[1])

    def test_synthetic_32_bit_data_layout_has_no_host_pointer_assumption(self):
        from llvmlite import binding
        target = TargetLayout(target_data=binding.create_target_data(
            'e-p:32:32-i64:64-n8:16:32-S128'))
        layout = target.primitive(ir.PointerType(ir.IntType(8)))
        self.assertEqual((layout.allocation_size, layout.alignment), (4, 4))

    def test_immutable_primitive_facts_are_queried_only_once(self):
        target = TargetLayout()
        llvm_type = ir.IntType(23)
        with patch.object(llvm_type, 'get_abi_size',
                          wraps=llvm_type.get_abi_size) as size_query:
            for _ in range(5):
                self.assertEqual(target.primitive(llvm_type).allocation_size, 4)
        self.assertEqual(size_query.call_count, 1)

    def test_identified_type_facts_are_not_frozen_in_primitive_cache(self):
        target = TargetLayout()
        context = ir.Context()
        named = context.get_identified_type('UncachedLayoutRecord')
        named.set_body(ir.IntType(64))
        with patch.object(named, 'get_abi_size', side_effect=[8, 16]):
            self.assertEqual(target.primitive(named).allocation_size, 8)
            self.assertEqual(target.primitive(named).allocation_size, 16)

    def test_integer_promotion_uses_value_width_not_allocation(self):
        self.assertFalse(i32.is_compatible_with(i24))
        self.assertTrue(i24.is_compatible_with(i32))

    def test_ctypes_rejects_unrepresentable_storage(self):
        for pc_type in (struct[i8, simd[i32, 4]], union[simd[i32, 4]],
                        array[simd[i32, 4], 2]):
            with self.subTest(type=pc_type):
                with self.assertRaisesRegex(ValueError, 'ctypes cannot represent'):
                    pc_type.get_ctypes_type()

    def test_layout_rejects_invalid_allocation(self):
        for args in ((4, 3, 1, 3), (1, 1, 3, 1), (1, 4, 4, 3),
                     (1, 3, 2, 4), (-1, 0, 1, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                TypeLayout(*args)


if __name__ == "__main__":
    unittest.main()
