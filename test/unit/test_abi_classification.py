"""Unit tests for ABI classification -- argument vs return splits."""

import unittest

from llvmlite import ir

from pythoc import array, f16, f32, f64, i8, i32, i64, simd, struct, u64, union
from pythoc.builder.abi.base import PassingKind
from pythoc.builder.abi.x86_64 import X86_64ABI


class TestX86_64ArgumentVsReturnClassification(unittest.TestCase):
    """The x86-64 classifier must mark the ``is_return`` flag correctly
    and must not turn an empty-aggregate argument into a ``void`` type.
    """

    def setUp(self):
        # SysV default (16-byte eightbyte threshold).
        self.abi = X86_64ABI()
        self.context = ir.Context()

    def test_half_pair_uses_one_sse_register(self):
        lane = ir.HalfType()
        pair = struct[f16, f16]
        for classify in (self.abi.classify_argument_type, self.abi.classify_return_type):
            carrier = classify(pair, self.context).abi_type
            self.assertEqual(carrier, ir.VectorType(lane, 2))
            self.assertEqual(self.abi._required_registers(carrier), (0, 1))

    # ------------------------------------------------------------------
    # is_return propagation
    # ------------------------------------------------------------------

    def test_non_aggregate_is_return_flag(self):
        r = self.abi.classify_return_type(i32, self.context)
        a = self.abi.classify_argument_type(i32, self.context)
        self.assertEqual(r.kind, PassingKind.DIRECT)
        self.assertEqual(a.kind, PassingKind.DIRECT)
        self.assertTrue(r.is_return)
        self.assertFalse(a.is_return)

    def test_small_struct_is_return_flag(self):
        # {i32, i32} -> coerce to i64, is_return set respectively.
        small = struct[i32, i32]
        r = self.abi.classify_return_type(small, self.context)
        a = self.abi.classify_argument_type(small, self.context)
        self.assertEqual(r.kind, PassingKind.COERCE)
        self.assertEqual(a.kind, PassingKind.COERCE)
        self.assertTrue(r.is_return)
        self.assertFalse(a.is_return)

    def test_large_struct_uses_indirect_with_correct_flag(self):
        # 24-byte struct -> INDIRECT: sret for return, byval for argument.
        large = struct[i64, i64, i64]
        r = self.abi.classify_return_type(large, self.context)
        a = self.abi.classify_argument_type(large, self.context)
        self.assertEqual(r.kind, PassingKind.INDIRECT)
        self.assertEqual(a.kind, PassingKind.INDIRECT)
        self.assertTrue(r.is_return)
        self.assertFalse(a.is_return)

    # ------------------------------------------------------------------
    # Empty aggregate: differ between return and argument
    # ------------------------------------------------------------------

    def test_empty_struct_return_becomes_void(self):
        r = self.abi.classify_return_type(struct[()], self.context)
        self.assertEqual(r.kind, PassingKind.DIRECT)
        self.assertIsInstance(r.coerced_type, ir.VoidType)
        self.assertTrue(r.is_return)

    def test_empty_struct_argument_stays_direct_without_void(self):
        """Zero-sized argument must NOT be rewritten to ``void`` type --
        a void parameter is illegal in LLVM IR. We leave the aggregate
        untouched and let the call site decide whether to drop it.
        """
        a = self.abi.classify_argument_type(struct[()], self.context)
        self.assertEqual(a.kind, PassingKind.DIRECT)
        # Crucially: coerced_type must NOT be a void type.
        self.assertFalse(
            isinstance(a.coerced_type, ir.VoidType),
            msg="empty aggregate argument must not be rewritten to void",
        )
        self.assertFalse(a.is_return)


    def test_two_integer_eightbytes_require_two_available_registers(self):
        pair = struct[i64, i64]
        pair_storage = pair.get_llvm_type(self.context)
        for preceding in (4, 5):
            with self.subTest(preceding=preceding):
                signature = self.abi.classify_signature(
                    [i64] * preceding + [pair, i64], i64, self.context,
                )
                self.assertEqual(signature.arguments[-2].is_indirect, preceding == 5)
                self.assertEqual(signature.function_type().args[-2],
                                 pair_storage.as_pointer() if preceding == 5 else pair_storage)

    def test_mixed_eightbytes_require_both_register_classes(self):
        mixed = struct[i64, f64]
        for preceding, indirect in (([i64] * 5 + [f64] * 7, False),
                                    ([i64] * 6, True), ([f64] * 8, True)):
            with self.subTest(preceding=preceding):
                signature = self.abi.classify_signature([*preceding, mixed], i64, self.context)
                self.assertEqual(signature.arguments[-1].is_indirect, indirect)

    def test_indirect_result_reserves_a_general_purpose_register(self):
        pair = struct[i64, i64]
        large = struct[i64, i64, i64]
        signature = self.abi.classify_signature([i64] * 4 + [pair], large, self.context)
        self.assertTrue(signature.result.is_indirect)
        self.assertTrue(signature.arguments[-1].is_indirect)
        self.assertEqual(signature.function_type().args[0],
                         large.get_llvm_type(self.context).as_pointer())

    def test_vector_eightbytes_share_one_xmm_register(self):
        vector = ir.VectorType(ir.FloatType(), 4)
        record = struct[struct[simd[f32, 4]]]
        for preceding in (7, 8):
            with self.subTest(preceding=preceding):
                signature = self.abi.classify_signature(
                    [f64] * preceding + [record], record, self.context,
                )
                self.assertEqual(signature.result.abi_type, vector)
                self.assertEqual(signature.arguments[-1].is_indirect, preceding == 8)
                self.assertEqual(signature.arguments[-1].abi_type,
                                 record.get_llvm_type(self.context).as_pointer()
                                 if preceding == 8 else vector)

    def test_two_short_vectors_use_independent_sse_registers(self):
        vector = simd[f32, 2]
        pair = struct[vector, vector]
        signature = self.abi.classify_signature([f64] * 7 + [pair], pair, self.context)
        self.assertEqual(signature.result.abi_type, ir.LiteralStructType([ir.DoubleType()] * 2))
        self.assertTrue(signature.arguments[-1].is_indirect)

    def test_union_merges_vector_upper_half_after_an_integer(self):
        vector = simd[f32, 4]
        data = union[vector, u64]
        expected = ir.LiteralStructType([ir.IntType(64), ir.DoubleType()])
        for classify in (self.abi.classify_argument_type, self.abi.classify_return_type):
            self.assertEqual(classify(data, self.context).abi_type, expected)

    def test_unions_with_same_storage_use_their_declared_member_class(self):
        for classify in (self.abi.classify_argument_type, self.abi.classify_return_type):
            floating = classify(union[f64], self.context)
            integer = classify(union[u64], self.context)
            self.assertEqual(floating.original_type, integer.original_type)
            self.assertEqual(floating.abi_type, ir.DoubleType())
            self.assertEqual(integer.abi_type, ir.IntType(64))

    def test_short_vector_classes_match_platform_and_alignment(self):
        from pythoc.builder.abi import get_target_abi

        boxed_int = struct[simd[i64, 1]]
        darwin = get_target_abi('x86_64-apple-darwin')
        self.assertEqual(self.abi.classify_argument_type(boxed_int, self.context).abi_type,
                         ir.DoubleType())
        self.assertEqual(darwin.classify_argument_type(boxed_int, self.context).abi_type,
                         ir.IntType(64))
        boxed_double = struct[simd[f64, 1]]
        self.assertTrue(self.abi.classify_argument_type(boxed_double, self.context).is_indirect)
        narrow = struct[simd[f32, 1]]
        self.assertEqual(self.abi.classify_return_type(narrow, self.context).abi_type,
                         ir.IntType(32))

        @struct(alignment=8)
        class Unaligned:
            prefix: i8
            value: simd[f32, 2]

        self.assertTrue(self.abi.classify_argument_type(Unaligned, self.context).is_indirect)

    def test_bit_storage_alignment_carrier_is_not_a_simd_member(self):
        @struct(storage_unit=64, abi_alignment=128)
        class Bits:
            value: u64

        expected = ir.LiteralStructType([ir.IntType(64)] * 2)
        self.assertEqual(self.abi.classify_argument_type(Bits, self.context).abi_type, expected)


class TestX86_64WindowsThreshold(unittest.TestCase):
    """Windows x64 uses an 8-byte register threshold and should inherit
    the same is_return handling.
    """

    def setUp(self):
        self.abi = X86_64ABI(max_register_size=8)
        self.context = ir.Context()

    def test_nine_byte_struct_becomes_indirect_on_windows(self):
        # {i64, i8} = 9 bytes > 8 -> INDIRECT under Windows x64 rules.
        nine = struct[i64, i8]
        r = self.abi.classify_return_type(nine, self.context)
        a = self.abi.classify_argument_type(nine, self.context)
        self.assertEqual(r.kind, PassingKind.INDIRECT)
        self.assertEqual(a.kind, PassingKind.INDIRECT)
        self.assertTrue(r.is_return)
        self.assertFalse(a.is_return)


class TestX86_64WindowsPow2Sizes(unittest.TestCase):
    """Windows x64 passes an aggregate by value in an integer register only
    when its size is exactly 1, 2, 4 or 8 bytes; every other size (3, 5, 6,
    7, and > 8) is passed by reference.  SysV has no such restriction.
    """

    def setUp(self):
        self.abi = X86_64ABI(
            max_register_size=8,
            use_byval_for_indirect_args=False,
            pow2_register_sizes=True,
        )
        self.context = ir.Context()

    def test_three_byte_struct_is_indirect(self):
        # {i8, i8, i8} = 3 bytes: register-passed on SysV, by-reference on
        # Windows x64.  Getting this wrong makes the C callee dereference
        # the register value as a pointer.
        three = struct[i8, i8, i8]
        r = self.abi.classify_return_type(three, self.context)
        a = self.abi.classify_argument_type(three, self.context)
        self.assertEqual(r.kind, PassingKind.INDIRECT)
        self.assertEqual(a.kind, PassingKind.INDIRECT)
        self.assertTrue(r.is_return)
        self.assertFalse(a.is_return)

    def test_pow2_sizes_stay_coerced(self):
        for size, width in ((1, 8), (2, 16), (4, 32), (8, 64)):
            ty = struct[(i8,) * size]
            r = self.abi.classify_return_type(ty, self.context)
            a = self.abi.classify_argument_type(ty, self.context)
            self.assertEqual(r.kind, PassingKind.COERCE, f"size={size}")
            self.assertEqual(a.kind, PassingKind.COERCE, f"size={size}")
            self.assertEqual(r.coerced_type, ir.IntType(width))

    def test_float_aggregate_uses_integer_register(self):
        float_record = struct[f32]
        for classify in (self.abi.classify_argument_type, self.abi.classify_return_type):
            self.assertEqual(classify(float_record, self.context).coerced_type, ir.IntType(32))

    def test_sysv_three_byte_struct_unaffected(self):
        # Without the pow2 rule (SysV), a 3-byte struct still coerces to i32.
        sysv = X86_64ABI()
        three = struct[i8, i8, i8]
        self.assertEqual(
            sysv.classify_argument_type(three, self.context).kind, PassingKind.COERCE)
        self.assertEqual(
            sysv.classify_return_type(three, self.context).kind, PassingKind.COERCE)

    def test_get_target_abi_sets_pow2_rule_on_windows(self):
        from pythoc.builder.abi import get_target_abi
        abi = get_target_abi('x86_64-pc-windows-gnu')
        self.assertTrue(getattr(abi, '_pow2_register_sizes', False))
        three = struct[i8, i8, i8]
        self.assertEqual(
            abi.classify_argument_type(three, self.context).kind, PassingKind.INDIRECT)
        # Linux x86_64 keeps SysV rules.
        abi_sysv = get_target_abi('x86_64-unknown-linux-gnu')
        self.assertFalse(getattr(abi_sysv, '_pow2_register_sizes', False))
        self.assertEqual(
            abi_sysv.classify_argument_type(three, self.context).kind, PassingKind.COERCE)


class TestAArch64HFAPrecedence(unittest.TestCase):
    """AAPCS64 HFA rule takes precedence over the 16-byte indirect rule:
    a homogeneous float aggregate with up to 4 members is register-passed
    even when it is 24 or 32 bytes.
    """

    def setUp(self):
        from pythoc.builder.abi.aarch64 import AArch64ABI
        self.abi = AArch64ABI()
        self.context = ir.Context()

    def test_half_pair_is_homogeneous(self):
        lane = ir.HalfType()
        pair = struct[f16, f16]
        for classify in (self.abi.classify_argument_type, self.abi.classify_return_type):
            self.assertEqual(classify(pair, self.context).abi_type, ir.ArrayType(lane, 2))

    def test_three_double_hfa_is_register_passed(self):
        # {f64, f64, f64} = 24 bytes but HFA(3) -> [3 x double], not sret.
        hfa3 = struct[f64, f64, f64]
        r = self.abi.classify_return_type(hfa3, self.context)
        a = self.abi.classify_argument_type(hfa3, self.context)
        self.assertEqual(r.kind, PassingKind.COERCE)
        self.assertEqual(a.kind, PassingKind.COERCE)
        self.assertEqual(r.coerced_type, ir.ArrayType(ir.DoubleType(), 3))
        self.assertEqual(a.coerced_type, ir.ArrayType(ir.DoubleType(), 3))
        self.assertTrue(r.is_return)
        self.assertFalse(a.is_return)

    def test_four_double_hfa_is_register_passed(self):
        # {f64 x 4} = 32 bytes but HFA(4) -> [4 x double].
        hfa4 = struct[(f64,) * 4]
        r = self.abi.classify_return_type(hfa4, self.context)
        a = self.abi.classify_argument_type(hfa4, self.context)
        self.assertEqual(r.kind, PassingKind.COERCE)
        self.assertEqual(a.kind, PassingKind.COERCE)
        self.assertEqual(r.coerced_type, ir.ArrayType(ir.DoubleType(), 4))

    def test_five_double_aggregate_is_indirect(self):
        # HFA membership is capped at 4: {f64 x 5} -> memory.
        hfa5 = struct[(f64,) * 5]
        r = self.abi.classify_return_type(hfa5, self.context)
        a = self.abi.classify_argument_type(hfa5, self.context)
        self.assertEqual(r.kind, PassingKind.INDIRECT)
        self.assertEqual(a.kind, PassingKind.INDIRECT)

    def test_large_integer_struct_stays_indirect(self):
        # {i64, i64, i64} = 24 bytes, not an HFA -> memory (unchanged).
        large = struct[i64, i64, i64]
        r = self.abi.classify_return_type(large, self.context)
        a = self.abi.classify_argument_type(large, self.context)
        self.assertEqual(r.kind, PassingKind.INDIRECT)
        self.assertEqual(a.kind, PassingKind.INDIRECT)

    def test_small_hfa_unchanged(self):
        # {f64, f64} = 16 bytes was already register-passed before.
        hfa2 = struct[f64, f64]
        r = self.abi.classify_return_type(hfa2, self.context)
        a = self.abi.classify_argument_type(hfa2, self.context)
        self.assertEqual(r.kind, PassingKind.COERCE)
        self.assertEqual(a.kind, PassingKind.COERCE)
        self.assertEqual(r.coerced_type, ir.ArrayType(ir.DoubleType(), 2))

    def test_mixed_float_struct_is_not_hfa(self):
        # {f32, f64} mixes base types -> not an HFA; 16 bytes -> integers.
        mixed = struct[f32, f64]
        r = self.abi.classify_return_type(mixed, self.context)
        self.assertEqual(r.kind, PassingKind.COERCE)
        self.assertNotEqual(r.coerced_type, ir.ArrayType(ir.FloatType(), 2))

    def test_explicit_record_alignment_affects_argument_carrier_by_platform(self):
        from pythoc.builder.abi import get_target_abi

        @struct(abi_alignment=128)
        class AlignedFloat:
            value: f32

        linux = get_target_abi('aarch64-unknown-linux-gnu')
        darwin = get_target_abi('arm64-apple-darwin')
        self.assertEqual(linux.classify_argument_type(AlignedFloat, self.context).coerced_type,
                         ir.ArrayType(ir.IntType(64), 2))
        self.assertEqual(darwin.classify_argument_type(AlignedFloat, self.context).coerced_type,
                         ir.IntType(128))
        self.assertEqual(linux.classify_return_type(AlignedFloat, self.context).coerced_type,
                         ir.IntType(128))

    def test_homogeneous_short_vectors_share_width_across_lane_types(self):
        f32x4, i32x4 = simd[f32, 4], simd[i32, 4]
        aggregate = struct[f32x4, i32x4, f32x4, i32x4]
        carrier = ir.ArrayType(ir.VectorType(ir.IntType(32), 4), 4)
        self.assertEqual(self.abi.classify_argument_type(aggregate, self.context).abi_type, carrier)
        self.assertEqual(self.abi.classify_return_type(aggregate, self.context).abi_type, carrier)
        too_many = array[f32x4, 5]
        self.assertTrue(self.abi.classify_argument_type(too_many, self.context).is_indirect)


if __name__ == "__main__":
    unittest.main()
