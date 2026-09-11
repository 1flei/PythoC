"""Unit tests for ABI classification -- argument vs return splits."""

import unittest

from llvmlite import ir

from pythoc.builder.abi.base import PassingKind
from pythoc.builder.abi.x86_64 import X86_64ABI


class TestX86_64ArgumentVsReturnClassification(unittest.TestCase):
    """The x86-64 classifier must mark the ``is_return`` flag correctly
    and must not turn an empty-aggregate argument into a ``void`` type.
    """

    def setUp(self):
        # SysV default (16-byte eightbyte threshold).
        self.abi = X86_64ABI()

    # ------------------------------------------------------------------
    # is_return propagation
    # ------------------------------------------------------------------

    def test_non_aggregate_is_return_flag(self):
        r = self.abi.classify_return_type(ir.IntType(32))
        a = self.abi.classify_argument_type(ir.IntType(32))
        self.assertEqual(r.kind, PassingKind.DIRECT)
        self.assertEqual(a.kind, PassingKind.DIRECT)
        self.assertTrue(r.is_return)
        self.assertFalse(a.is_return)

    def test_small_struct_is_return_flag(self):
        # {i32, i32} -> coerce to i64, is_return set respectively.
        small = ir.LiteralStructType([ir.IntType(32), ir.IntType(32)])
        r = self.abi.classify_return_type(small)
        a = self.abi.classify_argument_type(small)
        self.assertEqual(r.kind, PassingKind.COERCE)
        self.assertEqual(a.kind, PassingKind.COERCE)
        self.assertTrue(r.is_return)
        self.assertFalse(a.is_return)

    def test_large_struct_uses_indirect_with_correct_flag(self):
        # 24-byte struct -> INDIRECT: sret for return, byval for argument.
        large = ir.LiteralStructType([
            ir.IntType(64), ir.IntType(64), ir.IntType(64)
        ])
        r = self.abi.classify_return_type(large)
        a = self.abi.classify_argument_type(large)
        self.assertEqual(r.kind, PassingKind.INDIRECT)
        self.assertEqual(a.kind, PassingKind.INDIRECT)
        self.assertTrue(r.is_return)
        self.assertFalse(a.is_return)

    # ------------------------------------------------------------------
    # Empty aggregate: differ between return and argument
    # ------------------------------------------------------------------

    def test_empty_struct_return_becomes_void(self):
        empty = ir.LiteralStructType([])
        r = self.abi.classify_return_type(empty)
        self.assertEqual(r.kind, PassingKind.DIRECT)
        self.assertIsInstance(r.coerced_type, ir.VoidType)
        self.assertTrue(r.is_return)

    def test_empty_struct_argument_stays_direct_without_void(self):
        """Zero-sized argument must NOT be rewritten to ``void`` type --
        a void parameter is illegal in LLVM IR. We leave the aggregate
        untouched and let the call site decide whether to drop it.
        """
        empty = ir.LiteralStructType([])
        a = self.abi.classify_argument_type(empty)
        self.assertEqual(a.kind, PassingKind.DIRECT)
        # Crucially: coerced_type must NOT be a void type.
        self.assertFalse(
            isinstance(a.coerced_type, ir.VoidType),
            msg="empty aggregate argument must not be rewritten to void",
        )
        self.assertFalse(a.is_return)


class TestX86_64WindowsThreshold(unittest.TestCase):
    """Windows x64 uses an 8-byte register threshold and should inherit
    the same is_return handling.
    """

    def setUp(self):
        self.abi = X86_64ABI(max_register_size=8)

    def test_nine_byte_struct_becomes_indirect_on_windows(self):
        # {i64, i8} = 9 bytes > 8 -> INDIRECT under Windows x64 rules.
        nine = ir.LiteralStructType([ir.IntType(64), ir.IntType(8)])
        r = self.abi.classify_return_type(nine)
        a = self.abi.classify_argument_type(nine)
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

    def test_three_byte_struct_is_indirect(self):
        # {i8, i8, i8} = 3 bytes: register-passed on SysV, by-reference on
        # Windows x64.  Getting this wrong makes the C callee dereference
        # the register value as a pointer.
        three = ir.LiteralStructType([ir.IntType(8)] * 3)
        r = self.abi.classify_return_type(three)
        a = self.abi.classify_argument_type(three)
        self.assertEqual(r.kind, PassingKind.INDIRECT)
        self.assertEqual(a.kind, PassingKind.INDIRECT)
        self.assertTrue(r.is_return)
        self.assertFalse(a.is_return)

    def test_pow2_sizes_stay_coerced(self):
        for size, width in ((1, 8), (2, 16), (4, 32), (8, 64)):
            ty = ir.LiteralStructType([ir.IntType(8)] * size)
            r = self.abi.classify_return_type(ty)
            a = self.abi.classify_argument_type(ty)
            self.assertEqual(r.kind, PassingKind.COERCE, f"size={size}")
            self.assertEqual(a.kind, PassingKind.COERCE, f"size={size}")
            self.assertEqual(r.coerced_type, ir.IntType(width))

    def test_sysv_three_byte_struct_unaffected(self):
        # Without the pow2 rule (SysV), a 3-byte struct still coerces to i32.
        sysv = X86_64ABI()
        three = ir.LiteralStructType([ir.IntType(8)] * 3)
        self.assertEqual(
            sysv.classify_argument_type(three).kind, PassingKind.COERCE)
        self.assertEqual(
            sysv.classify_return_type(three).kind, PassingKind.COERCE)

    def test_get_target_abi_sets_pow2_rule_on_windows(self):
        from pythoc.builder.abi import get_target_abi
        abi = get_target_abi('x86_64-pc-windows-gnu')
        self.assertTrue(getattr(abi, '_pow2_register_sizes', False))
        three = ir.LiteralStructType([ir.IntType(8)] * 3)
        self.assertEqual(
            abi.classify_argument_type(three).kind, PassingKind.INDIRECT)
        # Linux x86_64 keeps SysV rules.
        abi_sysv = get_target_abi('x86_64-unknown-linux-gnu')
        self.assertFalse(getattr(abi_sysv, '_pow2_register_sizes', False))
        self.assertEqual(
            abi_sysv.classify_argument_type(three).kind, PassingKind.COERCE)


class TestAArch64HFAPrecedence(unittest.TestCase):
    """AAPCS64 HFA rule takes precedence over the 16-byte indirect rule:
    a homogeneous float aggregate with up to 4 members is register-passed
    even when it is 24 or 32 bytes.
    """

    def setUp(self):
        from pythoc.builder.abi.aarch64 import AArch64ABI
        self.abi = AArch64ABI()

    def test_three_double_hfa_is_register_passed(self):
        # {f64, f64, f64} = 24 bytes but HFA(3) -> [3 x double], not sret.
        hfa3 = ir.LiteralStructType(
            [ir.DoubleType(), ir.DoubleType(), ir.DoubleType()])
        r = self.abi.classify_return_type(hfa3)
        a = self.abi.classify_argument_type(hfa3)
        self.assertEqual(r.kind, PassingKind.COERCE)
        self.assertEqual(a.kind, PassingKind.COERCE)
        self.assertEqual(r.coerced_type, ir.ArrayType(ir.DoubleType(), 3))
        self.assertEqual(a.coerced_type, ir.ArrayType(ir.DoubleType(), 3))
        self.assertTrue(r.is_return)
        self.assertFalse(a.is_return)

    def test_four_double_hfa_is_register_passed(self):
        # {f64 x 4} = 32 bytes but HFA(4) -> [4 x double].
        hfa4 = ir.LiteralStructType([ir.DoubleType()] * 4)
        r = self.abi.classify_return_type(hfa4)
        a = self.abi.classify_argument_type(hfa4)
        self.assertEqual(r.kind, PassingKind.COERCE)
        self.assertEqual(a.kind, PassingKind.COERCE)
        self.assertEqual(r.coerced_type, ir.ArrayType(ir.DoubleType(), 4))

    def test_five_double_aggregate_is_indirect(self):
        # HFA membership is capped at 4: {f64 x 5} -> memory.
        hfa5 = ir.LiteralStructType([ir.DoubleType()] * 5)
        r = self.abi.classify_return_type(hfa5)
        a = self.abi.classify_argument_type(hfa5)
        self.assertEqual(r.kind, PassingKind.INDIRECT)
        self.assertEqual(a.kind, PassingKind.INDIRECT)

    def test_large_integer_struct_stays_indirect(self):
        # {i64, i64, i64} = 24 bytes, not an HFA -> memory (unchanged).
        large = ir.LiteralStructType([ir.IntType(64)] * 3)
        r = self.abi.classify_return_type(large)
        a = self.abi.classify_argument_type(large)
        self.assertEqual(r.kind, PassingKind.INDIRECT)
        self.assertEqual(a.kind, PassingKind.INDIRECT)

    def test_small_hfa_unchanged(self):
        # {f64, f64} = 16 bytes was already register-passed before.
        hfa2 = ir.LiteralStructType([ir.DoubleType(), ir.DoubleType()])
        r = self.abi.classify_return_type(hfa2)
        a = self.abi.classify_argument_type(hfa2)
        self.assertEqual(r.kind, PassingKind.COERCE)
        self.assertEqual(a.kind, PassingKind.COERCE)
        self.assertEqual(r.coerced_type, ir.ArrayType(ir.DoubleType(), 2))

    def test_mixed_float_struct_is_not_hfa(self):
        # {f32, f64} mixes base types -> not an HFA; 16 bytes -> integers.
        mixed = ir.LiteralStructType([ir.FloatType(), ir.DoubleType()])
        r = self.abi.classify_return_type(mixed)
        self.assertEqual(r.kind, PassingKind.COERCE)
        self.assertNotEqual(r.coerced_type, ir.ArrayType(ir.FloatType(), 2))


if __name__ == "__main__":
    unittest.main()
