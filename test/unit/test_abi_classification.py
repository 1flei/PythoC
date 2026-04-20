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


if __name__ == "__main__":
    unittest.main()
