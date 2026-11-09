"""Bit-field programs reject address taking, offsetof, and const writes."""

import unittest

from pythoc import compile, const, i64, offsetof, ptr, struct, u3, u8, union
from pythoc.build.output_manager import flush_all_pending_outputs
from pythoc.session import CompileSession
from test.utils.test_utils import DeferredTestCase


@struct(storage_unit=8)
class Bits:
    a: u3
    b: u3


@union(storage_unit=8)
class Overlap:
    a: u3
    b: u8


class BitfieldInvalidTest(DeferredTestCase):
    def test_rejects_address_attribute(self):
        with CompileSession(), self.assertRaisesRegex(
                TypeError, 'requires an addressable value'):
            @compile
            def invalid() -> i64:
                value: Bits = (1, 2)
                return i64(ptr(value.a))

            flush_all_pending_outputs()

    def test_rejects_address_subscript(self):
        with CompileSession(), self.assertRaisesRegex(
                TypeError, 'requires an addressable value'):
            @compile
            def invalid() -> i64:
                value: Bits = (1, 2)
                return i64(ptr(value[0]))

            flush_all_pending_outputs()

    def test_rejects_offsetof_struct(self):
        with CompileSession(), self.assertRaisesRegex(
                TypeError, 'cannot be applied to a bit field'):
            @compile
            def invalid() -> i64:
                return offsetof(Bits, 'a')

            flush_all_pending_outputs()

    def test_rejects_offsetof_union(self):
        with CompileSession(), self.assertRaisesRegex(
                TypeError, 'cannot be applied to a bit field'):
            @compile
            def invalid() -> i64:
                return offsetof(Overlap, 'b')

            flush_all_pending_outputs()

    def test_rejects_const_attribute(self):
        with CompileSession(), self.assertRaisesRegex(
                RuntimeError, 'Cannot reassign to const'):
            @compile
            def invalid() -> i64:
                value: const[Bits] = (1, 2)
                value.a = u3(3)
                return 0

            flush_all_pending_outputs()

    def test_rejects_const_subscript(self):
        with CompileSession(), self.assertRaisesRegex(
                RuntimeError, 'Cannot reassign to const'):
            @compile
            def invalid() -> i64:
                value: Overlap = Overlap()
                frozen: ptr[const[Overlap]] = ptr(value)
                frozen[0][0] = u3(3)
                return 0

            flush_all_pending_outputs()


if __name__ == '__main__':
    unittest.main()
