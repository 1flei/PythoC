"""Quoted pointees use the same subtraction stride as their concrete type."""

import unittest

from pythoc import array, compile, i8, i32, i64, ptr, void
from pythoc.build.output_manager import flush_all_pending_outputs
from pythoc.session import CompileSession
from test.utils.test_utils import DeferredTestCase
from fixtures.quoted_pointer_difference.model import PointerDiffRecord as ImportedRecord


@compile
def element_difference(begin: ptr['PointerDiffRecord'],
                       end: ptr['PointerDiffRecord']) -> i64:
    return end - begin


@compile
def record_difference(direction: i32) -> i64:
    values: array[ImportedRecord, 4]
    begin: ptr['PointerDiffRecord'] = ptr(values[0])
    end: ptr['PointerDiffRecord'] = ptr(values[3])
    if direction > 1:
        return element_difference(begin, end)
    if direction < 0:
        return begin - end
    if direction == 0:
        return begin - begin
    return end - begin


@compile
def byte_difference() -> i64:
    values: array[i8, 4]
    begin: ptr[void] = ptr[void](ptr(values[0]))
    end: ptr[void] = ptr[void](ptr(values[3]))
    return end - begin


class QuotedPointerDifferenceTest(DeferredTestCase):
    def test_positive_element_difference(self):
        self.assertEqual(record_difference(1), 3)

    def test_negative_element_difference(self):
        self.assertEqual(record_difference(-1), -3)

    def test_equal_pointers(self):
        self.assertEqual(record_difference(0), 0)

    def test_parameter_and_local_types_have_the_same_stride(self):
        self.assertEqual(record_difference(2), 3)

    def test_void_pointers_use_byte_stride(self):
        self.assertEqual(byte_difference(), 3)

    def test_incomplete_pointee_is_rejected(self):
        with CompileSession(), self.assertRaisesRegex(
                TypeError, 'complete pointee type'):
            @compile
            def invalid_difference() -> i64:
                Pointer = ptr['UnresolvedPointerDifference']
                begin: Pointer = Pointer(0)
                end: Pointer = Pointer(16)
                return end - begin

            flush_all_pending_outputs()


if __name__ == '__main__':
    unittest.main()
