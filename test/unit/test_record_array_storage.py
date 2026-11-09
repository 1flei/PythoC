"""Record storage resolves array elements through the array semantic API."""

import unittest

from llvmlite import ir

from pythoc import array, compile, i32, ptr, u16
from pythoc.builder.record_types import storage_type
from pythoc.forward_ref import mark_type_defined


@compile
class RecordArrayStorageElement:
    next: ptr['RecordArrayStorageElement']
    value: i32


class RecordArrayStorageTest(unittest.TestCase):
    def test_forward_record_elements_preserve_all_dimensions(self):
        context = ir.Context()
        actual = storage_type(array['RecordArrayStorageElement', 2, 3], context)
        expected = array[RecordArrayStorageElement, 2, 3].get_llvm_type(context)
        self.assertEqual(actual, expected)

    def test_forward_array_aliases_are_resolved_recursively(self):
        mark_type_defined('RecordStorageRow', array[u16, 3])
        actual = storage_type(array['RecordStorageRow', 2], ir.Context())
        self.assertEqual(actual, ir.ArrayType(ir.ArrayType(ir.IntType(16), 3), 2))

    def test_pointer_elements_do_not_require_complete_pointees(self):
        actual = storage_type(array[ptr['RecordStorageIncomplete'], 2], ir.Context())
        self.assertEqual(actual, ir.ArrayType(ir.PointerType(ir.IntType(8)), 2))


if __name__ == '__main__':
    unittest.main()
