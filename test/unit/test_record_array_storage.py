"""Record storage resolves array elements through the array semantic API."""

import unittest

from llvmlite import ir

from pythoc import array, compile, i32, ptr, u16
from pythoc.builtin_entities.struct import create_struct_type
from pythoc.forward_ref import mark_type_defined


@compile
class RecordArrayStorageElement:
    next: ptr['RecordArrayStorageElement']
    value: i32


class RecordArrayStorageTest(unittest.TestCase):
    def test_forward_record_elements_preserve_all_dimensions(self):
        context = ir.Context()
        holder = create_struct_type(
            [array['RecordArrayStorageElement', 2, 3]], ['values'],
            python_class=type('ArrayHolder', (), {}))
        actual = holder.get_llvm_type(context).elements[0]
        expected = array[RecordArrayStorageElement, 2, 3].get_llvm_type(context)
        self.assertEqual(actual, expected)

    def test_forward_array_aliases_are_resolved_recursively(self):
        mark_type_defined('RecordStorageRow', array[u16, 3])
        holder = create_struct_type(
            [array['RecordStorageRow', 2]], ['values'],
            python_class=type('ArrayAliasHolder', (), {}))
        actual = holder.get_llvm_type(ir.Context()).elements[0]
        self.assertEqual(actual, ir.ArrayType(ir.ArrayType(ir.IntType(16), 3), 2))

    def test_pointer_elements_do_not_require_complete_pointees(self):
        holder = create_struct_type(
            [array[ptr['RecordStorageIncomplete'], 2]], ['values'],
            python_class=type('IncompleteArrayHolder', (), {}))
        actual = holder.get_llvm_type(ir.Context()).elements[0]
        self.assertEqual(actual, ir.ArrayType(ir.PointerType(ir.IntType(8)), 2))


if __name__ == '__main__':
    unittest.main()
