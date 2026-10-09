"""Unit tests for pointer DWARF types with an unspecified pointee.

A pointer whose pointee has no DWARF type of its own (e.g. ptr[void])
must still emit a well-formed DIDerivedType with a baseType; a shared
'void' DIBasicType is used as the base.  Before this was added, the
pointer type was emitted without any baseType, producing malformed
DWARF that debuggers cannot follow.
"""

import unittest

from llvmlite import ir

from pythoc import i32, ptr, void
from pythoc.debug_info import DebugInfoBuilder


class TestPointerDebugTypeBase(unittest.TestCase):
    def _build(self, pc_type, builder=None, module=None):
        if module is None:
            module = ir.Module('m')
        if builder is None:
            builder = DebugInfoBuilder(module, 'test.c')
        builder._build_pointer_debug_type(pc_type)
        return str(module), builder

    @staticmethod
    def _pointer_lines(text):
        return [
            line for line in text.splitlines()
            if 'DW_TAG_pointer_type' in line
        ]

    def test_void_pointee_gets_void_base_type(self):
        text, _ = self._build(ptr[void])
        pointer_lines = self._pointer_lines(text)
        self.assertTrue(pointer_lines, 'no pointer debug type emitted')
        for line in pointer_lines:
            self.assertIn('baseType:', line)
        self.assertIn('DIBasicType(encoding: 7, name: "void"', text)

    def test_void_base_type_is_cached(self):
        module = ir.Module('m')
        builder = DebugInfoBuilder(module, 'test.c')
        builder._build_pointer_debug_type(ptr[void])
        builder._build_pointer_debug_type(ptr[void])
        text = str(module)
        self.assertEqual(text.count('name: "void"'), 1)

    def test_named_pointee_still_used(self):
        text, _ = self._build(ptr[i32])
        pointer_lines = self._pointer_lines(text)
        self.assertTrue(pointer_lines, 'no pointer debug type emitted')
        for line in pointer_lines:
            self.assertIn('baseType:', line)
        self.assertIn('name: "i32"', text)


if __name__ == '__main__':
    unittest.main()
