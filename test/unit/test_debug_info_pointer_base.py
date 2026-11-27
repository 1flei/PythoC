"""Unit tests for pointer DWARF types with an unspecified pointee.

A pointer whose pointee has no DWARF type of its own (e.g. ptr[void])
must still emit a well-formed DIDerivedType with a baseType; a shared
'void' DIBasicType is used as the base.  Before this was added, the
pointer type was emitted without any baseType, producing malformed
DWARF that debuggers cannot follow.
"""

import unittest
from unittest.mock import patch

from llvmlite import ir

from pythoc import i32, ptr, simd, void
from pythoc.compiler import LLVMCompiler
from pythoc.debug_info import DebugInfoBuilder


class TestPointerDebugTypeBase(unittest.TestCase):
    def test_native_module_interns_metadata_references_without_hashing_children(self):
        module = LLVMCompiler().module
        leaf = module.add_debug_info('DIBasicType', {'name': 'leaf', 'size': 32})
        with patch.object(ir.MDValue, '__hash__', side_effect=AssertionError('hashed tuple')), \
             patch.object(ir.DIValue, '__hash__', side_effect=AssertionError('hashed descriptor')):
            for _ in range(48):
                pair = module.add_metadata([leaf, leaf])
                self.assertIs(module.add_metadata([leaf, leaf]), pair)
                leaf = module.add_debug_info('DISubroutineType', {'types': pair})
                self.assertIs(module.add_debug_info('DISubroutineType', {'types': pair}), leaf)

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

    def test_unmapped_local_gets_an_unspecified_type(self):
        module = ir.Module('m')
        builder = DebugInfoBuilder(module, 'test.c')
        function = ir.Function(module, ir.FunctionType(ir.VoidType(), []), 'test')
        local = builder.declare_local(
            builder.get_subprogram('test', function, 1), 'vector', simd[i32, 4], 2)
        debug_type = dict(local.operands)['type']
        self.assertIn('DW_TAG_unspecified_type', str(debug_type))

    def test_unmapped_member_has_a_base_type(self):
        module = ir.Module('m')
        builder = DebugInfoBuilder(module, 'test.c')
        member = builder._build_member_type('vector', simd[i32, 4], 0)
        debug_type = dict(member.operands)['baseType']
        self.assertIn('DW_TAG_unspecified_type', str(debug_type))


if __name__ == '__main__':
    unittest.main()
