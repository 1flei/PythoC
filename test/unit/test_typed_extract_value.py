"""PCIR and LLVM agree on a projected field's semantic representation."""

import unittest

from llvmlite import binding, ir

from pythoc import array, i32, ptr
from pythoc.ast_visitor.pcir import infer_result_type
from pythoc.builder.llvm_simple_builder import LLVMBuilder


class TypedExtractValueTest(unittest.TestCase):
    def test_raw_extraction_preserves_physical_storage(self):
        aggregate = ir.Constant(ir.LiteralStructType([ir.PointerType(ir.IntType(8))]), None)
        result = infer_result_type('extract_value', (aggregate, 0), {})
        self.assertEqual(result, ir.PointerType(ir.IntType(8)))

    def test_semantic_hint_controls_pcir_and_replayed_pointer_type(self):
        module = ir.Module()
        storage = ir.LiteralStructType([ir.PointerType(ir.IntType(8))])
        function = ir.Function(module, ir.FunctionType(ir.PointerType(ir.IntType(32)), [storage]), 'project')
        builder = LLVMBuilder(ir.IRBuilder(function.append_basic_block('entry')))
        result = builder.extract_value(function.args[0], 0, value_type_hint=ptr[i32])
        builder.ret(result)
        expected = infer_result_type('extract_value', (function.args[0], 0),
                                     {'value_type_hint': ptr[i32], '_module_context': module.context})
        self.assertEqual(result.type, expected)
        self.assertEqual(result.type, ir.PointerType(ir.IntType(32)))
        binding.parse_assembly(str(module)).verify()

    def test_nested_pointer_arrays_are_represented_consistently(self):
        module = ir.Module()
        physical = ir.ArrayType(ir.PointerType(ir.IntType(8)), 2)
        target = array[ptr[i32], 2].get_llvm_type(module.context)
        storage = ir.LiteralStructType([physical])
        function = ir.Function(module, ir.FunctionType(target, [storage]), 'project_array')
        builder = LLVMBuilder(ir.IRBuilder(function.append_basic_block('entry')))
        result = builder.extract_value(function.args[0], 0, value_type_hint=array[ptr[i32], 2])
        builder.ret(result)
        self.assertEqual(result.type, target)
        binding.parse_assembly(str(module)).verify()


if __name__ == '__main__':
    unittest.main()
