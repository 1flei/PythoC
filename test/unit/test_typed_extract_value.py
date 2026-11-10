"""Aggregate operations preserve the types fixed by lowering."""

import unittest

from llvmlite import binding, ir

from pythoc import array, i32, ptr
from pythoc.ast_visitor.pcir import infer_result_type
from pythoc.builder.llvm_simple_builder import LLVMBuilder
from pythoc.builtin_entities.struct import create_struct_type


class TypedExtractValueTest(unittest.TestCase):
    def test_raw_extraction_preserves_physical_storage(self):
        aggregate = ir.Constant(ir.LiteralStructType([ir.PointerType(ir.IntType(8))]), None)
        result = infer_result_type('extract_value', (aggregate, 0), {})
        self.assertEqual(result, ir.PointerType(ir.IntType(8)))

    def test_record_field_type_controls_pcir_and_replayed_pointer_type(self):
        module = ir.Module(context=ir.Context())
        holder = create_struct_type(
            [ptr[i32]], ['value'], python_class=type('PointerHolder', (), {}))
        storage = holder.get_llvm_type(module.context)
        function = ir.Function(module, ir.FunctionType(ir.PointerType(ir.IntType(32)), [storage]), 'project')
        builder = LLVMBuilder(ir.IRBuilder(function.append_basic_block('entry')))
        result = builder.extract_value(function.args[0], 0)
        builder.ret(result)
        expected = infer_result_type('extract_value', (function.args[0], 0), {})
        self.assertEqual(result.type, expected)
        self.assertEqual(result.type, ir.PointerType(ir.IntType(32)))
        self.assertEqual([instruction.opname for instruction in builder.block.instructions],
                         ['extractvalue', 'ret'])
        binding.parse_assembly(str(module)).verify()

    def test_nested_pointer_arrays_are_represented_consistently(self):
        module = ir.Module(context=ir.Context())
        target = array[ptr[i32], 2].get_llvm_type(module.context)
        holder = create_struct_type(
            [array[ptr[i32], 2]], ['values'],
            python_class=type('PointerArrayHolder', (), {}))
        storage = holder.get_llvm_type(module.context)
        function = ir.Function(module, ir.FunctionType(target, [storage]), 'project_array')
        builder = LLVMBuilder(ir.IRBuilder(function.append_basic_block('entry')))
        result = builder.extract_value(function.args[0], 0)
        builder.ret(result)
        self.assertEqual(result.type, target)
        self.assertEqual(infer_result_type('extract_value', (function.args[0], 0), {}),
                         target)
        self.assertEqual([instruction.opname for instruction in builder.block.instructions],
                         ['extractvalue', 'ret'])
        binding.parse_assembly(str(module)).verify()

    def test_insert_does_not_implicitly_convert_pointer_fields(self):
        module = ir.Module()
        function = ir.Function(module, ir.FunctionType(ir.VoidType(), []), 'insert')
        builder = LLVMBuilder(ir.IRBuilder(function.append_basic_block('entry')))
        source = ir.Constant(ir.PointerType(ir.IntType(8)), None)
        target = ir.LiteralStructType([ir.PointerType(ir.IntType(32))])
        with self.assertRaises(TypeError):
            builder.insert_value(ir.Constant(target, ir.Undefined), source, 0)
        self.assertEqual(list(builder.block.instructions), [])

    def test_insert_does_not_rebuild_mismatched_pointer_arrays(self):
        module = ir.Module()
        function = ir.Function(module, ir.FunctionType(ir.VoidType(), []), 'insert_array')
        builder = LLVMBuilder(ir.IRBuilder(function.append_basic_block('entry')))
        source = ir.Constant(ir.ArrayType(ir.PointerType(ir.IntType(8)), 2), None)
        field = ir.ArrayType(ir.PointerType(ir.IntType(32)), 2)
        target = ir.LiteralStructType([field])
        with self.assertRaises(TypeError):
            builder.insert_value(ir.Constant(target, ir.Undefined), source, 0)
        self.assertEqual(list(builder.block.instructions), [])


if __name__ == '__main__':
    unittest.main()
