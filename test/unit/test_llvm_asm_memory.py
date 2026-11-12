"""Asm memory metadata follows semantic storage through backend emission."""

import unittest

from llvmlite import binding, ir

from pythoc import compile, i32, llvm_asm, ptr, void
from pythoc.build.output_manager import flush_all_pending_outputs
from pythoc.builder.llvm_simple_builder import LLVMBuilder
from pythoc.session import CompileSession
from test.utils.test_utils import DeferredTestCase


class AsmMemoryValidationTest(DeferredTestCase):
    def test_element_type_is_independent_of_pointer_storage(self):
        module = ir.Module()
        byte_pointer = ir.PointerType()
        function = ir.Function(module, ir.FunctionType(ir.VoidType(), [byte_pointer]),
                               name='store')
        builder = LLVMBuilder(ir.IRBuilder(function.append_basic_block('entry')))
        assembly = ir.InlineAsm(ir.FunctionType(ir.VoidType(), [byte_pointer]),
                                '', '=*m', side_effect=True)
        builder.call_inline_asm(assembly, [function.args[0]],
                                element_types={0: ir.IntType(16)})
        builder.ret_void()
        text = str(module)
        self.assertIn('elementtype(i16)', text)

    def test_typed_memory_operand_produces_valid_ir(self):
        module = ir.Module()
        element_type = ir.IntType(16)
        pointer_type = ir.PointerType(element_type)
        function = ir.Function(
            module, ir.FunctionType(ir.VoidType(), [pointer_type]), name='store')
        builder = LLVMBuilder(ir.IRBuilder(function.append_basic_block('entry')))
        assembly = ir.InlineAsm(
            ir.FunctionType(ir.VoidType(), [pointer_type]), '', '=*m', side_effect=True)
        builder.call_inline_asm(
            assembly, [function.args[0]], element_types={0: element_type})
        builder.ret_void()
        text = str(module)
        self.assertIn('elementtype(i16)', text)
        binding.parse_assembly(text).verify()

    def test_indirect_scalar_requires_pointer(self):
        with CompileSession(), self.assertRaisesRegex(TypeError, 'require a typed pointer'):
            @compile(suffix='asm_indirect_scalar')
            def invalid(value: i32) -> None:
                llvm_asm('', '*m', inputs=[value])

            flush_all_pending_outputs()

    def test_void_pointer_has_no_memory_access_type(self):
        with CompileSession(), self.assertRaisesRegex(TypeError, 'complete pointee type'):
            @compile(suffix='asm_indirect_void')
            def invalid(value: ptr[void]) -> None:
                llvm_asm('', '*m', inputs=[value])

            flush_all_pending_outputs()

    def test_constraint_operand_count_is_checked(self):
        with CompileSession(), self.assertRaisesRegex(TypeError, 'require more operands'):
            @compile(suffix='asm_missing_operand')
            def invalid(value: i32) -> None:
                llvm_asm('', 'r,r', inputs=[value])

            flush_all_pending_outputs()

    def test_unused_operands_are_rejected(self):
        with CompileSession(), self.assertRaisesRegex(TypeError, 'exceed constraint operands'):
            @compile(suffix='asm_extra_operand')
            def invalid(value: i32) -> None:
                llvm_asm('', '', inputs=[value])

            flush_all_pending_outputs()

    def test_incomplete_pointee_has_no_memory_access_type(self):
        with CompileSession(), self.assertRaisesRegex(TypeError, 'complete pointee type'):
            @compile(suffix='asm_incomplete_operand')
            def invalid(value: ptr['AsmUnknownPointee']) -> None:
                llvm_asm('', '*m', inputs=[value])

            flush_all_pending_outputs()


if __name__ == '__main__':
    unittest.main()
