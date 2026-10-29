"""Wide-argument bridges preserve void returns and native ABI attributes."""

import unittest

from llvmlite import binding, ir
from pythoc.compiler import _emit_wide_int_ffi_trampoline


class WideIntFFITrampolineTest(unittest.TestCase):
    def test_void_result_uses_a_void_return_instruction(self):
        module = ir.Module()
        fn = ir.Function(module, ir.FunctionType(ir.VoidType(), [ir.IntType(128)]),
                         name='consume_wide')
        ir.IRBuilder(fn.append_basic_block('entry')).ret_void()
        _emit_wide_int_ffi_trampoline(module, fn)
        binding.parse_assembly(str(module)).verify()

    def test_indirect_result_and_byval_attributes_are_forwarded(self):
        module = ir.Module()
        record = ir.LiteralStructType([ir.IntType(64)] * 3)
        fn = ir.Function(module, ir.FunctionType(ir.VoidType(), [
            record.as_pointer(), ir.IntType(128), record.as_pointer(),
        ]), name='make_record')
        fn.args[0].attributes.add('sret')
        fn.args[0].attributes.add('noalias')
        fn.args[0].attributes.align = 8
        fn.args[2].attributes.add('byval')
        fn.args[2].attributes.align = 16
        ir.IRBuilder(fn.append_basic_block('entry')).ret_void()
        _emit_wide_int_ffi_trampoline(module, fn)
        bridge = module.get_global('make_record$pffi')
        call = next(inst for inst in bridge.blocks[0].instructions
                    if isinstance(inst, ir.instructions.CallInstr))
        for index, attribute, alignment in ((0, 'sret', 8), (2, 'byval', 16)):
            self.assertIn(attribute, bridge.args[index].attributes)
            self.assertIn(attribute, call.arg_attributes[index])
            self.assertEqual(bridge.args[index].attributes.align, alignment)
            self.assertEqual(call.arg_attributes[index].align, alignment)
        self.assertIn('noalias', bridge.args[0].attributes)
        self.assertIn('noalias', call.arg_attributes[0])
        binding.parse_assembly(str(module)).verify()

    def test_narrow_return_attributes_are_preserved(self):
        for attribute in ('signext', 'zeroext'):
            with self.subTest(attribute=attribute):
                module = ir.Module()
                fn = ir.Function(module, ir.FunctionType(
                    ir.IntType(8), [ir.IntType(128)]), name='narrow')
                fn.return_value.attributes.add(attribute)
                ir.IRBuilder(fn.append_basic_block('entry')).ret(
                    ir.Constant(ir.IntType(8), 7))
                _emit_wide_int_ffi_trampoline(module, fn)
                bridge = module.get_global('narrow$pffi')
                self.assertIn(attribute, bridge.return_value.attributes)
                call = next(inst for inst in bridge.blocks[0].instructions
                            if isinstance(inst, ir.instructions.CallInstr))
                self.assertIn(attribute, str(call))
                binding.parse_assembly(str(module)).verify()

    def test_wide_result_shifts_only_bridge_parameter_attributes(self):
        module = ir.Module()
        record = ir.LiteralStructType([ir.IntType(64)] * 3)
        fn = ir.Function(module, ir.FunctionType(ir.IntType(128), [
            ir.IntType(128), record.as_pointer()]), name='wide_result')
        fn.args[1].attributes.add('byval')
        fn.args[1].attributes.align = 16
        ir.IRBuilder(fn.append_basic_block('entry')).ret(fn.args[0])
        _emit_wide_int_ffi_trampoline(module, fn)
        bridge = module.get_global('wide_result$pffi')
        call = next(inst for inst in bridge.blocks[0].instructions
                    if isinstance(inst, ir.instructions.CallInstr))
        self.assertIn('byval', bridge.args[2].attributes)
        self.assertEqual(bridge.args[2].attributes.align, 16)
        self.assertIn('byval', call.arg_attributes[1])
        self.assertEqual(call.arg_attributes[1].align, 16)
        self.assertFalse(bridge.args[0].attributes)
        self.assertFalse(bridge.args[1].attributes)
        binding.parse_assembly(str(module)).verify()

    def test_inner_call_preserves_calling_convention_and_return_attribute(self):
        module = ir.Module()
        fn = ir.Function(module, ir.FunctionType(
            ir.IntType(8), [ir.IntType(128)]), name='fast_narrow')
        fn.calling_convention = 'fastcc'
        fn.return_value.attributes.add('signext')
        ir.IRBuilder(fn.append_basic_block('entry')).ret(
            ir.Constant(ir.IntType(8), -7))
        _emit_wide_int_ffi_trampoline(module, fn)
        bridge = module.get_global('fast_narrow$pffi')
        self.assertEqual(bridge.calling_convention, '')
        self.assertIn('call fastcc signext i8', str(bridge))
        binding.parse_assembly(str(module)).verify()


if __name__ == '__main__':
    unittest.main()
