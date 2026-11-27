"""Trap lowering emits a terminating LLVM intrinsic."""

import unittest

from llvmlite import binding, ir

import pythoc
from pythoc.ast_visitor import LLVMIRVisitor
from pythoc.ast_visitor.control_flow_builder import ControlFlowBuilder
from pythoc.builder import LLVMSimpleBuilder
from pythoc.registry import get_builtin_entity


class TrapTest(unittest.TestCase):
    def test_trap_is_internal_to_lowering(self):
        self.assertFalse(hasattr(pythoc, 'trap'))
        self.assertIsNone(get_builtin_entity('trap'))

    def test_trap_terminates_the_cfg_and_lowers_in_the_builder(self):
        module = ir.Module()
        function = ir.Function(module, ir.FunctionType(ir.IntType(32), []), 'fail')
        block = function.append_basic_block('entry')
        visitor = LLVMIRVisitor(module=module)
        visitor.current_function = function
        builder = ControlFlowBuilder(LLVMSimpleBuilder(ir.IRBuilder(block)), visitor)
        visitor.builder = builder

        builder.trap()
        self.assertTrue(builder.is_terminated())
        self.assertNotIn('llvm.trap', module.globals)
        builder.finalize(return_type=pythoc.i32)
        self.assertFalse(builder.cfg.get_successors(builder.cfg.entry_id))

        builder.emit_ir()
        self.assertTrue(block.is_terminated)
        self.assertEqual(block.terminator.opname, 'unreachable')
        self.assertIn('call void @"llvm.trap"()', str(module))
        binding.parse_assembly(str(module)).verify()


if __name__ == '__main__':
    unittest.main()
