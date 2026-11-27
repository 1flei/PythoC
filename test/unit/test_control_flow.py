"""Per-function VReg IDs and LLVM replay order."""

import unittest

from llvmlite import binding, ir

from pythoc.ast_visitor import LLVMIRVisitor
from pythoc.ast_visitor.control_flow_builder import ControlFlowBuilder, _TerminatedExpression
from pythoc.ast_visitor.pcir import merge_values
from pythoc.builder import LLVMBuilder
from pythoc.builtin_entities import i32
from pythoc.valueref import ensure_ir, wrap_value


class TestVRegCounterPerInstance(unittest.TestCase):
    def test_independent_counters(self):
        """Two builders allocate VReg IDs independently."""
        cf_a = ControlFlowBuilder(None, None, "func_a")
        cf_b = ControlFlowBuilder(None, None, "func_b")

        # Interleave allocations: creating cf_b must not reset cf_a's counter
        a0 = cf_a._alloc_vreg_id()
        b0 = cf_b._alloc_vreg_id()
        a1 = cf_a._alloc_vreg_id()
        b1 = cf_b._alloc_vreg_id()

        self.assertEqual((a0, a1), (0, 1))
        self.assertEqual((b0, b1), (0, 1))

    def test_vregs_created_through_builder(self):
        """VRegs created via builder methods use the builder's counter."""
        cf = ControlFlowBuilder(None, None, "f")
        phi1 = cf.phi(ir.IntType(32))
        phi2 = cf.phi(ir.IntType(32))

        self.assertEqual(phi1.id, 0)
        self.assertEqual(phi2.id, 1)
        self.assertEqual(phi1.name, "vreg.phi.0")
        self.assertEqual(phi2.name, "vreg.phi.1")

    def test_ids_unique_within_one_builder(self):
        """All VReg kinds share one counter per function compilation."""
        cf = ControlFlowBuilder(None, None, "f")
        phi = cf.phi(ir.IntType(32))
        sentinel = cf._block_map[cf._current_block_id]
        from pythoc.ast_visitor.pcir import VRegSwitch
        vswitch = VRegSwitch(sentinel, id=cf._alloc_vreg_id())

        self.assertNotEqual(phi.id, vswitch.id)


class TestExpressionContinuation(unittest.TestCase):
    def test_expression_exit_depends_on_its_final_cursor(self):
        builder = ControlFlowBuilder(None, LLVMIRVisitor(module=ir.Module()), 'expression')
        successor = builder.create_block('next')

        def continue_in_successor():
            builder.branch(successor)
            builder.position_at_end(successor)
            return 42

        self.assertEqual(builder.lower_expression(continue_in_successor), 42)
        with self.assertRaises(_TerminatedExpression):
            builder.lower_expression(builder.unreachable)
        self.assertEqual(builder.lower_expression(lambda: 'label'), 'label')


class TestPCIRReplayOrder(unittest.TestCase):
    def test_merge_values_preserves_type_and_only_emits_phi_for_multiple_edges(self):
        module = ir.Module()
        signature = ir.FunctionType(ir.IntType(32), [ir.IntType(1)])
        function = ir.Function(module, signature, 'merge_values')
        entry = function.append_basic_block('entry')
        visitor = LLVMIRVisitor(module=module)
        visitor.current_function = function
        builder = ControlFlowBuilder(LLVMBuilder(ir.IRBuilder(entry)), visitor)
        visitor.builder = builder

        left = builder.create_block('left')
        right = builder.create_block('right')
        join = builder.create_block('join')
        entry_block = builder.block
        builder.cbranch(function.args[0], left, right)

        builder.position_at_end(left)
        first = wrap_value(ir.Constant(ir.IntType(32), 3), kind='value', type_hint=i32)
        single = merge_values(builder, [(first, entry_block)], i32)
        self.assertIs(single.type_hint, i32)
        self.assertIs(ensure_ir(single), ensure_ir(first))
        builder.branch(join)

        builder.position_at_end(right)
        second = ir.Constant(ir.IntType(32), 7)
        builder.branch(join)

        builder.position_at_end(join)
        merged = merge_values(builder, [(single, left), (second, right)], i32)
        self.assertIs(merged.type_hint, i32)
        builder.ret(ensure_ir(merged))
        builder.finalize()
        builder.emit_ir()

        phis = [inst for block in function.blocks
                for inst in block.instructions if inst.opname == 'phi']
        self.assertEqual(phis, [ensure_ir(merged).resolve()])
        binding.parse_assembly(str(module)).verify()

    def test_value_defined_in_later_created_predecessor(self):
        module = ir.Module()
        function = ir.Function(module, ir.FunctionType(ir.IntType(32), []),
                               'scope_result')
        entry = function.append_basic_block('entry')
        visitor = LLVMIRVisitor(module=module)
        visitor.current_function = function
        builder = ControlFlowBuilder(LLVMBuilder(ir.IRBuilder(entry)), visitor)
        visitor.builder = builder
        after = builder.create_block('after')
        body = builder.create_block('body')
        builder.branch(body)
        builder.position_at_end(body)
        value = builder.add(ir.Constant(ir.IntType(32), 40),
                            ir.Constant(ir.IntType(32), 2))
        builder.branch(after)
        builder.position_at_end(after)
        builder.ret(value)
        builder.finalize()
        builder.emit_ir()
        binding.parse_assembly(str(module)).verify()


if __name__ == '__main__':
    unittest.main()
