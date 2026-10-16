"""Array stores must copy values without dereferencing pointer variables."""

import ast
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from llvmlite import binding, ir

from pythoc import array, const, i32, ptr, volatile
from pythoc.ast_visitor.assignments import AssignmentsMixin
from pythoc.logger import logger
from pythoc.valueref import wrap_value


class TestArrayReassignment(unittest.TestCase):
    def setUp(self):
        self.module = ir.Module()
        function = ir.Function(
            self.module, ir.FunctionType(ir.VoidType(), []), "store_array"
        )
        self.builder = ir.IRBuilder(function.append_basic_block("entry"))
        self.visitor = AssignmentsMixin()
        self.visitor.builder = self.builder
        self.visitor.implicit_coercer = SimpleNamespace(
            coerce=lambda value, target_type, node: value
        )
        self.array_type = array[i32, 2]
        self.llvm_type = self.array_type.get_llvm_type()
        self.target = self.builder.alloca(self.llvm_type, name="target")
        self.source = self.builder.alloca(self.llvm_type, name="source")

    def store(self, source, source_type=None, target_type=None, target=None):
        self.visitor._store_to_lvalue(
            wrap_value(
                target or self.target,
                kind="address",
                type_hint=target_type or self.array_type,
                address=target or self.target,
            ),
            wrap_value(
                source,
                kind="value",
                type_hint=source_type or self.array_type,
            ),
        )

    def verify_module(self):
        self.builder.ret_void()
        binding.parse_assembly(str(self.module)).verify()

    def test_pointer_backed_array_copies_the_aggregate(self):
        self.store(self.source)
        load, store = self.builder.block.instructions[-2:]
        self.assertEqual(load.opname, "load")
        self.assertEqual(load.type, self.llvm_type)
        self.assertIs(store.operands[0], load)
        self.assertIs(store.operands[1], self.target)
        self.verify_module()

    def test_array_constant_does_not_need_a_load(self):
        value = ir.Constant(self.llvm_type, [7, 11])
        self.store(value)
        self.assertIs(self.builder.block.instructions[-1].operands[0], value)
        self.assertFalse(any(
            instruction.opname == "load"
            for instruction in self.builder.block.instructions
        ))
        self.verify_module()

    def test_pointer_to_array_assignment_preserves_the_pointer(self):
        pointer_slot = self.builder.alloca(self.source.type, name="pointer")
        self.store(
            self.source,
            source_type=ptr[self.array_type],
            target_type=ptr[self.array_type],
            target=pointer_slot,
        )
        self.assertIs(
            self.builder.block.instructions[-1].operands[0], self.source
        )
        self.verify_module()

    def test_array_copy_preserves_source_volatile_load(self):
        self.store(self.source, source_type=volatile[self.array_type])
        self.assertIn("load volatile", str(self.module))
        self.verify_module()

    def test_array_copy_preserves_target_volatile_store(self):
        self.store(self.source, target_type=volatile[self.array_type])
        self.assertIn("store volatile", str(self.module))
        self.verify_module()

    def test_array_copy_rejects_const_target(self):
        with patch.object(logger, "raise_on_error", True):
            with self.assertRaisesRegex(
                RuntimeError, "Cannot reassign to const"
            ):
                self.store(self.source, target_type=const[self.array_type])

    def test_incompatible_array_storage_is_rejected(self):
        other = self.builder.alloca(ir.ArrayType(ir.IntType(32), 3))
        with self.assertRaises(TypeError):
            self.store(other)

    def test_declaration_copies_pointer_backed_arrays(self):
        self.visitor.module = self.module
        self.visitor._create_alloca_in_entry = (
            lambda llvm_type, name: self.builder.alloca(llvm_type, name=name)
        )
        declarations = []
        self.visitor.declare_variable = lambda **kwargs: declarations.append(
            kwargs
        )
        self.visitor._store_to_new_lvalue(
            ast.parse("value = 0").body[0],
            "value",
            self.array_type,
            wrap_value(self.source, kind="value", type_hint=self.array_type),
        )
        self.assertEqual(len(declarations), 1)
        self.assertEqual(
            self.builder.block.instructions[-1].operands[0].type,
            self.llvm_type,
        )
        self.verify_module()


if __name__ == "__main__":
    unittest.main()
