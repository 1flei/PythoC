"""Inline environments must not reuse bindings from a different factory."""

import unittest

from pythoc import i32, i64
from pythoc._inline.kernel import InlineResult, inline_globals_scope
from pythoc.ast_visitor.base import LLVMIRVisitor


class GlobalBindingScopeTest(unittest.TestCase):
    def setUp(self):
        self.visitor = LLVMIRVisitor(user_globals={'Payload': i32})

    def payload(self):
        binding = self.visitor.lookup_variable('Payload')
        return binding.value_ref.get_python_value()

    def test_nested_factories_restore_their_type_binding(self):
        self.assertIs(self.payload(), i32)
        with inline_globals_scope(
                self.visitor, InlineResult([], {'Payload': i64})):
            self.assertIs(self.payload(), i64)
            with inline_globals_scope(
                    self.visitor, InlineResult([], {'Payload': i32})):
                self.assertIs(self.payload(), i32)
            self.assertIs(self.payload(), i64)
        self.assertIs(self.payload(), i32)

    def test_failed_expansion_restores_the_caller_binding(self):
        self.assertIs(self.payload(), i32)
        with self.assertRaisesRegex(ValueError, 'expansion failed'):
            with inline_globals_scope(
                    self.visitor, InlineResult([], {'Payload': i64})):
                self.assertIs(self.payload(), i64)
                raise ValueError('expansion failed')
        self.assertIs(self.payload(), i32)


if __name__ == '__main__':
    unittest.main()
