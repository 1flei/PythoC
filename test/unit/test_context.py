"""
Unit tests for context module
"""

import unittest
from llvmlite import ir

from pythoc.context import CompilationContext


class TestCompilationContext(unittest.TestCase):
    """Test CompilationContext functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.module = ir.Module(name="test_module")
        func_type = ir.FunctionType(ir.VoidType(), [])
        func = ir.Function(self.module, func_type, name="test_func")
        block = func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)
        self.context = CompilationContext(self.module, self.builder)

    def test_context_creation(self):
        """Test basic context creation"""
        self.assertIsNotNone(self.context)
        self.assertEqual(self.context.module, self.module)
        self.assertEqual(self.context.builder, self.builder)

    def test_context_has_scope_manager(self):
        """Test context has scope manager"""
        self.assertTrue(hasattr(self.context, 'scope_manager'))
        self.assertIsNotNone(self.context.scope_manager)

    def test_context_scope_management(self):
        """Test scope management via scope_manager"""
        sm = self.context.scope_manager
        initial_depth = sm.current_depth

        from pythoc.scope_manager import ScopeType
        sm.enter_scope(ScopeType.FUNCTION)
        self.assertEqual(sm.current_depth, initial_depth + 1)

    def test_context_label_generation(self):
        """Test unique label generation"""
        label1 = self.context.get_next_label("test")
        label2 = self.context.get_next_label("test")

        self.assertNotEqual(label1, label2)
        self.assertTrue(label1.startswith("test"))
        self.assertTrue(label2.startswith("test"))


if __name__ == '__main__':
    unittest.main()
