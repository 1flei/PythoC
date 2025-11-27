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
    
    def test_context_has_variable_registry(self):
        """Test context has variable registry"""
        self.assertTrue(hasattr(self.context, 'var_registry'))
        self.assertIsNotNone(self.context.var_registry)
    
    def test_context_scope_management(self):
        """Test scope management"""
        initial_level = self.context.var_registry._scope_level
        
        self.context.var_registry.enter_scope()
        self.assertEqual(self.context.var_registry._scope_level, initial_level + 1)
        
        self.context.var_registry.exit_scope()
        self.assertEqual(self.context.var_registry._scope_level, initial_level)
    
    def test_context_label_generation(self):
        """Test unique label generation"""
        label1 = self.context.get_next_label("test")
        label2 = self.context.get_next_label("test")
        
        self.assertNotEqual(label1, label2)
        self.assertTrue(label1.startswith("test"))
        self.assertTrue(label2.startswith("test"))


if __name__ == '__main__':
    unittest.main()
