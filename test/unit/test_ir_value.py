"""
Unit tests for ir_value module (ValueRef wrapper)
"""

import unittest
from llvmlite import ir
import ast

from pythoc.valueref import (
    ValueRef, ensure_ir, get_type, get_type_hint, 
    wrap_value, get_pc_type
)


class TestValueRef(unittest.TestCase):
    """Test ValueRef wrapper functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.module = ir.Module(name="test_module")
        self.func_type = ir.FunctionType(ir.VoidType(), [])
        self.func = ir.Function(self.module, self.func_type, name="test_func")
        self.block = self.func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(self.block)
    
    def test_value_ref_creation(self):
        """Test basic ValueRef creation"""
        from pythoc.builtin_entities import i32
        i32_val = ir.Constant(ir.IntType(32), 42)
        ref = ValueRef(kind="value", value=i32_val, type_hint=i32)
        
        self.assertEqual(ref.kind, "value")
        self.assertEqual(ref.ir_value, i32_val)
        self.assertEqual(ref.type, ir.IntType(32))
    
    def test_value_ref_with_type_hint(self):
        """Test ValueRef with type hint"""
        from pythoc.builtin_entities import i32
        
        i32_val = ir.Constant(ir.IntType(32), 42)
        ref = ValueRef(kind="value", value=i32_val, type_hint=i32)
        
        self.assertEqual(ref.type_hint, i32)
        self.assertEqual(get_type_hint(ref), i32)
    
    def test_value_ref_with_source_node(self):
        """Test ValueRef with source AST node"""
        from pythoc.builtin_entities import i32
        node = ast.Constant(value=42)
        i32_val = ir.Constant(ir.IntType(32), 42)
        ref = ValueRef(kind="value", value=i32_val, type_hint=i32, source_node=node)
        
        self.assertEqual(ref.source_node, node)
    
    def test_is_pointer(self):
        """Test pointer type detection"""
        # Non-pointer value
        from pythoc.builtin_entities import i32
        i32_val = ir.Constant(ir.IntType(32), 42)
        ref = ValueRef(kind="value", value=i32_val, type_hint=i32)
        self.assertFalse(ref.is_pointer())
        
        # Pointer value
        ptr_type = ir.PointerType(ir.IntType(32))
        from pythoc.builtin_entities import i32, ptr
        alloca = self.builder.alloca(ir.IntType(32))
        ptr_ref = ValueRef(kind="pointer", value=alloca, type_hint=ptr[i32])
        self.assertTrue(ptr_ref.is_pointer())
    
    def test_pointee(self):
        """Test getting pointee type"""
        from pythoc.builtin_entities import i32, ptr
        alloca = self.builder.alloca(ir.IntType(32))
        ptr_ref = ValueRef(kind="pointer", value=alloca, type_hint=ptr[i32])
        
        self.assertEqual(ptr_ref.pointee(), ir.IntType(32))
    
    def test_load(self):
        """Test load operation"""
        from pythoc.builtin_entities import i32
        
        from pythoc.builtin_entities import i32, ptr
        alloca = self.builder.alloca(ir.IntType(32))
        ptr_ref = ValueRef(kind="pointer", value=alloca, type_hint=ptr[i32])
        
        loaded = ptr_ref.load(self.builder)
        self.assertEqual(loaded.kind, "value")
        self.assertEqual(loaded.type_hint, i32)
    
    def test_ensure_ir(self):
        """Test ensure_ir helper function"""
        i32_val = ir.Constant(ir.IntType(32), 42)
        
        # Test with raw ir.Value
        self.assertEqual(ensure_ir(i32_val), i32_val)
        
        # Test with ValueRef
        from pythoc.builtin_entities import i32
        ref = ValueRef(kind="value", value=i32_val, type_hint=i32)
        self.assertEqual(ensure_ir(ref), i32_val)
    
    def test_get_type(self):
        """Test get_type helper function"""
        i32_val = ir.Constant(ir.IntType(32), 42)
        
        # Test with raw ir.Value
        self.assertEqual(get_type(i32_val), ir.IntType(32))
        
        # Test with ValueRef
        from pythoc.builtin_entities import i32
        ref = ValueRef(kind="value", value=i32_val, type_hint=i32)
        self.assertEqual(get_type(ref), ir.IntType(32))
    
    def test_wrap_value(self):
        """Test wrap_value helper function"""
        from pythoc.builtin_entities import i32
        
        i32_val = ir.Constant(ir.IntType(32), 42)
        ref = wrap_value(i32_val, kind="value", type_hint=i32)
        
        self.assertIsInstance(ref, ValueRef)
        self.assertEqual(ref.kind, "value")
        self.assertEqual(ref.ir_value, i32_val)
        self.assertEqual(ref.type_hint, i32)
    
    def test_repr(self):
        """Test string representation"""
        from pythoc.builtin_entities import i32
        
        i32_val = ir.Constant(ir.IntType(32), 42)
        ref = ValueRef(kind="value", value=i32_val, type_hint=i32)
        
        repr_str = repr(ref)
        self.assertIn("kind=value", repr_str)
        self.assertIn("type=i32", repr_str)


if __name__ == '__main__':
    unittest.main()
