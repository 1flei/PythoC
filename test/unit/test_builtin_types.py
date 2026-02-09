"""
Unit tests for builtin type entities
"""

import unittest
from llvmlite import ir

from pythoc.builtin_entities import (
    i8, i16, i32, i64,
    u8, u16, u32, u64,
    f32, f64,
    ptr, array,
    get_builtin_entity
)


class TestIntegerTypes(unittest.TestCase):
    """Test integer type entities"""
    
    def test_i8_properties(self):
        """Test i8 type properties"""
        self.assertEqual(i8.get_name(), "i8")
        self.assertEqual(i8.get_llvm_type(), ir.IntType(8))
        self.assertEqual(i8.get_size_bytes(), 1)
        self.assertTrue(i8.is_signed())
        self.assertTrue(i8.is_integer())
        self.assertTrue(i8.can_be_type())
    
    def test_i16_properties(self):
        """Test i16 type properties"""
        self.assertEqual(i16.get_name(), "i16")
        self.assertEqual(i16.get_llvm_type(), ir.IntType(16))
        self.assertEqual(i16.get_size_bytes(), 2)
        self.assertTrue(i16.is_signed())
    
    def test_i32_properties(self):
        """Test i32 type properties"""
        self.assertEqual(i32.get_name(), "i32")
        self.assertEqual(i32.get_llvm_type(), ir.IntType(32))
        self.assertEqual(i32.get_size_bytes(), 4)
        self.assertTrue(i32.is_signed())
    
    def test_i64_properties(self):
        """Test i64 type properties"""
        self.assertEqual(i64.get_name(), "i64")
        self.assertEqual(i64.get_llvm_type(), ir.IntType(64))
        self.assertEqual(i64.get_size_bytes(), 8)
        self.assertTrue(i64.is_signed())
    
    def test_u8_properties(self):
        """Test u8 type properties"""
        self.assertEqual(u8.get_name(), "u8")
        self.assertEqual(u8.get_llvm_type(), ir.IntType(8))
        self.assertEqual(u8.get_size_bytes(), 1)
        self.assertFalse(u8.is_signed())
        self.assertTrue(u8.is_integer())
    
    def test_u16_properties(self):
        """Test u16 type properties"""
        self.assertEqual(u16.get_name(), "u16")
        self.assertEqual(u16.get_llvm_type(), ir.IntType(16))
        self.assertEqual(u16.get_size_bytes(), 2)
        self.assertFalse(u16.is_signed())
    
    def test_u32_properties(self):
        """Test u32 type properties"""
        self.assertEqual(u32.get_name(), "u32")
        self.assertEqual(u32.get_llvm_type(), ir.IntType(32))
        self.assertEqual(u32.get_size_bytes(), 4)
        self.assertFalse(u32.is_signed())
    
    def test_u64_properties(self):
        """Test u64 type properties"""
        self.assertEqual(u64.get_name(), "u64")
        self.assertEqual(u64.get_llvm_type(), ir.IntType(64))
        self.assertEqual(u64.get_size_bytes(), 8)
        self.assertFalse(u64.is_signed())


class TestFloatTypes(unittest.TestCase):
    """Test floating point type entities"""
    
    def test_f32_properties(self):
        """Test f32 type properties"""
        self.assertEqual(f32.get_name(), "f32")
        self.assertEqual(f32.get_llvm_type(), ir.FloatType())
        self.assertEqual(f32.get_size_bytes(), 4)
        self.assertTrue(f32.is_float())
        self.assertTrue(f32.can_be_type())
    
    def test_f64_properties(self):
        """Test f64 type properties"""
        self.assertEqual(f64.get_name(), "f64")
        self.assertEqual(f64.get_llvm_type(), ir.DoubleType())
        self.assertEqual(f64.get_size_bytes(), 8)
        self.assertTrue(f64.is_float())


class TestPointerType(unittest.TestCase):
    """Test pointer type entity"""
    
    def test_ptr_basic_properties(self):
        """Test basic ptr properties"""
        # Void pointer canonical spelling.
        self.assertEqual(ptr.get_name(), "ptr[void]")
        self.assertTrue(ptr.can_be_type())
    
    def test_ptr_specialization(self):
        """Test ptr type specialization"""
        ptr_i32 = ptr[i32]
        
        self.assertTrue(hasattr(ptr_i32, 'get_name'))
        self.assertTrue(ptr_i32.can_be_type())
        
        llvm_type = ptr_i32.get_llvm_type()
        self.assertIsInstance(llvm_type, ir.PointerType)
        self.assertEqual(llvm_type.pointee, ir.IntType(32))
    
    def test_ptr_nested_specialization(self):
        """Test nested ptr specialization ptr[ptr[i32]]"""
        ptr_ptr_i32 = ptr[ptr[i32]]
        
        llvm_type = ptr_ptr_i32.get_llvm_type()
        self.assertIsInstance(llvm_type, ir.PointerType)
        self.assertIsInstance(llvm_type.pointee, ir.PointerType)
        self.assertEqual(llvm_type.pointee.pointee, ir.IntType(32))


class TestArrayType(unittest.TestCase):
    """Test array type entity"""
    
    def test_array_basic_properties(self):
        """Test basic array properties"""
        self.assertEqual(array.get_name(), "array")
        self.assertTrue(array.can_be_type())
    
    def test_array_1d_specialization(self):
        """Test 1D array specialization"""
        arr_i32_10 = array[i32, 10]
        
        self.assertTrue(hasattr(arr_i32_10, 'get_name'))
        self.assertTrue(arr_i32_10.can_be_type())
        
        llvm_type = arr_i32_10.get_llvm_type()
        self.assertIsInstance(llvm_type, ir.ArrayType)
        self.assertEqual(llvm_type.element, ir.IntType(32))
        self.assertEqual(llvm_type.count, 10)
    
    def test_array_2d_specialization(self):
        """Test 2D array specialization"""
        arr_i32_10_20 = array[i32, 10, 20]
        
        llvm_type = arr_i32_10_20.get_llvm_type()
        self.assertIsInstance(llvm_type, ir.ArrayType)
        # Outer array has 10 elements
        self.assertEqual(llvm_type.count, 10)
        # Each element is an array of 20 i32s
        self.assertIsInstance(llvm_type.element, ir.ArrayType)
        self.assertEqual(llvm_type.element.count, 20)
        self.assertEqual(llvm_type.element.element, ir.IntType(32))


class TestBuiltinEntityRegistry(unittest.TestCase):
    """Test builtin entity registry"""
    
    def test_get_builtin_entity_integer(self):
        """Test getting integer types from registry"""
        entity = get_builtin_entity("i32")
        self.assertEqual(entity, i32)
        
        entity = get_builtin_entity("u64")
        self.assertEqual(entity, u64)
    
    def test_get_builtin_entity_float(self):
        """Test getting float types from registry"""
        entity = get_builtin_entity("f32")
        self.assertEqual(entity, f32)
        
        entity = get_builtin_entity("f64")
        self.assertEqual(entity, f64)
    
    def test_get_builtin_entity_ptr(self):
        """Test getting ptr from registry"""
        entity = get_builtin_entity("ptr")
        self.assertEqual(entity, ptr)
    
    def test_get_builtin_entity_array(self):
        """Test getting array from registry"""
        entity = get_builtin_entity("array")
        self.assertEqual(entity, array)
    
    def test_get_builtin_entity_invalid(self):
        """Test getting invalid entity returns None"""
        entity = get_builtin_entity("invalid_type")
        self.assertIsNone(entity)


if __name__ == '__main__':
    unittest.main()
