"""
Unit tests for TypeConverter

Tests all type conversion scenarios to ensure correctness.
"""

import unittest
from llvmlite import ir
from pythoc.type_converter import TypeConverter
from pythoc.valueref import wrap_value, ensure_ir, get_type


class TestTypeConverter(unittest.TestCase):
    """Test TypeConverter functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.module = ir.Module(name="test_module")
        self.func_type = ir.FunctionType(ir.VoidType(), [])
        self.func = ir.Function(self.module, self.func_type, name="test_func")
        self.block = self.func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(self.block)
        
        # Create a mock visitor with builder and module attributes
        class MockVisitor:
            pass
        self.visitor = MockVisitor()
        self.visitor.builder = self.builder
        self.visitor.module = self.module
        self.converter = TypeConverter(self.visitor)
    
    def test_int_to_int_sext(self):
        """Test integer sign extension (i8 -> i32)"""
        from pythoc import i8, i32
        i8_val = wrap_value(ir.Constant(ir.IntType(8), 127), kind="value", type_hint=i8)
        result = self.converter.convert(i8_val, i32)
        
        self.assertEqual(get_type(result), ir.IntType(32))
        result_ir = ensure_ir(result)
        self.assertIn('sext', str(result_ir))
    
    def test_int_to_int_zext(self):
        """Test integer zero extension (u8 -> u32)"""
        from pythoc import u8, u32
        u8_val = wrap_value(ir.Constant(ir.IntType(8), 255), kind="value", type_hint=u8)
        result = self.converter.convert(u8_val, u32)
        
        self.assertEqual(get_type(result), ir.IntType(32))
        result_ir = ensure_ir(result)
        self.assertIn('zext', str(result_ir))
    
    def test_int_to_int_trunc(self):
        """Test integer truncation (i32 -> i8)"""
        from pythoc import i8, i32
        i32_val = wrap_value(ir.Constant(ir.IntType(32), 1000), kind="value", type_hint=i32)
        result = self.converter.convert(i32_val, i8)
        
        self.assertEqual(get_type(result), ir.IntType(8))
        result_ir = ensure_ir(result)
        self.assertIn('trunc', str(result_ir))
    
    def test_int_to_int_same_width(self):
        """Test integer conversion with same width (no-op)"""
        from pythoc import i32
        i32_val = wrap_value(ir.Constant(ir.IntType(32), 42), kind="value", type_hint=i32)
        result = self.converter.convert(i32_val, i32)
        
        self.assertEqual(get_type(result), ir.IntType(32))
    
    def test_int_to_float(self):
        """Test integer to float conversion"""
        from pythoc import i32, f32
        i32_val = wrap_value(ir.Constant(ir.IntType(32), 42), kind="value", type_hint=i32)
        result = self.converter.convert(i32_val, f32)
        
        self.assertEqual(get_type(result), ir.FloatType())
        result_ir = ensure_ir(result)
        self.assertIn('sitofp', str(result_ir))
    
    def test_int_to_double(self):
        """Test integer to double conversion"""
        from pythoc import i32, f64
        i32_val = wrap_value(ir.Constant(ir.IntType(32), 42), kind="value", type_hint=i32)
        result = self.converter.convert(i32_val, f64)
        
        self.assertEqual(get_type(result), ir.DoubleType())
        result_ir = ensure_ir(result)
        self.assertIn('sitofp', str(result_ir))
    
    def test_unsigned_int_to_float(self):
        """Test unsigned integer to float conversion"""
        from pythoc import u32, f32
        u32_val = wrap_value(ir.Constant(ir.IntType(32), 42), kind="value", type_hint=u32)
        result = self.converter.convert(u32_val, f32)
        
        self.assertEqual(get_type(result), ir.FloatType())
        result_ir = ensure_ir(result)
        self.assertIn('uitofp', str(result_ir))
    
    def test_float_to_int(self):
        """Test float to integer conversion"""
        from pythoc import f32, i32
        f32_val = wrap_value(ir.Constant(ir.FloatType(), 42.5), kind="value", type_hint=f32)
        result = self.converter.convert(f32_val, i32)
        
        self.assertEqual(get_type(result), ir.IntType(32))
        result_ir = ensure_ir(result)
        self.assertIn('fptosi', str(result_ir))
    
    def test_float_to_unsigned_int(self):
        """Test float to unsigned integer conversion"""
        from pythoc import f32, u32
        f32_val = wrap_value(ir.Constant(ir.FloatType(), 42.5), kind="value", type_hint=f32)
        result = self.converter.convert(f32_val, u32)
        
        self.assertEqual(get_type(result), ir.IntType(32))
        result_ir = ensure_ir(result)
        self.assertIn('fptoui', str(result_ir))
    
    def test_float_to_double(self):
        """Test float to double conversion"""
        from pythoc import f32, f64
        f32_val = wrap_value(ir.Constant(ir.FloatType(), 3.14), kind="value", type_hint=f32)
        result = self.converter.convert(f32_val, f64)
        
        self.assertEqual(get_type(result), ir.DoubleType())
        result_ir = ensure_ir(result)
        self.assertIn('fpext', str(result_ir))
    
    def test_double_to_float(self):
        """Test double to float conversion"""
        from pythoc import f32, f64
        f64_val = wrap_value(ir.Constant(ir.DoubleType(), 3.14159265359), kind="value", type_hint=f64)
        result = self.converter.convert(f64_val, f32)
        
        self.assertEqual(get_type(result), ir.FloatType())
        result_ir = ensure_ir(result)
        self.assertIn('fptrunc', str(result_ir))
    
    def test_ptr_to_ptr(self):
        """Test pointer to pointer conversion (bitcast)"""
        from pythoc import ptr, i8, i32
        i32_ptr = wrap_value(self.builder.alloca(ir.IntType(32)), kind="value", type_hint=ptr[i32])
        result = self.converter.convert(i32_ptr, ptr[i8])
        
        self.assertEqual(get_type(result), ir.PointerType(ir.IntType(8)))
        result_ir = ensure_ir(result)
        self.assertIn('bitcast', str(result_ir))
    
    def test_promote_to_float_int_to_float(self):
        """Test promote_to_float with integer to float"""
        from pythoc import i32, f32
        i32_val = wrap_value(ir.Constant(ir.IntType(32), 10), kind="value", type_hint=i32)
        
        result = self.converter.promote_to_float(i32_val, f32)
        
        # Integer should be promoted to float
        self.assertEqual(get_type(result), ir.FloatType())
        self.assertEqual(result.type_hint, f32)
    
    def test_promote_to_float_int_to_double(self):
        """Test promote_to_float with integer to double"""
        from pythoc import i32, f64
        i32_val = wrap_value(ir.Constant(ir.IntType(32), 10), kind="value", type_hint=i32)
        
        result = self.converter.promote_to_float(i32_val, f64)
        
        # Integer should be promoted to double
        self.assertEqual(get_type(result), ir.DoubleType())
        self.assertEqual(result.type_hint, f64)
    
    def test_promote_to_float_float_to_double(self):
        """Test promote_to_float with float to double"""
        from pythoc import f32, f64
        f32_val = wrap_value(ir.Constant(ir.FloatType(), 10.0), kind="value", type_hint=f32)
        
        result = self.converter.promote_to_float(f32_val, f64)
        
        # Float should be promoted to double
        self.assertEqual(get_type(result), ir.DoubleType())
        self.assertEqual(result.type_hint, f64)
    
    def test_unify_integer_types_same_width(self):
        """Test unify_integer_types with same width"""
        from pythoc import i32
        i32_val1 = wrap_value(ir.Constant(ir.IntType(32), 10), kind="value", type_hint=i32)
        i32_val2 = wrap_value(ir.Constant(ir.IntType(32), 20), kind="value", type_hint=i32)
        
        result1, result2 = self.converter.unify_integer_types(i32_val1, i32_val2)
        
        self.assertEqual(get_type(result1), ir.IntType(32))
        self.assertEqual(get_type(result2), ir.IntType(32))
    
    def test_unify_integer_types_different_width(self):
        """Test unify_integer_types with different widths"""
        from pythoc import i8, i32
        i8_val = wrap_value(ir.Constant(ir.IntType(8), 10), kind="value", type_hint=i8)
        i32_val = wrap_value(ir.Constant(ir.IntType(32), 20), kind="value", type_hint=i32)
        
        result1, result2 = self.converter.unify_integer_types(i8_val, i32_val)
        
        # Both should be promoted to i32
        self.assertEqual(get_type(result1), ir.IntType(32))
        self.assertEqual(get_type(result2), ir.IntType(32))
    
    def test_convert_with_value_ref(self):
        """Test conversion with ValueRef input"""
        from pythoc import i8, i32
        i8_val = ir.Constant(ir.IntType(8), 42)
        wrapped = wrap_value(i8_val, kind="value", type_hint=i8)
        
        result = self.converter.convert(wrapped, i32)
        
        self.assertEqual(get_type(result), ir.IntType(32))
    
    def test_convert_null_pointer(self):
        """Test conversion of null pointer"""
        from pythoc import ptr, i8, i32
        null_ptr = wrap_value(ir.Constant(ir.PointerType(ir.IntType(32)), None), kind="value", type_hint=ptr[i32])
        
        result = self.converter.convert(null_ptr, ptr[i8])
        
        self.assertEqual(get_type(result), ir.PointerType(ir.IntType(8)))
        # Should be a constant null pointer
        self.assertIsInstance(ensure_ir(result), ir.Constant)
    
    def test_convert_constant_int(self):
        """Test conversion of constant integer"""
        from pythoc import i8, i32
        i8_const = wrap_value(ir.Constant(ir.IntType(8), 42), kind="value", type_hint=i8)
        result = self.converter.convert(i8_const, i32)
        
        self.assertEqual(get_type(result), ir.IntType(32))
        # Converter uses builder instructions even for constants
        result_ir = ensure_ir(result)
        self.assertIn('sext', str(result_ir))
    
    def test_unify_binop_types_int_int(self):
        """Test unify_binop_types with two integers"""
        from pythoc.builtin_entities import i8, i32
        i8_val = wrap_value(ir.Constant(ir.IntType(8), 10), kind="value", type_hint=i8)
        i32_val = wrap_value(ir.Constant(ir.IntType(32), 20), kind="value", type_hint=i32)
        
        left, right, is_float = self.converter.unify_binop_types(i8_val, i32_val)
        
        # Both should be promoted to i32
        self.assertEqual(get_type(left), ir.IntType(32))
        self.assertEqual(get_type(right), ir.IntType(32))
        self.assertFalse(is_float)
    
    def test_unify_binop_types_int_float(self):
        """Test unify_binop_types with integer and float"""
        from pythoc.builtin_entities import i32, f64
        i32_val = wrap_value(ir.Constant(ir.IntType(32), 10), kind="value", type_hint=i32)
        f64_val = wrap_value(ir.Constant(ir.DoubleType(), 20.0), kind="value", type_hint=f64)
        
        left, right, is_float = self.converter.unify_binop_types(i32_val, f64_val)
        
        # Both should be promoted to f64
        self.assertEqual(get_type(left), ir.DoubleType())
        self.assertEqual(get_type(right), ir.DoubleType())
        self.assertTrue(is_float)
    
    def test_unify_binop_types_float_int(self):
        """Test unify_binop_types with float and integer"""
        from pythoc.builtin_entities import f32, i32
        f32_val = wrap_value(ir.Constant(ir.FloatType(), 10.0), kind="value", type_hint=f32)
        i32_val = wrap_value(ir.Constant(ir.IntType(32), 20), kind="value", type_hint=i32)
        
        left, right, is_float = self.converter.unify_binop_types(f32_val, i32_val)
        
        # Both should be promoted to f32
        self.assertEqual(get_type(left), ir.FloatType())
        self.assertEqual(get_type(right), ir.FloatType())
        self.assertTrue(is_float)
    
    def test_unify_binop_types_float_float(self):
        """Test unify_binop_types with two floats"""
        from pythoc.builtin_entities import f32, f64
        f32_val = wrap_value(ir.Constant(ir.FloatType(), 10.0), kind="value", type_hint=f32)
        f64_val = wrap_value(ir.Constant(ir.DoubleType(), 20.0), kind="value", type_hint=f64)
        
        left, right, is_float = self.converter.unify_binop_types(f32_val, f64_val)
        
        # Both should be promoted to f64 (wider type)
        self.assertEqual(get_type(left), ir.DoubleType())
        self.assertEqual(get_type(right), ir.DoubleType())
        self.assertTrue(is_float)
    
    def test_unify_binop_types_same_type(self):
        """Test unify_binop_types with same types"""
        from pythoc.builtin_entities import i32
        i32_val1 = wrap_value(ir.Constant(ir.IntType(32), 10), kind="value", type_hint=i32)
        i32_val2 = wrap_value(ir.Constant(ir.IntType(32), 20), kind="value", type_hint=i32)
        
        left, right, is_float = self.converter.unify_binop_types(i32_val1, i32_val2)
        
        # Both should remain i32
        self.assertEqual(get_type(left), ir.IntType(32))
        self.assertEqual(get_type(right), ir.IntType(32))
        self.assertFalse(is_float)


if __name__ == '__main__':
    unittest.main()
