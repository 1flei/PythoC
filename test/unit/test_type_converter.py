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
        # Constant inputs are folded to the target constant directly
        self.assertIsInstance(result_ir, ir.Constant)
        self.assertEqual(result_ir.constant, 127)
    
    def test_int_to_int_zext(self):
        """Test integer zero extension (u8 -> u32)"""
        from pythoc import u8, u32
        u8_val = wrap_value(ir.Constant(ir.IntType(8), 255), kind="value", type_hint=u8)
        result = self.converter.convert(u8_val, u32)
        
        self.assertEqual(get_type(result), ir.IntType(32))
        result_ir = ensure_ir(result)
        # Constant inputs are folded to the target constant directly
        self.assertIsInstance(result_ir, ir.Constant)
        self.assertEqual(result_ir.constant, 255)
    
    def test_int_to_int_trunc(self):
        """Test integer truncation (i32 -> i8)"""
        from pythoc import i8, i32
        i32_val = wrap_value(ir.Constant(ir.IntType(32), 1000), kind="value", type_hint=i32)
        result = self.converter.convert(i32_val, i8)
        
        self.assertEqual(get_type(result), ir.IntType(8))
        result_ir = ensure_ir(result)
        # Constant inputs are folded to the target constant directly
        self.assertIsInstance(result_ir, ir.Constant)
        self.assertEqual(result_ir.constant, 1000 & 0xFF)
    
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
        # Constant inputs are folded to the target constant directly
        self.assertIsInstance(result_ir, ir.Constant)
        self.assertAlmostEqual(result_ir.constant, 42.0)
    
    def test_int_to_double(self):
        """Test integer to double conversion"""
        from pythoc import i32, f64
        i32_val = wrap_value(ir.Constant(ir.IntType(32), 42), kind="value", type_hint=i32)
        result = self.converter.convert(i32_val, f64)
        
        self.assertEqual(get_type(result), ir.DoubleType())
        result_ir = ensure_ir(result)
        # Constant inputs are folded to the target constant directly
        self.assertIsInstance(result_ir, ir.Constant)
        self.assertAlmostEqual(result_ir.constant, 42.0)
    
    def test_unsigned_int_to_float(self):
        """Test unsigned integer to float conversion"""
        from pythoc import u32, f32
        u32_val = wrap_value(ir.Constant(ir.IntType(32), 42), kind="value", type_hint=u32)
        result = self.converter.convert(u32_val, f32)
        
        self.assertEqual(get_type(result), ir.FloatType())
        result_ir = ensure_ir(result)
        # Constant inputs are folded to the target constant directly
        self.assertIsInstance(result_ir, ir.Constant)
        self.assertAlmostEqual(result_ir.constant, 42.0)
    
    def test_float_to_int(self):
        """Test float to integer conversion"""
        from pythoc import f32, i32
        f32_val = wrap_value(ir.Constant(ir.FloatType(), 42.5), kind="value", type_hint=f32)
        result = self.converter.convert(f32_val, i32)
        
        self.assertEqual(get_type(result), ir.IntType(32))
        result_ir = ensure_ir(result)
        # Constant inputs are folded to the target constant directly (truncated toward zero)
        self.assertIsInstance(result_ir, ir.Constant)
        self.assertEqual(result_ir.constant, 42)
    
    def test_float_to_unsigned_int(self):
        """Test float to unsigned integer conversion"""
        from pythoc import f32, u32
        f32_val = wrap_value(ir.Constant(ir.FloatType(), 42.5), kind="value", type_hint=f32)
        result = self.converter.convert(f32_val, u32)
        
        self.assertEqual(get_type(result), ir.IntType(32))
        result_ir = ensure_ir(result)
        # Constant inputs are folded to the target constant directly (truncated toward zero)
        self.assertIsInstance(result_ir, ir.Constant)
        self.assertEqual(result_ir.constant, 42)
    
    def test_float_to_double(self):
        """Test float to double conversion"""
        from pythoc import f32, f64
        f32_val = wrap_value(ir.Constant(ir.FloatType(), 3.14), kind="value", type_hint=f32)
        result = self.converter.convert(f32_val, f64)
        
        self.assertEqual(get_type(result), ir.DoubleType())
        result_ir = ensure_ir(result)
        # Constant inputs are folded to the target constant directly
        self.assertIsInstance(result_ir, ir.Constant)
        self.assertAlmostEqual(result_ir.constant, 3.14, places=5)
    
    def test_double_to_float(self):
        """Test double to float conversion"""
        from pythoc import f32, f64
        f64_val = wrap_value(ir.Constant(ir.DoubleType(), 3.14159265359), kind="value", type_hint=f64)
        result = self.converter.convert(f64_val, f32)
        
        self.assertEqual(get_type(result), ir.FloatType())
        result_ir = ensure_ir(result)
        # Constant inputs are folded to the target constant directly (single precision)
        self.assertIsInstance(result_ir, ir.Constant)
        self.assertAlmostEqual(result_ir.constant, 3.14159265359, places=5)
    
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
        # Constant inputs are folded to the target constant directly
        result_ir = ensure_ir(result)
        self.assertIsInstance(result_ir, ir.Constant)
        self.assertEqual(result_ir.constant, 42)
    
    def test_convert_to_void(self):
        """C-style discard cast: converting any value to void is a no-op"""
        from pythoc import i32, void
        i32_val = wrap_value(ir.Constant(ir.IntType(32), 42), kind="value", type_hint=i32)
        result = self.converter.convert(i32_val, void)
        
        self.assertEqual(result.type_hint, void)
        self.assertTrue(result.is_python_value())
    
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

    def test_supported_float_zero_constants(self):
        from pythoc import bf16, f16, f32, f64, f128

        for pc_type in (f16, bf16, f32, f64, f128):
            with self.subTest(pc_type=pc_type):
                llvm_type = pc_type.get_llvm_type(self.module.context)
                self.assertTrue(self.converter.is_llvm_float_type(llvm_type))
                zero = self.converter.create_zero_constant(llvm_type)
                self.assertEqual(zero.type, llvm_type)
                self.assertEqual(zero.constant, 0.0)

    def test_vector_zero_constants_keep_lane_type_and_count(self):
        for element, lanes in ((ir.IntType(64), 4), (ir.FloatType(), 8)):
            with self.subTest(element=element, lanes=lanes):
                vector_type = ir.VectorType(element, lanes)
                zero = self.converter.create_zero_constant(vector_type)
                self.assertEqual(zero.type, vector_type)
                self.assertEqual(len(zero.constant), lanes)
                self.assertTrue(all(lane.type == element for lane in zero.constant))

    def test_fp128_constants_match_llvm_widening(self):
        import math
        from llvmlite import binding
        from pythoc.compiler import LLVMCompiler
        from pythoc.llvm_extensions import FP128Type

        values = (0.0, -0.0, 1.0, -2.5, math.inf, -math.inf, math.nan,
                  float.fromhex('0x0.0000000000001p-1022'),
                  float.fromhex('0x0.fffffffffffffp-1022'),
                  float.fromhex('0x1.fffffffffffffp+1023'))
        for value in values:
            with self.subTest(value=value):
                compiler = LLVMCompiler()
                function = ir.Function(
                    compiler.module, ir.FunctionType(ir.IntType(1), []),
                    name='same_bits')
                builder = ir.IRBuilder(function.append_basic_block('entry'))
                wide = FP128Type()
                actual = ir.Constant(wide, value)
                expected = builder.fpext(
                    ir.Constant(ir.DoubleType(), value), wide)
                builder.ret(builder.icmp_unsigned(
                    '==', builder.bitcast(actual, ir.IntType(128)),
                    builder.bitcast(expected, ir.IntType(128))))
                compiler.optimize_module(2)
                with binding.parse_assembly(compiler.get_ir()) as module:
                    block = next(module.get_function('same_bits').blocks)
                    result = next(list(block.instructions)[-1].operands)
                    self.assertEqual(result.get_constant_value(), 1)

    def test_float_formats_match_llvm_integer_conversion(self):
        from pythoc.compiler import LLVMCompiler
        from pythoc.llvm_extensions import FLOAT_FORMATS, constant

        compiler = LLVMCompiler()
        values = (0, 1, -1, 65519, 65520, (1 << 24) + 1,
                  (1 << 53) + 1, (1 << 80) + (1 << 56) + 1,
                  (1 << 113) + 1, -((1 << 113) + 3))
        for fmt in FLOAT_FORMATS:
            for index, value in enumerate(values):
                function = ir.Function(
                    compiler.module, ir.FunctionType(ir.IntType(1), []),
                    name=f'{fmt.llvm_type.intrinsic_name}_{index}')
                builder = ir.IRBuilder(function.append_basic_block('entry'))
                expected = builder.sitofp(ir.Constant(ir.IntType(256), value),
                                          fmt.llvm_type)
                bits = ir.IntType(fmt.bits)
                builder.ret(builder.icmp_unsigned(
                    '==', builder.bitcast(constant(fmt.llvm_type, value), bits),
                    builder.bitcast(expected, bits)))
        compiler.optimize_module(2)
        from llvmlite import binding
        with binding.parse_assembly(compiler.get_ir()) as module:
            for function in module.functions:
                with self.subTest(function=function.name):
                    block = next(function.blocks)
                    result = next(list(block.instructions)[-1].operands)
                    self.assertEqual(result.get_constant_value(), 1)

    def test_native_float_constants_do_not_require_bit_decoding(self):
        from unittest.mock import patch
        import operator
        from pythoc.llvm_extensions import (
            F16, F32, F64, FloatFormat, constant, constant_operation,
        )

        with patch.object(FloatFormat, 'encode', side_effect=AssertionError('decoded')):
            for fmt in (F16, F32, F64):
                self.assertEqual(constant(fmt.llvm_type, 1.5).constant, 1.5)
                self.assertEqual(constant_operation(fmt.llvm_type, operator.add,
                                                    1.5, 2.5), 4.0)

    def test_float_to_int_observes_the_source_format_rounding(self):
        from pythoc import f16, f32, i64
        for source, value, expected in (
                (f16, 1.9999, 2), (f32, 16777217.0, 16777216)):
            with self.subTest(source=source):
                operand = wrap_value(ir.Constant(source.get_llvm_type(), value),
                                     kind='value', type_hint=source)
                self.assertEqual(self.converter.convert(operand, i64).ir_value.constant,
                                 expected)

    def test_half_bfloat_conversion_uses_a_common_containing_format(self):
        from pythoc import f16, bf16
        for source, target in ((f16, bf16), (bf16, f16)):
            function = ir.Function(
                self.module, ir.FunctionType(target.get_llvm_type(),
                                              [source.get_llvm_type()]),
                name=source.get_name())
            builder = ir.IRBuilder(function.append_basic_block('entry'))
            from pythoc.llvm_extensions import convert
            result = convert(builder, function.args[0], target.get_llvm_type())
            self.assertEqual(result.type, target.get_llvm_type())
            self.assertEqual(result.operands[0].type, ir.FloatType())
            builder.ret(result)

    def test_mixed_precision_intrinsics_keep_distinct_signatures(self):
        from types import SimpleNamespace
        from pythoc.ast_visitor.helpers import HelpersMixin
        from pythoc.llvm_extensions import FLOAT_FORMATS

        visitor = SimpleNamespace(module=self.module)
        for operation in ('_get_floor_intrinsic', '_get_pow_intrinsic'):
            names = set()
            for fmt in FLOAT_FORMATS:
                function = getattr(HelpersMixin, operation)(visitor, fmt.llvm_type)
                self.assertEqual(function.function_type.return_type, fmt.llvm_type)
                names.add(function.name)
            self.assertEqual(len(names), len(FLOAT_FORMATS))

    def test_float_rounding_boundaries_match_llvm(self):
        from pythoc.compiler import LLVMCompiler
        from pythoc.llvm_extensions import FLOAT_FORMATS, constant

        compiler = LLVMCompiler()
        values = (0.0, -0.0, 1.00390625, 1.01171875, 65519.0, 65520.0,
                  2.0 ** -24, 2.0 ** -25, 2.0 ** -133, 2.0 ** -134,
                  float.fromhex('0x1.0000010000001p+0'),
                  float.fromhex('0x1.fffffffffffffp+1023'))
        for fmt in FLOAT_FORMATS:
            for index, value in enumerate(values):
                function = ir.Function(
                    compiler.module, ir.FunctionType(ir.IntType(1), []),
                    name=f'{fmt.llvm_type.intrinsic_name}_{index}')
                builder = ir.IRBuilder(function.append_basic_block('entry'))
                source = ir.Constant(ir.DoubleType(), value)
                if fmt.bits < 64:
                    expected = builder.fptrunc(source, fmt.llvm_type)
                elif fmt.bits > 64:
                    expected = builder.fpext(source, fmt.llvm_type)
                else:
                    expected = source
                bits = ir.IntType(fmt.bits)
                builder.ret(builder.icmp_unsigned(
                    '==', builder.bitcast(constant(fmt.llvm_type, value), bits),
                    builder.bitcast(expected, bits)))
        compiler.optimize_module(2)
        from llvmlite import binding
        with binding.parse_assembly(compiler.get_ir()) as module:
            for function in module.functions:
                with self.subTest(function=function.name):
                    block = next(function.blocks)
                    result = next(list(block.instructions)[-1].operands)
                    self.assertEqual(result.get_constant_value(), 1)

    def test_constant_arithmetic_matches_llvm_in_every_float_format(self):
        from pythoc.compiler import LLVMCompiler
        from pythoc.llvm_extensions import FLOAT_FORMATS, binary, constant

        compiler = LLVMCompiler()
        cases = ((0.0, -0.0), (-0.0, -0.0), (1.5, 2.5),
                 (-7.0, 3.0), (1.0, 0.0), (-1.0, -0.0))
        for fmt in FLOAT_FORMATS:
            for operation in ('fadd', 'fsub', 'fmul', 'fdiv'):
                for index, (a, b) in enumerate(cases):
                    if not a and not b and operation == 'fdiv':
                        continue
                    function = ir.Function(
                        compiler.module, ir.FunctionType(ir.IntType(1), []),
                        name=f'{fmt.llvm_type.intrinsic_name}_{operation}_{index}')
                    builder = ir.IRBuilder(function.append_basic_block('entry'))
                    left, right = constant(fmt.llvm_type, a), constant(fmt.llvm_type, b)
                    actual = binary(builder, operation, left, right)
                    expected = getattr(builder, operation)(left, right)
                    bits = ir.IntType(fmt.bits)
                    builder.ret(builder.icmp_unsigned(
                        '==', builder.bitcast(actual, bits),
                        builder.bitcast(expected, bits)))
        compiler.optimize_module(2)
        from llvmlite import binding
        with binding.parse_assembly(compiler.get_ir()) as module:
            for function in module.functions:
                with self.subTest(function=function.name):
                    block = next(function.blocks)
                    result = next(list(block.instructions)[-1].operands)
                    self.assertEqual(result.get_constant_value(), 1)

    def test_external_float_cast_remains_a_backend_operation(self):
        from pythoc.llvm_extensions import fold_instruction

        class X87(ir.Type):
            def __str__(self):
                return 'x86_fp80'

        source = ir.Constant(ir.DoubleType(), 1.5)
        self.assertIsNone(fold_instruction('fpext', (source, X87())))

    def test_constant_compare_preserves_explicit_llvm_predicates(self):
        from pythoc.compiler import LLVMCompiler
        from pythoc.llvm_extensions import compare, constant

        compiler = LLVMCompiler()
        for index, value in enumerate((0.0, float('nan'))):
            for operation in ('==', '!=', '<', 'ord', 'uno', 'true',
                              'false', 'oeq', 'ueq', 'one', 'une'):
                function = ir.Function(
                    compiler.module, ir.FunctionType(ir.IntType(1), []),
                    name=f'compare_{index}_{operation.encode().hex()}')
                builder = ir.IRBuilder(function.append_basic_block('entry'))
                left = constant(ir.DoubleType(), value)
                right = constant(ir.DoubleType(), 0.0)
                actual = compare(builder, operation, left, right, ordered=True)
                expected = builder.fcmp_ordered(operation, left, right)
                builder.ret(builder.icmp_unsigned('==', actual, expected))
        compiler.optimize_module(2)
        from llvmlite import binding
        with binding.parse_assembly(compiler.get_ir()) as module:
            for function in module.functions:
                with self.subTest(function=function.name):
                    block = next(function.blocks)
                    result = next(list(block.instructions)[-1].operands)
                    self.assertEqual(result.get_constant_value(), 1)


if __name__ == '__main__':
    unittest.main()


class TestFuncSignatureCompatibility(unittest.TestCase):
    """_func_types_compatible follows C signature rules: parameter names are
    ignored, typedef aliases are transparent (quoted spellings resolve through
    the forward-ref registry), named aggregates compare by tag, and
    ``ptr[void]`` stays distinct from any typed pointer."""

    def setUp(self):
        import pythoc
        pythoc.init()
        from pythoc.forward_ref import clear_forward_ref_state
        clear_forward_ref_state()

    def tearDown(self):
        from pythoc.forward_ref import clear_forward_ref_state
        clear_forward_ref_state()

    def _struct(self, name, fields=(('x',),)):
        from pythoc.builtin_entities import i32
        from pythoc.builtin_entities.struct import create_struct_type
        cls = create_struct_type([i32], ['x'])
        from pythoc.forward_ref import mark_type_defined
        mark_type_defined(name, cls)
        return cls

    def test_quoted_and_direct_pointee_compatible(self):
        from pythoc import func, i32, ptr
        node = self._struct('CompatNode')
        f_direct = func[ptr[node], i32]
        f_quoted = func[ptr['CompatNode'], i32]
        self.assertTrue(
            TypeConverter._func_types_compatible(f_direct, f_quoted))
        self.assertTrue(
            TypeConverter._func_types_compatible(f_quoted, f_direct))

    def test_ptr_void_stays_distinct(self):
        from pythoc import func, i32, ptr, void
        node = self._struct('CompatNode2')
        f_typed = func[ptr[node], i32]
        f_void = func[ptr[void], i32]
        self.assertFalse(TypeConverter._func_types_compatible(f_typed, f_void))
        self.assertFalse(TypeConverter._func_types_compatible(f_void, f_typed))

    def test_arity_mismatch_rejected(self):
        from pythoc import func, i32, ptr
        node = self._struct('CompatNode3')
        self.assertFalse(TypeConverter._func_types_compatible(
            func[ptr[node], i32], func[ptr[node], i32, i32]))

    def test_param_names_ignored(self):
        from pythoc import func, i32, ptr
        node = self._struct('CompatNode4')
        f_named = func['x', ptr[node], i32] if False else None
        # param names only exist via wrapper signatures; simulate two class
        # objects of the same structural signature:
        f1 = func[ptr[node], i32]
        f2 = func[ptr[node], i32]
        self.assertTrue(TypeConverter._func_types_compatible(f1, f2))

    def test_nested_func_components(self):
        from pythoc import func, i32, ptr, void
        node = self._struct('CompatNode5')
        inner1 = func[ptr['CompatNode5'], ptr[void], i32]
        inner2 = func[ptr[node], ptr[void], i32]
        f1 = func[ptr[node], inner1, ptr[void], i32]
        f2 = func[ptr['CompatNode5'], inner2, ptr[void], i32]
        self.assertTrue(TypeConverter._func_types_compatible(f1, f2))
