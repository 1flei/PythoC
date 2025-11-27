"""
Unit tests for utility functions
"""

import unittest
from llvmlite import ir, binding

from pythoc.utils import (
    analyze_function, get_llvm_version, validate_ir,
    create_build_info
)


class TestAnalyzeFunction(unittest.TestCase):
    """Test function analysis utility"""
    
    def test_analyze_simple_function(self):
        """Test analyzing a simple function"""
        source = """
def add(x: i32, y: i32) -> i32:
    return x + y
"""
        info = analyze_function(source, "add")
        
        self.assertEqual(info['name'], 'add')
        self.assertEqual(len(info['parameters']), 2)
        self.assertEqual(info['parameters'][0]['name'], 'x')
        self.assertEqual(info['parameters'][0]['type'], 'i32')
        self.assertEqual(info['return_type'], 'i32')
        self.assertIn('add', info['operations'])
    
    def test_analyze_function_with_control_flow(self):
        """Test analyzing function with control flow"""
        source = """
def max_val(x: i32, y: i32) -> i32:
    if x > y:
        return x
    return y
"""
        info = analyze_function(source, "max_val")
        
        self.assertTrue(info['has_control_flow'])
        self.assertIn('compare', info['operations'])
    
    def test_analyze_function_with_operations(self):
        """Test analyzing function with various operations"""
        source = """
def compute(x: i32) -> i32:
    a = x + 10
    b = a - 5
    c = b * 2
    d = c / 4
    return -d
"""
        info = analyze_function(source, "compute")
        
        self.assertIn('add', info['operations'])
        self.assertIn('sub', info['operations'])
        self.assertIn('mul', info['operations'])
        self.assertIn('div', info['operations'])
        self.assertIn('neg', info['operations'])
    
    def test_analyze_nonexistent_function(self):
        """Test analyzing non-existent function"""
        source = """
def foo():
    pass
"""
        info = analyze_function(source, "bar")
        
        self.assertEqual(info, {})


class TestLLVMUtilities(unittest.TestCase):
    """Test LLVM-related utilities"""
    
    def test_get_llvm_version(self):
        """Test getting LLVM version"""
        version = get_llvm_version()
        self.assertIsNotNone(version)
    
    def test_validate_ir_valid(self):
        """Test validating valid IR"""
        module = ir.Module(name="test")
        func_type = ir.FunctionType(ir.IntType(32), [ir.IntType(32)])
        func = ir.Function(module, func_type, name="test_func")
        block = func.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)
        builder.ret(func.args[0])
        
        ir_code = str(module)
        result = validate_ir(ir_code)
        self.assertTrue(result)
    
    def test_validate_ir_invalid(self):
        """Test validating invalid IR"""
        invalid_ir = "invalid llvm ir code"
        
        with self.assertRaises(Exception):
            validate_ir(invalid_ir)
    
    def test_create_build_info(self):
        """Test creating build information"""
        info = create_build_info()
        
        self.assertIn('llvm_version', info)
        self.assertIn('target_triple', info)
        self.assertIn('host_cpu', info)
        self.assertIn('host_cpu_features', info)
        
        self.assertIsNotNone(info['llvm_version'])
        self.assertIsNotNone(info['target_triple'])


if __name__ == '__main__':
    unittest.main()
