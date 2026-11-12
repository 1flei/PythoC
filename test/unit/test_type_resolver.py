"""
Unit tests for TypeResolver
"""

import unittest
import ast
from llvmlite import ir

from pythoc.type_resolver import TypeResolver
from pythoc.builtin_entities import i32, i64, f32, f64, u8, u16, ptr, array
from pythoc.builtin_entities.python_type import PythonType


class TestTypeResolver(unittest.TestCase):
    """Test TypeResolver functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.module = ir.Module(name="test_module")
        # Create user_globals with all builtin entities (simulate 'from pythoc import *')
        user_globals = {
            'i32': i32, 'i64': i64, 'f32': f32, 'f64': f64,
            'u8': u8, 'u16': u16,
            'ptr': ptr, 'array': array,
        }
        
        self.resolver = TypeResolver(
            module_context=self.module,
            user_globals=user_globals  # 添加这个参数
        )
    
    def test_parse_simple_type_from_string(self):
        """Test parsing simple type from string"""
        result = self.resolver.parse_annotation("i32")
        self.assertEqual(result, i32)
        
        result = self.resolver.parse_annotation("f64")
        self.assertEqual(result, f64)
    
    def test_parse_simple_type_from_ast_name(self):
        """Test parsing simple type from AST Name node"""
        node = ast.Name(id="i32")
        result = self.resolver.parse_annotation(node)
        self.assertEqual(result, i32)
    
    def test_parse_type_class_directly(self):
        """Test parsing type class directly"""
        result = self.resolver.parse_annotation(i32)
        self.assertEqual(result, i32)

    def test_wrap_direct_type_class_as_pyconst(self):
        """Test direct non-AST type input uses canonical pyconst wrapping"""
        value_ref = self.resolver._wrap_as_python_value(i32)
        self.assertTrue(value_ref.is_python_value())
        self.assertEqual(value_ref.value, i32)
        self.assertIsInstance(value_ref.type_hint, PythonType)
        self.assertIs(value_ref.type_hint.get_python_object(), i32)
    
    def test_parse_unsigned_types(self):
        """Test parsing unsigned integer types"""
        result = self.resolver.parse_annotation("u8")
        self.assertEqual(result, u8)
        
        result = self.resolver.parse_annotation("u16")
        self.assertEqual(result, u16)
    
    def test_parse_pointer_type(self):
        """Test parsing pointer type ptr[T]"""
        # Create AST for ptr[i32]
        node = ast.Subscript(
            value=ast.Name(id="ptr"),
            slice=ast.Name(id="i32")
        )
        
        result = self.resolver.parse_annotation(node)
        
        # Should return a specialized ptr class
        self.assertTrue(hasattr(result, 'get_name'))
        self.assertTrue(result.can_be_type())
    
    def test_parse_array_type_1d(self):
        """Test parsing 1D array type array[T, N]"""
        # Create AST for array[i32, 10]
        node = ast.Subscript(
            value=ast.Name(id="array"),
            slice=ast.Tuple(elts=[
                ast.Name(id="i32"),
                ast.Constant(value=10)
            ])
        )
        
        result = self.resolver.parse_annotation(node)
        
        # Should return a specialized array class
        self.assertTrue(hasattr(result, 'get_name'))
        self.assertTrue(result.can_be_type())
    
    def test_parse_array_type_2d(self):
        """Test parsing 2D array type array[T, N, M]"""
        # Create AST for array[i32, 10, 20]
        node = ast.Subscript(
            value=ast.Name(id="array"),
            slice=ast.Tuple(elts=[
                ast.Name(id="i32"),
                ast.Constant(value=10),
                ast.Constant(value=20)
            ])
        )
        
        result = self.resolver.parse_annotation(node)
        
        self.assertTrue(hasattr(result, 'get_name'))
        self.assertTrue(result.can_be_type())
    
    def test_annotation_to_llvm_type_simple(self):
        """Test converting simple type annotation to LLVM type"""
        node = ast.Name(id="i32")
        result = self.resolver.annotation_to_llvm_type(node)
        
        self.assertEqual(result, ir.IntType(32))
    
    def test_annotation_to_llvm_type_float(self):
        """Test converting float type annotation to LLVM type"""
        node = ast.Name(id="f32")
        result = self.resolver.annotation_to_llvm_type(node)
        
        self.assertEqual(result, ir.FloatType())
    
    def test_annotation_to_llvm_type_double(self):
        """Test converting double type annotation to LLVM type"""
        node = ast.Name(id="f64")
        result = self.resolver.annotation_to_llvm_type(node)
        
        self.assertEqual(result, ir.DoubleType())
    
    def test_annotation_to_llvm_type_pointer(self):
        """Test converting pointer type annotation to LLVM type"""
        # Create AST for ptr[i32]
        node = ast.Subscript(
            value=ast.Name(id="ptr"),
            slice=ast.Name(id="i32")
        )
        
        result = self.resolver.annotation_to_llvm_type(node)
        
        self.assertIsInstance(result, ir.PointerType)
        self.assertEqual(result.pointee, ir.IntType(32))
    
    def test_annotation_to_llvm_type_array(self):
        """Test converting array type annotation to LLVM type"""
        # Create AST for array[i32, 10]
        node = ast.Subscript(
            value=ast.Name(id="array"),
            slice=ast.Tuple(elts=[
                ast.Name(id="i32"),
                ast.Constant(value=10)
            ])
        )
        
        result = self.resolver.annotation_to_llvm_type(node)
        
        self.assertIsInstance(result, ir.ArrayType)
        self.assertEqual(result.element, ir.IntType(32))
        self.assertEqual(result.count, 10)
    
    def test_parse_invalid_type(self):
        """Test parsing invalid type raises NameError"""
        with self.assertRaises(NameError):
            self.resolver.parse_annotation("invalid_type")
    
    def test_parse_none_annotation(self):
        """Test parsing None annotation"""
        result = self.resolver.parse_annotation(None)
        self.assertIsNone(result)
    
    def test_nested_pointer_type(self):
        """Test parsing nested pointer type ptr[ptr[i32]]"""
        # Create AST for ptr[ptr[i32]]
        inner = ast.Subscript(
            value=ast.Name(id="ptr"),
            slice=ast.Name(id="i32")
        )
        node = ast.Subscript(
            value=ast.Name(id="ptr"),
            slice=inner
        )
        
        result = self.resolver.annotation_to_llvm_type(node)
        
        self.assertIsInstance(result, ir.PointerType)
        self.assertIsInstance(result.pointee, ir.PointerType)
        self.assertEqual(result.pointee.pointee, ir.IntType(32))

    def test_quoted_annotations_expose_the_same_names(self):
        expression = TypeResolver.annotation_expression(
            ast.Constant('"array[i32, 2]"'))
        self.assertEqual(ast.dump(expression),
                         ast.dump(ast.parse('array[i32, 2]', mode='eval').body))


    def test_definition_captures_annotation_only_local(self):
        local_type = i64

        def declaration(value: 'local_type'):
            pass

        namespace = TypeResolver.capture_type_namespace(definition=declaration)
        self.assertIs(namespace['local_type'], local_type)

    def test_method_definition_captures_class_body(self):
        class Owner:
            local_type = i64

            def declaration(value: 'local_type'):
                pass

            namespace = TypeResolver.capture_type_namespace(definition=declaration)

        self.assertIs(Owner.namespace['local_type'], i64)

    def test_class_definition_captures_its_factory(self):
        local_type = i64

        class Record:
            pass

        namespace = TypeResolver.capture_type_namespace(
            {'local_type'}, definition=Record)
        self.assertIs(namespace['local_type'], local_type)

    def test_same_named_caller_is_not_a_definition_scope(self):
        class Origin:
            @staticmethod
            def factory():
                def declaration(value: 'local_type'):
                    pass
                return declaration

        class Consumer:
            @staticmethod
            def factory(definition):
                local_type = i32
                return TypeResolver.capture_type_namespace(definition=definition)

        self.assertNotIn('local_type', Consumer.factory(Origin.factory()))


if __name__ == '__main__':
    unittest.main()
