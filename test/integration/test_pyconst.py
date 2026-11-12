"""
Test pyconst and typeof features
"""
import ast
from unittest.mock import patch
from pythoc import CompileSession, LLVMCompiler
from pythoc.logger import logger
import unittest
from pythoc import *


# Define struct outside @compile
TestStruct = struct[('a', i32), ('b', pyconst[42])]


@compile
def test_pyconst_field_access() -> i32:
    """Test accessing pyconst field in struct"""

    def get_b(x: TestStruct) -> i32:
        return i32(x.b)  # Should return constant 42
    
    s: TestStruct
    s.a = 10
    s.b = 42  # Should type-check and be no-op
    
    return get_b(s)


class TestPyconst(unittest.TestCase):
    def test_pyconst_basic(self):
        """Test basic pyconst type"""
        T = pyconst[42]
        self.assertTrue(hasattr(T, 'get_size_bytes'))
        self.assertEqual(T.get_size_bytes(), 0)
    
    def test_typeof_basic(self):
        """Test basic typeof functionality"""
        t1 = typeof(5)
        t2 = typeof(100)
        self.assertIsNot(t1, t2)  # Different values = different types
    
    def test_struct_with_pyconst(self):
        """Test struct with pyconst fields"""
        S = struct[('a', i32), ('b', pyconst[42]), ('c', i32)]
        # Size should exclude pyconst field (zero-sized)
        # sizeof(S) == sizeof(i32) + sizeof(i32) = 8
        expected_size = 8
        actual_size = S.get_size_bytes()
        self.assertEqual(actual_size, expected_size)
    
    def test_typeof_in_struct(self):
        """Test using typeof in struct definition"""
        def Vec(T, size_spec):
            return struct[('size', typeof(size_spec)), ('data', ptr[T])]
        
        # Static vector - size is pyconst[100]
        StaticVec = Vec(i32, 100)
        # Dynamic vector - size is i32
        DynamicVec = Vec(i32, i32)
        
        static_size = StaticVec.get_size_bytes()
        dynamic_size = DynamicVec.get_size_bytes()
        
        # Static: only ptr (8 bytes on 64-bit)
        # Dynamic: i32 + ptr (12-16 bytes depending on alignment)
        self.assertLess(static_size, dynamic_size)
    
    def test_pyconst_field_access(self):
        """Test compiled function with pyconst field"""
        result = test_pyconst_field_access()
        self.assertEqual(result, 42)

    def test_typed_constants_work_as_python_indices(self):
        self.assertEqual(typed_constant_indices(), 70)


TABLE = (10, 20)


MAPPING = {-1: 40, 1: 20}


@compile
def typed_constant_indices() -> i32:
    return TABLE[i32(1)] + TABLE[i32(0)] + MAPPING[i32(-1)]


class TestPythonConstantIndexing(unittest.TestCase):
    def test_runtime_index_is_rejected_by_normal_compilation(self):
        syntax = ast.parse('def read(index):\n    return TABLE[index]').body[0]
        with CompileSession(), patch.object(logger, 'raise_on_error', True):
            compiler = LLVMCompiler(user_globals={'TABLE': (10, 20)})
            with self.assertRaisesRegex(TypeError, 'compile-time constant index'):
                compiler.compile_function_from_ast(
                    syntax, param_type_hints={'index': i32}, return_type_hint=i32)


if __name__ == "__main__":
    unittest.main()
