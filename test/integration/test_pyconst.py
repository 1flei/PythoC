"""
Test pyconst and typeof features
"""
import unittest
from pythoc import *


# Define struct outside @compile
TestStruct = struct[('a', i32), ('b', pyconst[42])]


@compile
def test_pyconst_field_access() -> i32:
    """Test accessing pyconst field in struct"""
    
    @compile(suffix=TestStruct)
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


if __name__ == "__main__":
    unittest.main()
