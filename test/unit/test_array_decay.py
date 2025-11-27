"""
Unit tests for array.get_decay_pointer_type()
"""

import unittest
from pythoc.builtin_entities.array import array
from pythoc.builtin_entities.types import i32, f64, ptr


class TestArrayDecayPointerType(unittest.TestCase):
    """Test array decay to pointer type conversion"""
    
    def test_single_dimensional_array_decay(self):
        """Test array[i32, 10] -> ptr[i32]"""
        arr_type = array[i32, 10]
        ptr_type = arr_type.get_decay_pointer_type()
        
        # Check it's a ptr type
        self.assertTrue(issubclass(ptr_type, ptr))
        
        # Check element type is i32
        self.assertIs(ptr_type.pointee_type, i32)
    
    def test_single_dimensional_f64_array_decay(self):
        """Test array[f64, 5] -> ptr[f64]"""
        arr_type = array[f64, 5]
        ptr_type = arr_type.get_decay_pointer_type()
        
        self.assertTrue(issubclass(ptr_type, ptr))
        self.assertIs(ptr_type.pointee_type, f64)
    
    def test_two_dimensional_array_decay(self):
        """Test array[i32, 3, 5] -> ptr[array[i32, 5]]"""
        arr_type = array[i32, 3, 5]
        ptr_type = arr_type.get_decay_pointer_type()
        
        # Should be ptr[array[i32, 5]]
        self.assertTrue(issubclass(ptr_type, ptr))
        
        # Check pointee is array type
        pointee = ptr_type.pointee_type
        self.assertTrue(issubclass(pointee, array))
        
        # Check inner array element type and dimensions
        self.assertIs(pointee.element_type, i32)
        self.assertEqual(pointee.dimensions, (5,))
    
    def test_three_dimensional_array_decay(self):
        """Test array[i32, 2, 3, 4] -> ptr[array[i32, 3, 4]]"""
        arr_type = array[i32, 2, 3, 4]
        ptr_type = arr_type.get_decay_pointer_type()
        
        # Should be ptr[array[i32, 3, 4]]
        self.assertTrue(issubclass(ptr_type, ptr))
        
        pointee = ptr_type.pointee_type
        self.assertTrue(issubclass(pointee, array))
        
        # Check it's a 2D array (dimensions 3, 4)
        self.assertEqual(len(pointee.dimensions), 2)
        self.assertEqual(pointee.dimensions, (3, 4))
        
        # Check element type
        self.assertIs(pointee.element_type, i32)
    
    def test_array_without_element_type_raises(self):
        """Test that decay fails for unspecialized array"""
        with self.assertRaises(TypeError) as ctx:
            array.get_decay_pointer_type()
        
        self.assertIn("Cannot decay array without element_type", str(ctx.exception))
    
    def test_decay_type_name(self):
        """Test that decayed type has correct name"""
        arr_type = array[i32, 10]
        ptr_type = arr_type.get_decay_pointer_type()
        
        # Should have ptr name (specialized ptr types include pointee in name)
        name = ptr_type.get_name()
        self.assertTrue(name.startswith('ptr'))
    
    def test_decay_is_idempotent_via_ptr(self):
        """Test that decaying twice gives same result as ptr[elem]"""
        arr_type = array[i32, 10]
        decayed = arr_type.get_decay_pointer_type()
        
        # Decaying array[i32, 10] should give same type as ptr[i32]
        expected = ptr[i32]
        
        # Both should point to i32
        self.assertIs(decayed.pointee_type, expected.pointee_type)


if __name__ == '__main__':
    unittest.main()
