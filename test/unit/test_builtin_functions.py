"""
Unit tests for builtin function entities
"""

import unittest
from llvmlite import ir
import ast

from pythoc.builtin_entities import (
    sizeof,
    get_builtin_entity
)


class TestSizeofFunction(unittest.TestCase):
    """Test sizeof builtin function"""
    
    def test_sizeof_properties(self):
        """Test sizeof basic properties"""
        self.assertEqual(sizeof.get_name(), "sizeof")
        self.assertTrue(sizeof.can_be_called())
        self.assertFalse(sizeof.can_be_type())
    
    def test_sizeof_in_registry(self):
        """Test sizeof is registered"""
        entity = get_builtin_entity("sizeof")
        self.assertEqual(entity, sizeof)


class TestBuiltinFunctionRegistry(unittest.TestCase):
    """Test builtin function registry"""
    
    def test_get_sizeof_from_registry(self):
        """Test getting sizeof from registry"""
        entity = get_builtin_entity("sizeof")
        self.assertEqual(entity, sizeof)
        self.assertTrue(entity.can_be_called())


if __name__ == '__main__':
    unittest.main()
