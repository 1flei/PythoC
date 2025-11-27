"""
Unit tests for meta programming utilities
"""

import unittest
from pythoc.builtin_entities import static


class TestStaticQualifier(unittest.TestCase):
    """Test static type qualifier"""
    
    def test_static_is_builtin_type(self):
        """Test static is a builtin type"""
        self.assertTrue(hasattr(static, 'get_name'))
        self.assertEqual(static.get_name(), 'static')
    
    def test_static_can_be_type(self):
        """Test static can be used as type"""
        self.assertTrue(static.can_be_type())


if __name__ == '__main__':
    unittest.main()
