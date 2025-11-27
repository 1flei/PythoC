"""
Unit tests for libc bindings
"""

import unittest
from pythoc.libc import printf, malloc, free, memcpy, strlen


class TestLibcStdio(unittest.TestCase):
    """Test libc stdio bindings"""
    
    def test_printf_exists(self):
        """Test printf binding exists"""
        self.assertIsNotNone(printf)
        self.assertTrue(callable(printf))


class TestLibcMemory(unittest.TestCase):
    """Test libc memory bindings"""
    
    def test_malloc_exists(self):
        """Test malloc binding exists"""
        self.assertIsNotNone(malloc)
        self.assertTrue(callable(malloc))
    
    def test_free_exists(self):
        """Test free binding exists"""
        self.assertIsNotNone(free)
        self.assertTrue(callable(free))
    
    def test_memcpy_exists(self):
        """Test memcpy binding exists"""
        self.assertIsNotNone(memcpy)
        self.assertTrue(callable(memcpy))


class TestLibcString(unittest.TestCase):
    """Test libc string bindings"""
    
    def test_strlen_exists(self):
        """Test strlen binding exists"""
        self.assertIsNotNone(strlen)
        self.assertTrue(callable(strlen))


if __name__ == '__main__':
    unittest.main()
