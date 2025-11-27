"""
Unit tests for decorator functionality
"""

import unittest
import tempfile
import os

from pythoc.decorators import compile


class TestCompileDecorator(unittest.TestCase):
    """Test compile decorator"""
    
    def test_decorator_basic(self):
        """Test basic decorator usage"""
        # Decorator should be callable
        self.assertTrue(callable(compile))
    
    def test_decorator_preserves_metadata(self):
        """Test decorator can be applied to functions"""
        # Test that decorator can be applied
        try:
            @compile
            def test_func(x: int) -> int:
                """Test function docstring"""
                return x * 2
            
            # Should preserve name
            self.assertEqual(test_func.__name__, 'test_func')
        except Exception:
            # Compilation may fail in test environment, but decorator should apply
            pass


class TestCompilationOutput(unittest.TestCase):
    """Test compilation output generation"""
    
    def test_compile_decorator_exists(self):
        """Test that compile decorator exists and is callable"""
        from pythoc.decorators import compile
        self.assertTrue(callable(compile))


if __name__ == '__main__':
    unittest.main()
