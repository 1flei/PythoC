"""
Test utilities for pythoc integration tests

This module provides:
1. ErrorTestRunner - Run tests that expect compilation errors
2. expect_error decorator - Simplify error test helper functions
3. DeferredTestCase - Base class for tests with proper setup
"""

import os
import unittest
from typing import List, Callable, Tuple, Optional
from dataclasses import dataclass

from pythoc.build.output_manager import flush_all_pending_outputs, clear_failed_group
from pythoc.logger import set_raise_on_error


@dataclass
class ErrorTestResult:
    """Result of an error test"""
    passed: bool
    message: str


class ErrorTestRunner:
    """
    Runner for tests that expect compilation errors.
    
    Handles:
    1. Setting up error raising mode
    2. Catching expected errors
    3. Cleaning up failed groups
    4. Verifying error messages contain expected patterns
    """
    
    @staticmethod
    def run(
        define_func: Callable[[], None],
        expected_patterns: List[str],
        suffix: str,
        source_file: Optional[str] = None
    ) -> ErrorTestResult:
        """
        Run a test that expects a compilation error.
        
        Args:
            define_func: Callable that defines the @compile decorated function
            expected_patterns: List of patterns (any one must match in error message)
            suffix: The suffix used in @compile(suffix=...)
            source_file: Source file path (auto-detected if None)
            
        Returns:
            ErrorTestResult with passed status and message
        """
        if source_file is None:
            # Get caller's file
            import inspect
            frame = inspect.currentframe()
            if frame and frame.f_back:
                source_file = os.path.abspath(frame.f_back.f_code.co_filename)
            else:
                source_file = __file__
        
        group_key = (source_file, 'module', suffix)
        
        try:
            define_func()
            flush_all_pending_outputs()
            return ErrorTestResult(
                passed=False,
                message="Expected error but none was raised"
            )
        except (RuntimeError, SyntaxError, TypeError, ValueError) as e:
            err_str = str(e).lower()
            for pattern in expected_patterns:
                if pattern.lower() in err_str:
                    return ErrorTestResult(
                        passed=True,
                        message=str(e)
                    )
            return ErrorTestResult(
                passed=False,
                message=f"Error raised but no expected pattern matched. Error: {e}"
            )
        finally:
            clear_failed_group(group_key)


def expect_error(
    expected_patterns: List[str],
    suffix: str
) -> Callable[[Callable], Callable[[], Tuple[bool, str]]]:
    """
    Decorator for error test helper functions.
    
    Usage:
        @expect_error(["inconsistent", "not consumed"], suffix="bad_case")
        def run_error_test_bad_case():
            @compile(suffix="bad_case")
            def bad_case() -> void:
                t = linear()
                # missing consume
        
        # In test:
        passed, msg = run_error_test_bad_case()
        self.assertTrue(passed, msg)
    
    Args:
        expected_patterns: List of patterns (any one must match)
        suffix: The suffix used in @compile
        
    Returns:
        Decorator that wraps the function
    """
    def decorator(func: Callable[[], None]) -> Callable[[], Tuple[bool, str]]:
        def wrapper() -> Tuple[bool, str]:
            # Get the source file from the original function
            source_file = os.path.abspath(func.__code__.co_filename)
            
            result = ErrorTestRunner.run(
                define_func=func,
                expected_patterns=expected_patterns,
                suffix=suffix,
                source_file=source_file
            )
            return result.passed, result.message
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator


class DeferredTestCase(unittest.TestCase):
    """
    Base class for pythoc integration tests.
    
    Features:
    1. Automatically enables error raising mode
    2. Provides common setup/teardown
    """
    
    @classmethod
    def setUpClass(cls):
        """Enable error raising for all tests in this class"""
        set_raise_on_error(True)
    
    @classmethod
    def tearDownClass(cls):
        """Cleanup after all tests"""
        pass
