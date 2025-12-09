"""
Test user_globals capture for yield/inline/closure across different files

CRITICAL TEST SCENARIO:
- Callee site (user_globals_lib/*.py): Defines inline/yield/closure functions
  that use module-level globals (type aliases, constants, helpers)
- Caller site (this file): Imports and uses those functions but does NOT
  have access to the callee's module-level globals

The kernel must capture callee's __globals__ and merge them into the
compilation context so that inlined code can resolve:
1. Type aliases defined in callee's module
2. Constants defined in callee's module  
3. Helper functions defined in callee's module

Without proper user_globals capture, these tests will fail with
"name 'LocalInt' is not defined" or similar errors.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pythoc import compile, i32
from pythoc.build.output_manager import flush_all_pending_outputs

# Import callee functions from separate modules
# These modules have their own globals that THIS module cannot see
from test.integration.user_globals_lib.callee_inline import (
    inline_using_type_alias,
    inline_using_constant,
    inline_using_local_alias_only,
)
from test.integration.user_globals_lib.callee_yield import (
    yield_using_type_alias,
    yield_using_constant,
    yield_using_helper,
)
from test.integration.user_globals_lib.callee_closure import (
    func_with_closure_using_type_alias,
    func_with_closure_using_constant,
    func_with_closure_using_helper,
)


# =============================================================================
# Caller site functions - these use imported functions from other modules
# The callee's globals (LocalInt, YIELD_MAGIC, etc.) are NOT in this module
# =============================================================================

# -----------------------------------------------------------------------------
# Inline tests: call @inline functions from callee_inline.py
# -----------------------------------------------------------------------------

@compile
def test_inline_cross_file_type_alias() -> i32:
    """
    Call inline function that uses LocalInt type alias.
    
    LocalInt is defined in callee_inline.py, NOT here.
    Without proper globals capture, this fails with "LocalInt not defined".
    """
    result = inline_using_type_alias(i32(50))
    return result  # Should be 50 + 100 = 150


@compile
def test_inline_cross_file_constant() -> i32:
    """
    Call inline function that uses LOCAL_MAGIC constant.
    
    LOCAL_MAGIC is defined in callee_inline.py, NOT here.
    """
    return inline_using_constant(i32(25))  # Should be 25 + 100 = 125


@compile
def test_inline_cross_file_local_alias_only() -> i32:
    """
    Call inline function that only uses local type alias.
    """
    return inline_using_local_alias_only()  # Should be 42


# -----------------------------------------------------------------------------
# Yield tests: iterate yield functions from callee_yield.py
# -----------------------------------------------------------------------------

@compile
def test_yield_cross_file_type_alias() -> i32:
    """
    Iterate yield function that uses YieldInt type alias.
    
    YieldInt is defined in callee_yield.py, NOT here.
    """
    total: i32 = 0
    for val in yield_using_type_alias():
        total = total + val
    return total  # Should be 10 + 20 = 30


@compile
def test_yield_cross_file_constant() -> i32:
    """
    Iterate yield function that uses YIELD_MAGIC constant.
    
    YIELD_MAGIC is defined in callee_yield.py, NOT here.
    """
    total: i32 = 0
    for val in yield_using_constant():
        total = total + val
    return total  # Should be 200 + 250 = 450


@compile
def test_yield_cross_file_helper() -> i32:
    """
    Iterate yield function that uses yield_helper_add.
    
    yield_helper_add is defined in callee_yield.py, NOT here.
    """
    total: i32 = 0
    for val in yield_using_helper():
        total = total + val
    return total  # Should be 15 + 27 = 42


# -----------------------------------------------------------------------------
# Closure tests: call functions that use closures from callee_closure.py
# -----------------------------------------------------------------------------

@compile
def test_closure_cross_file_type_alias() -> i32:
    """
    Call function with closure using ClosureInt type alias.
    
    ClosureInt is defined in callee_closure.py, NOT here.
    """
    return func_with_closure_using_type_alias(i32(5))  # Should be 10 + 5 = 15


@compile
def test_closure_cross_file_constant() -> i32:
    """
    Call function with closure using CLOSURE_MAGIC constant.
    
    CLOSURE_MAGIC is defined in callee_closure.py, NOT here.
    """
    return func_with_closure_using_constant(i32(7))  # Should be 7 + 300 = 307


@compile
def test_closure_cross_file_helper() -> i32:
    """
    Call function with closure using closure_helper_mul.
    
    closure_helper_mul is defined in callee_closure.py, NOT here.
    """
    return func_with_closure_using_helper(i32(6), i32(7))  # Should be 6 * 7 = 42


# =============================================================================
# Test runner
# =============================================================================

class TestUserGlobalsCrossFile(unittest.TestCase):
    """
    Test suite for cross-file user_globals capture.
    
    These tests verify that inline/yield/closure correctly capture
    globals from their DEFINITION site (callee module), not just
    from the CALL site (this module).
    """
    
    @classmethod
    def setUpClass(cls):
        """Compile all functions before running tests"""
        flush_all_pending_outputs()
    
    # =========================================================================
    # Inline tests - @inline functions from callee_inline.py
    # =========================================================================
    
    def test_inline_type_alias(self):
        """Test inline captures type alias from callee module"""
        result = test_inline_cross_file_type_alias()
        self.assertEqual(result, 150)
    
    def test_inline_constant(self):
        """Test inline captures constant from callee module"""
        result = test_inline_cross_file_constant()
        self.assertEqual(result, 125)
    
    def test_inline_local_alias_only(self):
        """Test inline captures local alias from callee module"""
        result = test_inline_cross_file_local_alias_only()
        self.assertEqual(result, 42)
    
    # =========================================================================
    # Yield tests - yield functions from callee_yield.py
    # =========================================================================
    
    def test_yield_type_alias(self):
        """Test yield captures type alias from callee module"""
        result = test_yield_cross_file_type_alias()
        self.assertEqual(result, 30)
    
    def test_yield_constant(self):
        """Test yield captures constant from callee module"""
        result = test_yield_cross_file_constant()
        self.assertEqual(result, 450)
    
    def test_yield_helper(self):
        """Test yield captures helper function from callee module"""
        result = test_yield_cross_file_helper()
        self.assertEqual(result, 42)
    
    # =========================================================================
    # Closure tests - closure functions from callee_closure.py
    # =========================================================================
    
    def test_closure_type_alias(self):
        """Test closure captures type alias from callee module"""
        result = test_closure_cross_file_type_alias()
        self.assertEqual(result, 15)
    
    def test_closure_constant(self):
        """Test closure captures constant from callee module"""
        result = test_closure_cross_file_constant()
        self.assertEqual(result, 307)
    
    def test_closure_helper(self):
        """Test closure captures helper function from callee module"""
        result = test_closure_cross_file_helper()
        self.assertEqual(result, 42)


if __name__ == '__main__':
    unittest.main()
