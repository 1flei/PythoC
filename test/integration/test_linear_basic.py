"""
Basic tests for linear token functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import unittest
from pythoc.decorators.compile import compile
from pythoc.builtin_entities import linear, consume, void
from pythoc.std.utility import move
from pythoc.build.output_manager import flush_all_pending_outputs

from test.utils.test_utils import DeferredTestCase, expect_error


# =============================================================================
# Error test helper functions using @expect_error decorator
# =============================================================================

@expect_error(["not consumed"], suffix="bad_not_consumed")
def run_error_not_consumed():
    @compile(suffix="bad_not_consumed")
    def bad_not_consumed() -> void:
        t = linear()
        # Missing consume(t)


@expect_error(["consumed"], suffix="bad_double_consume")
def run_error_double_consume():
    @compile(suffix="bad_double_consume")
    def bad_double_consume() -> void:
        t = linear()
        consume(t)
        consume(t)  # ERROR


@expect_error(["consumed"], suffix="bad_use_after_move")
def run_error_use_after_move():
    @compile(suffix="bad_use_after_move")
    def bad_use_after_move() -> void:
        t = linear()
        t2 = move(t)
        consume(t)  # ERROR: t was consumed by move()
        consume(t2)


@expect_error(["cannot assign linear token", "use move()"], suffix="bad_copy_linear")
def run_error_assignment():
    @compile(suffix="bad_copy_linear")
    def bad_copy_linear() -> void:
        t = linear()
        t2 = t  # ERROR: cannot copy
        consume(t2)


@expect_error(["not consumed", "cannot reassign"], suffix="bad_reassign_linear")
def run_error_reassignment():
    @compile(suffix="bad_reassign_linear")
    def bad_reassign_linear() -> void:
        t = linear()
        t = linear()  # ERROR: first token not consumed
        consume(t)


@expect_error(["undefined", "cannot consume", "consumed"], suffix="bad_undefined_consume")
def run_error_undefined_consume():
    @compile(suffix="bad_undefined_consume", anonymous=True)
    def bad() -> void:
        t: linear
        consume(t)


# =============================================================================
# Test class
# =============================================================================

class TestLinearBasic(DeferredTestCase):
    """Basic tests for linear token functionality"""
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Define all valid functions first
        cls._define_valid_functions()
        # Flush to compile them
        flush_all_pending_outputs()
    
    @classmethod
    def _define_valid_functions(cls):
        """Define all valid @compile functions"""
        
        @compile(suffix="test_create_consume")
        def test() -> void:
            t = linear()
            consume(t)
        cls.test_create_consume = test
        
        @compile(suffix="use_token")
        def use_token(t: linear) -> void:
            consume(t)
        cls.use_token = use_token
        
        @compile(suffix="create_and_pass")
        def create_and_pass() -> void:
            t = linear()
            use_token(t)
        cls.create_and_pass = create_and_pass
        
        @compile(suffix="create_token")
        def create_token() -> linear:
            t = linear()
            return t
        cls.create_token = create_token
        
        @compile(suffix="use_returned")
        def use_returned() -> void:
            t = create_token()
            consume(t)
        cls.use_returned = use_returned
        
        @compile(suffix="test_move")
        def test_move() -> void:
            t = linear()
            t2 = move(t)
            consume(t2)
        cls.test_move = test_move
        
        @compile(suffix="test_undefined")
        def test_undefined() -> void:
            t: linear
        cls.test_undefined = test_undefined
        
        @compile(suffix="test_undef_assign")
        def test_undef_assign() -> void:
            t: linear
            t = linear()
            consume(t)
        cls.test_undef_assign = test_undef_assign
        
        @compile(suffix="test_undef_reassign")
        def test_undef_reassign() -> void:
            t: linear
            t = linear()
            consume(t)
            t = linear()
            consume(t)
        cls.test_undef_reassign = test_undef_reassign
    
    # =========================================================================
    # Valid case tests
    # =========================================================================
    
    def test_basic_linear_create_and_consume(self):
        """Test creating and consuming a linear token"""
        # Function compiled successfully in setUpClass
        self.assertIsNotNone(self.test_create_consume)
    
    def test_linear_parameter(self):
        """Test linear token as function parameter"""
        self.assertIsNotNone(self.use_token)
        self.assertIsNotNone(self.create_and_pass)
    
    def test_linear_return(self):
        """Test returning a linear token"""
        self.assertIsNotNone(self.create_token)
        self.assertIsNotNone(self.use_returned)
    
    def test_linear_move(self):
        """Test move() to transfer ownership"""
        self.assertIsNotNone(self.test_move)
    
    def test_linear_undefined(self):
        """Test declaring linear variable without initialization"""
        self.assertIsNotNone(self.test_undefined)
    
    def test_linear_undefined_assign(self):
        """Test undefined linear variable can be assigned"""
        self.assertIsNotNone(self.test_undef_assign)
    
    def test_linear_undefined_reassign(self):
        """Test consumed linear variable can be reassigned"""
        self.assertIsNotNone(self.test_undef_reassign)
    
    # =========================================================================
    # Error case tests
    # =========================================================================
    
    def test_error_not_consumed(self):
        """Test error when linear token is not consumed"""
        passed, msg = run_error_not_consumed()
        self.assertTrue(passed, msg)
    
    def test_error_double_consume(self):
        """Test error when consuming token twice"""
        passed, msg = run_error_double_consume()
        self.assertTrue(passed, msg)
    
    def test_error_use_after_move(self):
        """Test error when using token after move"""
        passed, msg = run_error_use_after_move()
        self.assertTrue(passed, msg)
    
    def test_error_assignment(self):
        """Test error when copying a linear token (should use move())"""
        passed, msg = run_error_assignment()
        self.assertTrue(passed, msg)
    
    def test_error_reassignment(self):
        """Test error when reassigning to unconsumed linear token"""
        passed, msg = run_error_reassignment()
        self.assertTrue(passed, msg)
    
    def test_error_undefined_consume(self):
        """Test error when consuming undefined linear token"""
        passed, msg = run_error_undefined_consume()
        self.assertTrue(passed, msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
