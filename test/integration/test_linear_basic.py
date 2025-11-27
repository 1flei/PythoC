"""
Basic tests for linear token functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc.decorators.compile import compile
from pythoc.builtin_entities import linear, consume, void
from pythoc.std.utility import move


def test_basic_linear_create_and_consume():
    """Test creating and consuming a linear token"""
    @compile
    def test() -> void:
        t = linear()
        consume(t)
    
    # Should compile successfully (don't call it)
    print("OK test_basic_linear_create_and_consume passed")


def test_linear_parameter():
    """Test linear token as function parameter"""
    @compile
    def use_token(t: linear) -> void:
        consume(t)
    
    @compile
    def create_and_pass() -> void:
        t = linear()
        use_token(t)
    
    print("OK test_linear_parameter passed")


def test_linear_return():
    """Test returning a linear token"""
    @compile
    def create_token() -> linear:
        t = linear()
        return t
    
    @compile
    def use_returned() -> void:
        t = create_token()
        consume(t)
    
    print("OK test_linear_return passed")


def test_linear_move():
    """Test move() to transfer ownership"""
    @compile
    def test_move() -> void:
        t = linear()
        t2 = move(t)
        consume(t2)
    
    print("OK test_linear_move passed")


def test_linear_undefined():
    """Test declaring linear variable without initialization"""
    @compile
    def test_undefined() -> void:
        t: linear
    
    print("OK test_linear_undefined passed")


def test_linear_undefined_assign():
    """Test undefined linear variable can be assigned"""
    @compile
    def test_undef_assign() -> void:
        t: linear
        t = linear()
        consume(t)
    
    print("OK test_linear_undefined_assign passed")


def test_linear_undefined_reassign():
    """Test consumed linear variable can be reassigned"""
    @compile
    def test_undef_reassign() -> void:
        t: linear
        t = linear()
        consume(t)
        t = linear()
        consume(t)
    
    print("OK test_linear_undefined_reassign passed")


def test_error_not_consumed():
    """Test error when linear token is not consumed"""
    try:
        @compile
        def bad_not_consumed() -> void:
            t = linear()
            # Missing consume(t)
        
        print("FAIL test_error_not_consumed failed - should have raised TypeError")
    except TypeError as e:
        if "not consumed" in str(e):
            print(f"OK test_error_not_consumed passed: {e}")
        else:
            print(f"FAIL test_error_not_consumed failed - wrong error: {e}")


def test_error_double_consume():
    """Test error when consuming token twice"""
    try:
        @compile
        def bad_double_consume() -> void:
            t = linear()
            consume(t)
            consume(t)  # ERROR
        
        print("FAIL test_error_double_consume failed - should have raised TypeError")
    except TypeError as e:
        if "already consumed" in str(e):
            print(f"OK test_error_double_consume passed: {e}")
        else:
            print(f"FAIL test_error_double_consume failed - wrong error: {e}")


def test_error_use_after_move():
    """Test error when using token after move"""
    try:
        @compile
        def bad_use_after_move() -> void:
            t = linear()
            t2 = move(t)
            consume(t)  # ERROR: t was consumed by move()
            consume(t2)
        
        print("FAIL test_error_use_after_move failed - should have raised TypeError")
    except TypeError as e:
        # move() is a @compile function, so it consumes its argument
        if "already consumed" in str(e):
            print(f"OK test_error_use_after_move passed: {e}")
        else:
            print(f"FAIL test_error_use_after_move failed - wrong error: {e}")


def test_error_assignment():
    """Test error when copying a linear token (should use move())"""
    try:
        @compile
        def bad_copy_linear() -> void:
            t = linear()
            t2 = t  # ERROR: cannot copy
            consume(t2)
        
        print("FAIL test_error_assignment failed - should have raised TypeError")
    except TypeError as e:
        if "Cannot assign linear token" in str(e) or "use move()" in str(e):
            print(f"OK test_error_assignment passed: {e}")
        else:
            print(f"FAIL test_error_assignment failed - wrong error: {e}")


def test_error_reassignment():
    """Test error when reassigning to unconsumed linear token"""
    try:
        @compile
        def bad_reassign_linear() -> void:
            t = linear()
            t = linear()  # ERROR: first token not consumed
            consume(t)
        
        print("FAIL test_error_reassignment failed - should have raised TypeError")
    except TypeError as e:
        if "not consumed" in str(e) or "Cannot reassign" in str(e):
            print(f"OK test_error_reassignment passed: {e}")
        else:
            print(f"FAIL test_error_reassignment failed - wrong error: {e}")


def test_error_undefined_consume():
    """Test error when consuming undefined linear token"""
    try:
        @compile(anonymous=True)
        def bad() -> void:
            t: linear
            consume(t)
        
        print("FAIL test_error_undefined_consume failed - should have raised TypeError")
    except TypeError as e:
        if "undefined" in str(e) or "Cannot consume" in str(e):
            print(f"OK test_error_undefined_consume passed: {e}")
        else:
            print(f"FAIL test_error_undefined_consume failed - wrong error: {e}")


if __name__ == "__main__":
    print("Running basic linear token tests...")
    print()
    
    test_basic_linear_create_and_consume()
    test_linear_parameter()
    test_linear_return()
    test_linear_move()
    test_linear_undefined()
    test_linear_undefined_assign()
    test_linear_undefined_reassign()
    print()
    
    print("Testing error cases...")
    test_error_not_consumed()
    test_error_double_consume()
    test_error_use_after_move()
    test_error_assignment()
    test_error_reassignment()
    test_error_undefined_consume()
    print()
    
    print("All basic linear token tests completed!")
