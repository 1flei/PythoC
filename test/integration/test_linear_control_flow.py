"""
Tests for linear token control flow (if/else, loops)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc.decorators.compile import compile
from pythoc.builtin_entities import linear, consume, void, i32
from pythoc.std.utility import move
from pythoc.logger import set_raise_on_error

# Enable exception raising for tests that expect to catch exceptions
set_raise_on_error(True)


def test_if_else_both_consume():
    """Test if/else where both branches consume the token"""
    @compile
    def test_ifelse(cond: i32) -> void:
        t = linear()
        if cond:
            consume(t)
        else:
            consume(t)
    
    print("OK test_if_else_both_consume passed")


def test_if_without_else_error():
    """Test that if without else cannot consume token"""
    try:
        @compile
        def bad_no_else(cond: i32) -> void:
            t = linear()
            if cond:
                consume(t)
            # ERROR: else branch doesn't consume
        
        print("FAIL test_if_without_else_error failed - should have raised RuntimeError")
    except RuntimeError as e:
        if "modified in if without else" in str(e).lower() or "consistently" in str(e).lower():
            print(f"OK test_if_without_else_error passed: {e}")
        else:
            print(f"FAIL test_if_without_else_error failed - wrong error: {e}")


def test_if_else_inconsistent_error():
    """Test error when if/else branches handle token inconsistently"""
    try:
        @compile
        def bad_inconsistent(cond: i32) -> void:
            t = linear()
            if cond:
                consume(t)
            else:
                pass  # ERROR: doesn't consume
        
        print("FAIL test_if_else_inconsistent_error failed - should have raised RuntimeError")
    except RuntimeError as e:
        if "consistently" in str(e).lower():
            print(f"OK test_if_else_inconsistent_error passed: {e}")
        else:
            print(f"FAIL test_if_else_inconsistent_error failed - wrong error: {e}")


def test_loop_consume_internal_token():
    """Test that loop can consume token created inside loop"""
    @compile
    def test_loop_internal() -> void:
        i: i32 = 0
        while i < 3:
            t = linear()
            consume(t)
            i = i + 1
    
    print("OK test_loop_consume_internal_token passed")


def test_loop_cannot_consume_external_token():
    """Test that loop cannot consume token created outside loop"""
    try:
        @compile
        def bad_loop_external() -> void:
            t = linear()
            i: i32 = 0
            while i < 3:
                consume(t)  # ERROR: external token
                i = i + 1
        
        print("FAIL test_loop_cannot_consume_external_token failed - should have raised RuntimeError")
    except RuntimeError as e:
        if "external" in str(e).lower() or "scope" in str(e).lower():
            print(f"OK test_loop_cannot_consume_external_token passed: {e}")
        else:
            print(f"FAIL test_loop_cannot_consume_external_token failed - wrong error: {e}")


def test_for_loop_consume_internal():
    """Test for loop consuming internal token"""
    @compile
    def test_for_internal() -> void:
        for i in range(3):
            t = linear()
            consume(t)
    
    print("OK test_for_loop_consume_internal passed")


def test_move_identity():
    """Test that move() is just an identity function"""
    @compile
    def test_move_func() -> void:
        t = linear()
        t2 = move(t)  # move() transfers ownership via function call
        consume(t2)
    
    print("OK test_move_identity passed")


if __name__ == "__main__":
    print("Running linear token control flow tests...")
    print()
    
    print("Testing loops...")
    test_loop_consume_internal_token()
    test_for_loop_consume_internal()
    print()
    
    print("Testing if/else...")
    test_if_else_both_consume()
    print()
    
    print("Testing move()...")
    test_move_identity()
    print()
    
    # Run error tests separately to avoid polluting the module
    print("Testing error cases...")
    test_loop_cannot_consume_external_token()
    test_if_without_else_error()
    test_if_else_inconsistent_error()
    print()
    
    print("All linear token control flow tests completed!")
