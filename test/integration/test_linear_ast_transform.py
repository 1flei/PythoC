"""
Test linear types with AST transformations (yield, inline, closure)

This tests the interaction between linear types and pythoc's AST transformation
mechanisms. All transformations that convert function call semantics to assignment
semantics need special handling for linear types.

See docs/yield-linear-type-analysis.md for detailed analysis.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pythoc import compile, i32, i8, void, struct
from pythoc.builtin_entities import linear, consume
from pythoc.std.utility import move
from pythoc.decorators.inline import inline
from pythoc.build.output_manager import flush_all_pending_outputs


# =============================================================================
# Test 1: Basic Yield with Linear Types
# =============================================================================

@compile
def yield_linear_basic() -> linear:
    """Yield function that yields a linear token"""
    prf = linear()
    yield prf

@compile
def test_yield_linear_basic() -> i32:
    """Test basic yield with linear type"""
    for prf in yield_linear_basic():
        consume(prf)
    return 0


# =============================================================================
# Test 2: Yield in Loop with Linear Types
# =============================================================================

@compile
def yield_linear_loop() -> linear:
    """Yield function that yields linear tokens in a loop"""
    i: i32 = 0
    while i < 3:
        prf = linear()
        yield prf
        i = i + 1

@compile
def test_yield_linear_loop() -> i32:
    """Test yield in loop with linear type"""
    count: i32 = 0
    for prf in yield_linear_loop():
        consume(prf)
        count = count + 1
    return count  # Should be 3


# =============================================================================
# Test 3: Yield Tuple with Linear Element
# =============================================================================

# TODO: Currently fails - tuple unpacking with linear types
# Uncomment when fixed
#
# @compile
# def yield_linear_tuple() -> struct[linear, i32]:
#     """Yield function that yields tuple with linear element"""
#     prf = linear()
#     yield prf, 42
#
# @compile
# def test_yield_linear_tuple() -> i32:
#     """Test yield tuple with linear element"""
#     for prf, val in yield_linear_tuple():
#         consume(prf)
#     return 0


# =============================================================================
# Test 4: Inline Function with Linear Argument
# =============================================================================

# TODO: Currently fails - inline arg transformation doesn't wrap in move()
# Uncomment when fixed
#
# @inline
# def consume_inline(t: linear) -> void:
#     """Inline function that consumes a linear token"""
#     consume(t)
#
# @compile
# def test_inline_linear_arg() -> i32:
#     """Test inline function with linear argument"""
#     t = linear()
#     consume_inline(t)
#     return 0


# =============================================================================
# Test 5: Inline Function Returning Linear
# =============================================================================

# TODO: Currently fails - inline return transformation doesn't wrap in move()
# Uncomment when fixed
#
# @inline
# def create_inline() -> linear:
#     """Inline function that creates and returns a linear token"""
#     t = linear()
#     return move(t)
#
# @compile
# def test_inline_linear_return() -> i32:
#     """Test inline function returning linear"""
#     t = create_inline()
#     consume(t)
#     return 0


# =============================================================================
# Test 6: Closure with Linear Argument
# =============================================================================

# TODO: Currently fails - closure arg transformation doesn't wrap in move()
# Uncomment when fixed
#
# @compile
# def test_closure_linear_arg() -> i32:
#     """Test closure with linear argument"""
#     def consume_closure(t: linear) -> void:
#         consume(t)
#     
#     t = linear()
#     consume_closure(t)
#     return 0


# =============================================================================
# Test 7: Closure Returning Linear
# =============================================================================

# TODO: Currently fails - closure return transformation doesn't wrap in move()
# Uncomment when fixed
#
# @compile
# def test_closure_linear_return() -> i32:
#     """Test closure returning linear"""
#     def create_closure() -> linear:
#         t = linear()
#         return move(t)
#     
#     t = create_closure()
#     consume(t)
#     return 0


# =============================================================================
# Test 8: Closure Capturing Linear from Outer Scope
# =============================================================================

# TODO: This is a complex case - capturing linear types may have different semantics
# Uncomment when fixed
#
# @compile
# def test_closure_capture_linear() -> i32:
#     """Test closure capturing linear from outer scope"""
#     t = linear()
#     
#     def consume_captured() -> void:
#         consume(t)  # Captures t from outer scope
#     
#     consume_captured()
#     return 0


# =============================================================================
# Test 9: Conditional Yield with Linear
# =============================================================================

# TODO: This requires proper handling of non-yield branches
# Uncomment when fixed
#
# @compile
# def yield_conditional(flag: i8) -> linear:
#     """Yield function with conditional yield"""
#     prf = linear()
#     if flag == 1:
#         yield prf
#     else:
#         consume(prf)  # Consume internally if not yielding
#
# @compile
# def test_conditional_yield_linear() -> i32:
#     """Test conditional yield with linear type"""
#     for prf in yield_conditional(1):
#         consume(prf)
#     return 0


# =============================================================================
# Working Tests (for reference - these should pass)
# =============================================================================

@compile
def test_basic_linear() -> i32:
    """Basic linear type test - should always pass"""
    t = linear()
    consume(t)
    return 0


@compile
def test_linear_move() -> i32:
    """Test move() with linear types - should always pass"""
    t = linear()
    t2 = move(t)
    consume(t2)
    return 0


@compile
def test_linear_if_else() -> i32:
    """Test linear in if-else - should always pass"""
    t = linear()
    flag: i8 = 1
    if flag == 1:
        consume(t)
    else:
        consume(t)
    return 0


class TestLinearASTTransform(unittest.TestCase):
    """Test suite for linear types with AST transformations"""
    
    @classmethod
    def setUpClass(cls):
        """Compile all functions before running tests"""
        flush_all_pending_outputs()
    
    # =========================================================================
    # Working tests
    # =========================================================================
    
    def test_basic_linear(self):
        """Basic linear type test"""
        result = test_basic_linear()
        self.assertEqual(result, 0)
    
    def test_linear_move(self):
        """Test move() with linear types"""
        result = test_linear_move()
        self.assertEqual(result, 0)
    
    def test_linear_if_else(self):
        """Test linear in if-else"""
        result = test_linear_if_else()
        self.assertEqual(result, 0)
    
    # =========================================================================
    # Yield tests - now working with move() fix
    # =========================================================================
    
    def test_yield_linear_basic(self):
        """Test basic yield with linear type"""
        result = test_yield_linear_basic()
        self.assertEqual(result, 0)
    
    def test_yield_linear_loop(self):
        """Test yield in loop with linear type"""
        result = test_yield_linear_loop()
        self.assertEqual(result, 3)
    
    # =========================================================================
    # TODO tests - uncomment when implementations are fixed
    # =========================================================================
    
    # def test_yield_linear_tuple(self):
    #     """Test yield tuple with linear element"""
    #     result = test_yield_linear_tuple()
    #     self.assertEqual(result, 0)
    
    # def test_inline_linear_arg(self):
    #     """Test inline function with linear argument"""
    #     result = test_inline_linear_arg()
    #     self.assertEqual(result, 0)
    
    # def test_inline_linear_return(self):
    #     """Test inline function returning linear"""
    #     result = test_inline_linear_return()
    #     self.assertEqual(result, 0)
    
    # def test_closure_linear_arg(self):
    #     """Test closure with linear argument"""
    #     result = test_closure_linear_arg()
    #     self.assertEqual(result, 0)
    
    # def test_closure_linear_return(self):
    #     """Test closure returning linear"""
    #     result = test_closure_linear_return()
    #     self.assertEqual(result, 0)
    
    # def test_closure_capture_linear(self):
    #     """Test closure capturing linear from outer scope"""
    #     result = test_closure_capture_linear()
    #     self.assertEqual(result, 0)
    
    # def test_conditional_yield_linear(self):
    #     """Test conditional yield with linear type"""
    #     result = test_conditional_yield_linear()
    #     self.assertEqual(result, 0)


if __name__ == '__main__':
    unittest.main()
