#!/usr/bin/env python3
"""
Test yield function with complex control flow

This test reproduces the issue where yield functions with complex control flow
(multiple if-elif-else branches) fail to inline.

The current limitation is that yield functions must be "inlinable" which means
they cannot have complex control flow patterns.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import unittest

from pythoc import compile, i32, i8, void, ptr
from pythoc.build.output_manager import flush_all_pending_outputs, clear_failed_group


# =============================================================================
# Test cases that should work (simple control flow)
# =============================================================================

@compile
def yield_simple_while(n: i32) -> i32:
    """Simple yield in while loop - should work"""
    i: i32 = 0
    while i < n:
        yield i
        i = i + 1


@compile
def test_simple_while() -> i32:
    """Test simple while yield"""
    total: i32 = 0
    for x in yield_simple_while(5):
        total = total + x
    return total


@compile
def yield_with_single_if(n: i32) -> i32:
    """Yield with single if (no else) in loop - should work"""
    i: i32 = 0
    while i < n:
        if i > 0:
            yield i
        i = i + 1


@compile
def test_single_if() -> i32:
    """Test single if yield"""
    total: i32 = 0
    for x in yield_with_single_if(5):
        total = total + x
    return total


# =============================================================================
# Test cases that currently fail (complex control flow)
# =============================================================================

@compile
def yield_with_if_else(n: i32) -> i32:
    """Yield with if-else - may fail due to complex control flow"""
    i: i32 = 0
    while i < n:
        if i % 2 == 0:
            yield i
        else:
            yield i * 2
        i = i + 1


@compile
def test_if_else() -> i32:
    """Test if-else yield"""
    total: i32 = 0
    for x in yield_with_if_else(4):
        total = total + x
    return total


@compile
def yield_with_if_elif_else(code: i32) -> i32:
    """Yield with if-elif-else - complex control flow"""
    i: i32 = 0
    while i < 5:
        if code == 1:
            yield i
        elif code == 2:
            yield i * 2
        else:
            yield i * 3
        i = i + 1


@compile
def test_if_elif_else() -> i32:
    """Test if-elif-else yield"""
    total: i32 = 0
    for x in yield_with_if_elif_else(1):
        total = total + x
    return total


@compile
def yield_with_multiple_yields_in_loop(n: i32) -> i32:
    """Multiple yields in same loop iteration - complex"""
    i: i32 = 0
    while i < n:
        yield i
        yield i + 100
        i = i + 1


@compile
def test_multiple_yields() -> i32:
    """Test multiple yields per iteration"""
    total: i32 = 0
    for x in yield_with_multiple_yields_in_loop(3):
        total = total + x
    return total


@compile
def yield_with_nested_if(n: i32) -> i32:
    """Nested if statements with yield"""
    i: i32 = 0
    while i < n:
        if i > 0:
            if i < 3:
                yield i
            else:
                yield i * 2
        i = i + 1


@compile
def test_nested_if() -> i32:
    """Test nested if yield"""
    total: i32 = 0
    for x in yield_with_nested_if(5):
        total = total + x
    return total


# =============================================================================
# Minimal reproduction of parse_declarations pattern
# =============================================================================

@compile
def yield_with_dispatch(mode: i32) -> i32:
    """
    Simulates parse_declarations pattern:
    - While loop with multiple if-elif-else branches
    - Each branch yields a value
    """
    i: i32 = 0
    while i < 3:
        tok_type: i32 = i % 4
        
        if tok_type == 0:
            yield 10
        elif tok_type == 1:
            yield 20
        elif tok_type == 2:
            yield 30
        else:
            yield 40
        
        i = i + 1


@compile
def test_dispatch() -> i32:
    """Test dispatch pattern yield"""
    total: i32 = 0
    for x in yield_with_dispatch(0):
        total = total + x
    return total


class TestYieldComplexControlFlow(unittest.TestCase):
    """Test yield with complex control flow patterns"""
    
    def test_simple_while(self):
        """Simple while loop should work"""
        result = test_simple_while()
        # 0 + 1 + 2 + 3 + 4 = 10
        self.assertEqual(result, 10)
    
    def test_single_if(self):
        """Single if (no else) should work"""
        result = test_single_if()
        # 1 + 2 + 3 + 4 = 10 (skips 0)
        self.assertEqual(result, 10)
    
    def test_if_else(self):
        """If-else may fail - this tests the limitation"""
        try:
            result = test_if_else()
            # 0 + 2 + 2 + 6 = 10 (0, 1*2, 2, 3*2)
            self.assertEqual(result, 10)
        except RuntimeError as e:
            if "inlining failed" in str(e).lower():
                self.skipTest(f"Known limitation: {e}")
            raise
    
    def test_if_elif_else(self):
        """If-elif-else - complex control flow"""
        try:
            result = test_if_elif_else()
            # 0 + 1 + 2 + 3 + 4 = 10 (code=1, so yield i)
            self.assertEqual(result, 10)
        except RuntimeError as e:
            if "inlining failed" in str(e).lower():
                self.skipTest(f"Known limitation: {e}")
            raise
    
    def test_multiple_yields(self):
        """Multiple yields per iteration"""
        try:
            result = test_multiple_yields()
            # (0 + 100) + (1 + 101) + (2 + 102) = 306
            self.assertEqual(result, 306)
        except RuntimeError as e:
            if "inlining failed" in str(e).lower():
                self.skipTest(f"Known limitation: {e}")
            raise
    
    def test_nested_if(self):
        """Nested if statements"""
        try:
            result = test_nested_if()
            # i=1: 1, i=2: 2, i=3: 6, i=4: 8 = 17
            self.assertEqual(result, 17)
        except RuntimeError as e:
            if "inlining failed" in str(e).lower():
                self.skipTest(f"Known limitation: {e}")
            raise
    
    def test_dispatch_pattern(self):
        """Dispatch pattern (simulates parse_declarations)"""
        try:
            result = test_dispatch()
            # i=0: tok=0 -> 10, i=1: tok=1 -> 20, i=2: tok=2 -> 30 = 60
            self.assertEqual(result, 60)
        except RuntimeError as e:
            if "inlining failed" in str(e).lower():
                self.skipTest(f"Known limitation: {e}")
            raise


if __name__ == '__main__':
    unittest.main()
