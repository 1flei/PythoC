"""
Yield-based seq iterator implementation

This module provides yield-based seq() functions that can be inlined
by the yield inlining optimization, eliminating vtable overhead.
"""

from ..builtin_entities.types import i32
from ..decorators.compile import compile



@compile
def counter(end: i32) -> i32:
    """seq(end) - iterate from 0 to end-1 using yield
    
    This yield-based implementation can be inlined by the yield inlining
    optimization, generating efficient loop code without vtable overhead.
    """
    i: i32 = 0
    while i < end:
        yield i
        i = i + 1


@compile
def counter_range(start: i32, end: i32) -> i32:
    """seq(start, end) - iterate from start to end-1 using yield
    
    This yield-based implementation can be inlined by the yield inlining
    optimization, generating efficient loop code without vtable overhead.
    """
    i: i32 = start
    while i < end:
        yield i
        i = i + 1


@compile
def counter_range_step(start: i32, end: i32, step: i32) -> i32:
    """seq(start, end) - iterate from start to end-1 using yield
    
    This yield-based implementation can be inlined by the yield inlining
    optimization, generating efficient loop code without vtable overhead.
    """
    i: i32 = start
    while i < end:
        yield i
        i += step
