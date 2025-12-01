"""Helper module for cross-module yield test"""

from pythoc import compile, i32


@compile
def external_add(a: i32, b: i32) -> i32:
    """External helper function in different module"""
    return a + b
