"""Helper module with yield function that calls assume"""

from pythoc import compile, i32, assume


@compile
def generator_with_builtin(n: i32) -> i32:
    """Yield generator that uses assume builtin"""
    i: i32 = 0
    while i < n:
        val: i32 = assume(i, "test")
        yield val
        i = i + 1
