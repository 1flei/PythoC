"""The second pair of same-named handlers returns distinct values."""

from pythoc import compile, f64, i32


@compile(suffix='second')
def integer(value: i32) -> i32:
    return value + 100


@compile(suffix='second')
def floating(value: f64) -> i32:
    return i32(value) + 110
