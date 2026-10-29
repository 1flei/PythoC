"""The first pair of handlers with names also used in another module."""

from pythoc import compile, f64, i32


@compile(suffix='first')
def integer(value: i32) -> i32:
    return value


@compile(suffix='first')
def floating(value: f64) -> i32:
    return i32(value) + 10
