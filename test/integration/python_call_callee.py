"""Callee compiled in a separate module so the caller IR keeps a native call."""

from pythoc import compile, i64


@compile
def inc(x: i64) -> i64:
    return x + 1
