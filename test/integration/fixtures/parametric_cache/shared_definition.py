"""Two generic functions shared by independent compiler processes."""

from __future__ import annotations

from pythoc import compile, param


@compile
def identity(T: param, value: T) -> T:
    return value


@compile
def increment(T: param, value: T) -> T:
    return value + T(2)
