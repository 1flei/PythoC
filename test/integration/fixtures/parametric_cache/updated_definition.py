"""An edited physical template must invalidate its cached specializations."""

from __future__ import annotations

from pythoc import compile, param


@compile
def identity(T: param, value: T) -> T:
    return value + T(1)


@compile
def increment(T: param, value: T) -> T:
    return value + T(2)
