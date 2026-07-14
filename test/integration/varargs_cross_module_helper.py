"""Helper module for cross-module bare *args regression test."""

from pythoc import compile, i32, i8, ptr


@compile
def wrapper(fmt: ptr[i8], *args) -> i32:
    # Body is intentionally trivial: the bug manifests during call lowering
    # ("Function expects 1 arguments, got 2") before the body ever runs.
    return 0
