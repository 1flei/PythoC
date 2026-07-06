from __future__ import annotations

"""
C-style helper operations for PythoC.

Increment/decrement are exposed as four parametric functions:
  post_inc, post_dec, pre_inc, pre_dec

Each takes a compile-time type parameter ``T`` and a pointer to the lvalue::

    post_inc(i32, ptr(x))       # x++ for scalar x
    post_inc(ptr[i32], ptr(p))  # p++ for pointer p

Pointer-to-aggregate increment/decrement are handled by the pcc backend with
statement-level lowering, since the parametric body ``old + 1`` is not valid
for aggregate types.
"""

from pythoc import compile, param, ptr


@compile
def post_inc(T: param, p: ptr[T]) -> T:
    old: T = p[0]
    p[0] = old + 1
    return old


@compile
def post_dec(T: param, p: ptr[T]) -> T:
    old: T = p[0]
    p[0] = old - 1
    return old


@compile
def pre_inc(T: param, p: ptr[T]) -> T:
    p[0] = p[0] + 1
    return p[0]


@compile
def pre_dec(T: param, p: ptr[T]) -> T:
    p[0] = p[0] - 1
    return p[0]
