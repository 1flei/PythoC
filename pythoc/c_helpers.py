"""
C-style helper operations for PythoC.

Increment/decrement are exposed as four polymorphic dispatchers:
  post_inc, post_dec, pre_inc, pre_dec

Each dispatcher selects the appropriate compiled overload based on the
argument type at the call site.  Callers pass the address of the lvalue:

    post_inc(ptr(x))     # x++ for scalar x
    post_inc(ptr(p))     # p++ for pointer p

Overload implementations are generated from parametric templates so that a
single definition covers all scalar and pointer-to-scalar types.
Pointer-to-aggregate increment/decrement are handled by the pcc backend with
statement-level lowering, since their stride cannot be known in advance by
this generic module.
"""

from __future__ import annotations

from .decorators import compile
from .builtin_entities import ptr, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, param
from .std.poly import Poly


# All scalar types covered by the generic inc/dec helpers.
_SCALAR_TYPES = [
    i8, i16, i32, i64,
    u8, u16, u32, u64,
    f32, f64,
]


# ---------------------------------------------------------------------------
# Parametric templates
# ---------------------------------------------------------------------------

@compile
def _post_inc(T: param, p: ptr[T]) -> T:
    old: T = p[0]
    p[0] = old + 1
    return old


@compile
def _post_dec(T: param, p: ptr[T]) -> T:
    old: T = p[0]
    p[0] = old - 1
    return old


@compile
def _pre_inc(T: param, p: ptr[T]) -> T:
    p[0] = p[0] + 1
    return p[0]


@compile
def _pre_dec(T: param, p: ptr[T]) -> T:
    p[0] = p[0] - 1
    return p[0]


@compile
def _post_inc_ptr(T: param, pp: ptr[ptr[T]]) -> ptr[T]:
    old: ptr[T] = pp[0]
    pp[0] = old + 1
    return old


@compile
def _post_dec_ptr(T: param, pp: ptr[ptr[T]]) -> ptr[T]:
    old: ptr[T] = pp[0]
    pp[0] = old - 1
    return old


@compile
def _pre_inc_ptr(T: param, pp: ptr[ptr[T]]) -> ptr[T]:
    pp[0] = pp[0] + 1
    return pp[0]


@compile
def _pre_dec_ptr(T: param, pp: ptr[ptr[T]]) -> ptr[T]:
    pp[0] = pp[0] - 1
    return pp[0]


# ---------------------------------------------------------------------------
# Public Poly dispatchers
# ---------------------------------------------------------------------------
post_inc = Poly(
    *[_post_inc(t) for t in _SCALAR_TYPES],
    *[_post_inc_ptr(t) for t in _SCALAR_TYPES],
)

post_dec = Poly(
    *[_post_dec(t) for t in _SCALAR_TYPES],
    *[_post_dec_ptr(t) for t in _SCALAR_TYPES],
)

pre_inc = Poly(
    *[_pre_inc(t) for t in _SCALAR_TYPES],
    *[_pre_inc_ptr(t) for t in _SCALAR_TYPES],
)

pre_dec = Poly(
    *[_pre_dec(t) for t in _SCALAR_TYPES],
    *[_pre_dec_ptr(t) for t in _SCALAR_TYPES],
)
