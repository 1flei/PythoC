"""
C-style helper operations for PythoC.

Increment/decrement are exposed as four polymorphic dispatchers:
  post_inc, post_dec, pre_inc, pre_dec

Each dispatcher selects the appropriate compiled overload based on the
argument type at the call site.  Callers pass the address of the lvalue:

    post_inc(ptr(x))     # x++ for scalar x
    post_inc(ptr(p))     # p++ for pointer p

Overload definitions are explicit so PythoC's source inspection
(inspect.getsource) can locate them.  Pointer-to-aggregate increment/decrement
are handled by the pcc backend with statement-level lowering, since their
stride cannot be known in advance by this generic module.
"""

from .decorators import compile
from .builtin_entities import ptr, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64
from .std.poly import Poly


# ---------------------------------------------------------------------------
# i8
# ---------------------------------------------------------------------------
@compile
def post_inc_i8(p: ptr[i8]) -> i8:
    old: i8 = p[0]
    p[0] = old + 1
    return old


@compile
def post_dec_i8(p: ptr[i8]) -> i8:
    old: i8 = p[0]
    p[0] = old - 1
    return old


@compile
def pre_inc_i8(p: ptr[i8]) -> i8:
    p[0] = p[0] + 1
    return p[0]


@compile
def pre_dec_i8(p: ptr[i8]) -> i8:
    p[0] = p[0] - 1
    return p[0]


@compile
def post_inc_ptr_i8(pp: ptr[ptr[i8]]) -> ptr[i8]:
    old: ptr[i8] = pp[0]
    pp[0] = old + 1
    return old


@compile
def post_dec_ptr_i8(pp: ptr[ptr[i8]]) -> ptr[i8]:
    old: ptr[i8] = pp[0]
    pp[0] = old - 1
    return old


@compile
def pre_inc_ptr_i8(pp: ptr[ptr[i8]]) -> ptr[i8]:
    pp[0] = pp[0] + 1
    return pp[0]


@compile
def pre_dec_ptr_i8(pp: ptr[ptr[i8]]) -> ptr[i8]:
    pp[0] = pp[0] - 1
    return pp[0]


# ---------------------------------------------------------------------------
# i16
# ---------------------------------------------------------------------------
@compile
def post_inc_i16(p: ptr[i16]) -> i16:
    old: i16 = p[0]
    p[0] = old + 1
    return old


@compile
def post_dec_i16(p: ptr[i16]) -> i16:
    old: i16 = p[0]
    p[0] = old - 1
    return old


@compile
def pre_inc_i16(p: ptr[i16]) -> i16:
    p[0] = p[0] + 1
    return p[0]


@compile
def pre_dec_i16(p: ptr[i16]) -> i16:
    p[0] = p[0] - 1
    return p[0]


@compile
def post_inc_ptr_i16(pp: ptr[ptr[i16]]) -> ptr[i16]:
    old: ptr[i16] = pp[0]
    pp[0] = old + 1
    return old


@compile
def post_dec_ptr_i16(pp: ptr[ptr[i16]]) -> ptr[i16]:
    old: ptr[i16] = pp[0]
    pp[0] = old - 1
    return old


@compile
def pre_inc_ptr_i16(pp: ptr[ptr[i16]]) -> ptr[i16]:
    pp[0] = pp[0] + 1
    return pp[0]


@compile
def pre_dec_ptr_i16(pp: ptr[ptr[i16]]) -> ptr[i16]:
    pp[0] = pp[0] - 1
    return pp[0]


# ---------------------------------------------------------------------------
# i32
# ---------------------------------------------------------------------------
@compile
def post_inc_i32(p: ptr[i32]) -> i32:
    old: i32 = p[0]
    p[0] = old + 1
    return old


@compile
def post_dec_i32(p: ptr[i32]) -> i32:
    old: i32 = p[0]
    p[0] = old - 1
    return old


@compile
def pre_inc_i32(p: ptr[i32]) -> i32:
    p[0] = p[0] + 1
    return p[0]


@compile
def pre_dec_i32(p: ptr[i32]) -> i32:
    p[0] = p[0] - 1
    return p[0]


@compile
def post_inc_ptr_i32(pp: ptr[ptr[i32]]) -> ptr[i32]:
    old: ptr[i32] = pp[0]
    pp[0] = old + 1
    return old


@compile
def post_dec_ptr_i32(pp: ptr[ptr[i32]]) -> ptr[i32]:
    old: ptr[i32] = pp[0]
    pp[0] = old - 1
    return old


@compile
def pre_inc_ptr_i32(pp: ptr[ptr[i32]]) -> ptr[i32]:
    pp[0] = pp[0] + 1
    return pp[0]


@compile
def pre_dec_ptr_i32(pp: ptr[ptr[i32]]) -> ptr[i32]:
    pp[0] = pp[0] - 1
    return pp[0]


# ---------------------------------------------------------------------------
# i64
# ---------------------------------------------------------------------------
@compile
def post_inc_i64(p: ptr[i64]) -> i64:
    old: i64 = p[0]
    p[0] = old + 1
    return old


@compile
def post_dec_i64(p: ptr[i64]) -> i64:
    old: i64 = p[0]
    p[0] = old - 1
    return old


@compile
def pre_inc_i64(p: ptr[i64]) -> i64:
    p[0] = p[0] + 1
    return p[0]


@compile
def pre_dec_i64(p: ptr[i64]) -> i64:
    p[0] = p[0] - 1
    return p[0]


@compile
def post_inc_ptr_i64(pp: ptr[ptr[i64]]) -> ptr[i64]:
    old: ptr[i64] = pp[0]
    pp[0] = old + 1
    return old


@compile
def post_dec_ptr_i64(pp: ptr[ptr[i64]]) -> ptr[i64]:
    old: ptr[i64] = pp[0]
    pp[0] = old - 1
    return old


@compile
def pre_inc_ptr_i64(pp: ptr[ptr[i64]]) -> ptr[i64]:
    pp[0] = pp[0] + 1
    return pp[0]


@compile
def pre_dec_ptr_i64(pp: ptr[ptr[i64]]) -> ptr[i64]:
    pp[0] = pp[0] - 1
    return pp[0]


# ---------------------------------------------------------------------------
# u8
# ---------------------------------------------------------------------------
@compile
def post_inc_u8(p: ptr[u8]) -> u8:
    old: u8 = p[0]
    p[0] = old + 1
    return old


@compile
def post_dec_u8(p: ptr[u8]) -> u8:
    old: u8 = p[0]
    p[0] = old - 1
    return old


@compile
def pre_inc_u8(p: ptr[u8]) -> u8:
    p[0] = p[0] + 1
    return p[0]


@compile
def pre_dec_u8(p: ptr[u8]) -> u8:
    p[0] = p[0] - 1
    return p[0]


@compile
def post_inc_ptr_u8(pp: ptr[ptr[u8]]) -> ptr[u8]:
    old: ptr[u8] = pp[0]
    pp[0] = old + 1
    return old


@compile
def post_dec_ptr_u8(pp: ptr[ptr[u8]]) -> ptr[u8]:
    old: ptr[u8] = pp[0]
    pp[0] = old - 1
    return old


@compile
def pre_inc_ptr_u8(pp: ptr[ptr[u8]]) -> ptr[u8]:
    pp[0] = pp[0] + 1
    return pp[0]


@compile
def pre_dec_ptr_u8(pp: ptr[ptr[u8]]) -> ptr[u8]:
    pp[0] = pp[0] - 1
    return pp[0]


# ---------------------------------------------------------------------------
# u16
# ---------------------------------------------------------------------------
@compile
def post_inc_u16(p: ptr[u16]) -> u16:
    old: u16 = p[0]
    p[0] = old + 1
    return old


@compile
def post_dec_u16(p: ptr[u16]) -> u16:
    old: u16 = p[0]
    p[0] = old - 1
    return old


@compile
def pre_inc_u16(p: ptr[u16]) -> u16:
    p[0] = p[0] + 1
    return p[0]


@compile
def pre_dec_u16(p: ptr[u16]) -> u16:
    p[0] = p[0] - 1
    return p[0]


@compile
def post_inc_ptr_u16(pp: ptr[ptr[u16]]) -> ptr[u16]:
    old: ptr[u16] = pp[0]
    pp[0] = old + 1
    return old


@compile
def post_dec_ptr_u16(pp: ptr[ptr[u16]]) -> ptr[u16]:
    old: ptr[u16] = pp[0]
    pp[0] = old - 1
    return old


@compile
def pre_inc_ptr_u16(pp: ptr[ptr[u16]]) -> ptr[u16]:
    pp[0] = pp[0] + 1
    return pp[0]


@compile
def pre_dec_ptr_u16(pp: ptr[ptr[u16]]) -> ptr[u16]:
    pp[0] = pp[0] - 1
    return pp[0]


# ---------------------------------------------------------------------------
# u32
# ---------------------------------------------------------------------------
@compile
def post_inc_u32(p: ptr[u32]) -> u32:
    old: u32 = p[0]
    p[0] = old + 1
    return old


@compile
def post_dec_u32(p: ptr[u32]) -> u32:
    old: u32 = p[0]
    p[0] = old - 1
    return old


@compile
def pre_inc_u32(p: ptr[u32]) -> u32:
    p[0] = p[0] + 1
    return p[0]


@compile
def pre_dec_u32(p: ptr[u32]) -> u32:
    p[0] = p[0] - 1
    return p[0]


@compile
def post_inc_ptr_u32(pp: ptr[ptr[u32]]) -> ptr[u32]:
    old: ptr[u32] = pp[0]
    pp[0] = old + 1
    return old


@compile
def post_dec_ptr_u32(pp: ptr[ptr[u32]]) -> ptr[u32]:
    old: ptr[u32] = pp[0]
    pp[0] = old - 1
    return old


@compile
def pre_inc_ptr_u32(pp: ptr[ptr[u32]]) -> ptr[u32]:
    pp[0] = pp[0] + 1
    return pp[0]


@compile
def pre_dec_ptr_u32(pp: ptr[ptr[u32]]) -> ptr[u32]:
    pp[0] = pp[0] - 1
    return pp[0]


# ---------------------------------------------------------------------------
# u64
# ---------------------------------------------------------------------------
@compile
def post_inc_u64(p: ptr[u64]) -> u64:
    old: u64 = p[0]
    p[0] = old + 1
    return old


@compile
def post_dec_u64(p: ptr[u64]) -> u64:
    old: u64 = p[0]
    p[0] = old - 1
    return old


@compile
def pre_inc_u64(p: ptr[u64]) -> u64:
    p[0] = p[0] + 1
    return p[0]


@compile
def pre_dec_u64(p: ptr[u64]) -> u64:
    p[0] = p[0] - 1
    return p[0]


@compile
def post_inc_ptr_u64(pp: ptr[ptr[u64]]) -> ptr[u64]:
    old: ptr[u64] = pp[0]
    pp[0] = old + 1
    return old


@compile
def post_dec_ptr_u64(pp: ptr[ptr[u64]]) -> ptr[u64]:
    old: ptr[u64] = pp[0]
    pp[0] = old - 1
    return old


@compile
def pre_inc_ptr_u64(pp: ptr[ptr[u64]]) -> ptr[u64]:
    pp[0] = pp[0] + 1
    return pp[0]


@compile
def pre_dec_ptr_u64(pp: ptr[ptr[u64]]) -> ptr[u64]:
    pp[0] = pp[0] - 1
    return pp[0]


# ---------------------------------------------------------------------------
# f32
# ---------------------------------------------------------------------------
@compile
def post_inc_f32(p: ptr[f32]) -> f32:
    old: f32 = p[0]
    p[0] = old + 1
    return old


@compile
def post_dec_f32(p: ptr[f32]) -> f32:
    old: f32 = p[0]
    p[0] = old - 1
    return old


@compile
def pre_inc_f32(p: ptr[f32]) -> f32:
    p[0] = p[0] + 1
    return p[0]


@compile
def pre_dec_f32(p: ptr[f32]) -> f32:
    p[0] = p[0] - 1
    return p[0]


@compile
def post_inc_ptr_f32(pp: ptr[ptr[f32]]) -> ptr[f32]:
    old: ptr[f32] = pp[0]
    pp[0] = old + 1
    return old


@compile
def post_dec_ptr_f32(pp: ptr[ptr[f32]]) -> ptr[f32]:
    old: ptr[f32] = pp[0]
    pp[0] = old - 1
    return old


@compile
def pre_inc_ptr_f32(pp: ptr[ptr[f32]]) -> ptr[f32]:
    pp[0] = pp[0] + 1
    return pp[0]


@compile
def pre_dec_ptr_f32(pp: ptr[ptr[f32]]) -> ptr[f32]:
    pp[0] = pp[0] - 1
    return pp[0]


# ---------------------------------------------------------------------------
# f64
# ---------------------------------------------------------------------------
@compile
def post_inc_f64(p: ptr[f64]) -> f64:
    old: f64 = p[0]
    p[0] = old + 1
    return old


@compile
def post_dec_f64(p: ptr[f64]) -> f64:
    old: f64 = p[0]
    p[0] = old - 1
    return old


@compile
def pre_inc_f64(p: ptr[f64]) -> f64:
    p[0] = p[0] + 1
    return p[0]


@compile
def pre_dec_f64(p: ptr[f64]) -> f64:
    p[0] = p[0] - 1
    return p[0]


@compile
def post_inc_ptr_f64(pp: ptr[ptr[f64]]) -> ptr[f64]:
    old: ptr[f64] = pp[0]
    pp[0] = old + 1
    return old


@compile
def post_dec_ptr_f64(pp: ptr[ptr[f64]]) -> ptr[f64]:
    old: ptr[f64] = pp[0]
    pp[0] = old - 1
    return old


@compile
def pre_inc_ptr_f64(pp: ptr[ptr[f64]]) -> ptr[f64]:
    pp[0] = pp[0] + 1
    return pp[0]


@compile
def pre_dec_ptr_f64(pp: ptr[ptr[f64]]) -> ptr[f64]:
    pp[0] = pp[0] - 1
    return pp[0]


# ---------------------------------------------------------------------------
# Public Poly dispatchers
# ---------------------------------------------------------------------------
post_inc = Poly(
    post_inc_i8, post_inc_ptr_i8,
    post_inc_i16, post_inc_ptr_i16,
    post_inc_i32, post_inc_ptr_i32,
    post_inc_i64, post_inc_ptr_i64,
    post_inc_u8, post_inc_ptr_u8,
    post_inc_u16, post_inc_ptr_u16,
    post_inc_u32, post_inc_ptr_u32,
    post_inc_u64, post_inc_ptr_u64,
    post_inc_f32, post_inc_ptr_f32,
    post_inc_f64, post_inc_ptr_f64,
)

post_dec = Poly(
    post_dec_i8, post_dec_ptr_i8,
    post_dec_i16, post_dec_ptr_i16,
    post_dec_i32, post_dec_ptr_i32,
    post_dec_i64, post_dec_ptr_i64,
    post_dec_u8, post_dec_ptr_u8,
    post_dec_u16, post_dec_ptr_u16,
    post_dec_u32, post_dec_ptr_u32,
    post_dec_u64, post_dec_ptr_u64,
    post_dec_f32, post_dec_ptr_f32,
    post_dec_f64, post_dec_ptr_f64,
)

pre_inc = Poly(
    pre_inc_i8, pre_inc_ptr_i8,
    pre_inc_i16, pre_inc_ptr_i16,
    pre_inc_i32, pre_inc_ptr_i32,
    pre_inc_i64, pre_inc_ptr_i64,
    pre_inc_u8, pre_inc_ptr_u8,
    pre_inc_u16, pre_inc_ptr_u16,
    pre_inc_u32, pre_inc_ptr_u32,
    pre_inc_u64, pre_inc_ptr_u64,
    pre_inc_f32, pre_inc_ptr_f32,
    pre_inc_f64, pre_inc_ptr_f64,
)

pre_dec = Poly(
    pre_dec_i8, pre_dec_ptr_i8,
    pre_dec_i16, pre_dec_ptr_i16,
    pre_dec_i32, pre_dec_ptr_i32,
    pre_dec_i64, pre_dec_ptr_i64,
    pre_dec_u8, pre_dec_ptr_u8,
    pre_dec_u16, pre_dec_ptr_u16,
    pre_dec_u32, pre_dec_ptr_u32,
    pre_dec_u64, pre_dec_ptr_u64,
    pre_dec_f32, pre_dec_ptr_f32,
    pre_dec_f64, pre_dec_ptr_f64,
)
