"""Hash/equality implementations shared by hash containers."""

from __future__ import annotations

from pythoc import (
    compile, param, i8, i16, i32, i64, u8, u16, u32, u64, typeof, bool,
)
from pythoc.type_descriptor import same_type


# Constants from Abseil's MixingHashState.  kStaticRandomData[0] is used as a
# fixed seed; the multiplier kMul is chosen so that the 128-bit product spreads
# entropy across all output bits.
_HASH_K_MUL = u64(0x79d5f9e0de1e8cf5)
_HASH_SEED = u64(0x243f6a8885a308d3)


@compile
def _mix64(a: u64, b: u64) -> u64:
    """Return high64(a*b) ^ low64(a*b) without using a 128-bit type."""
    a_lo: u64 = a & u64(0xFFFFFFFF)
    a_hi: u64 = a >> 32
    b_lo: u64 = b & u64(0xFFFFFFFF)
    b_hi: u64 = b >> 32

    lo: u64 = a_lo * b_lo
    mid1: u64 = a_lo * b_hi
    mid2: u64 = a_hi * b_lo
    hi: u64 = a_hi * b_hi

    mid: u64 = mid1 + mid2
    mid_carry: u64 = u64(0)
    if mid < mid1:
        mid_carry = u64(1)

    mid_lo_shifted: u64 = (mid & u64(0xFFFFFFFF)) << 32
    low64: u64 = lo + mid_lo_shifted
    carry: u64 = (mid >> 32) + (mid_carry << 32)
    if low64 < lo:
        carry += 1
    high64: u64 = hi + carry

    return high64 ^ low64


@compile
def _scalar_hash(T: param, key: T) -> u64:
    """Abseil-style hash for scalar integer keys.

    Mirrors ``absl::Hash``'s IntegralFastPath: cast to unsigned 64-bit and mix
    with a seed and the fixed multiplier ``kMul``.
    """
    v: u64 = u64(key)
    return _mix64(_HASH_SEED ^ v, _HASH_K_MUL)


@compile
def _identity_eq(T: param, a: T, b: T) -> bool:
    return a == b


_SCALAR_INTEGER_TYPES = [i8, i16, i32, i64, u8, u16, u32, u64]


def key_functions(key_type, hash_fn=None, eq_fn=None):
    """Resolve one fixed pair; equivalent keys must have identical hashes."""
    if hash_fn is None and eq_fn is None:
        if key_type not in _SCALAR_INTEGER_TYPES:
            raise TypeError(f'no default hash for key type {key_type}')
        return _scalar_hash(key_type), _identity_eq(key_type)
    if hash_fn is None or eq_fn is None:
        raise TypeError('Custom hash_fn and eq_fn must be supplied together')
    for function, parameters, result in (
            (hash_fn, (key_type,), u64), (eq_fn, (key_type, key_type), bool)):
        signature = typeof(function)
        actual = getattr(signature, 'param_types', ())
        if (len(actual) != len(parameters)
                or not all(same_type(a, b) for a, b in zip(actual, parameters))
                or not same_type(getattr(signature, 'return_type', None), result)):
            raise TypeError('Hash/equality functions have incompatible signatures')
    return hash_fn, eq_fn
