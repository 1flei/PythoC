#!/usr/bin/env python3
"""
Extended builtins: i128/u128, generic-width atomics, bit operations,
IEEE-754 float constants, incomplete-type pointers, quoted array element
types, and per-platform libc accessors.

Note: @compile wrappers are defined at module level because pythoc requires
all @compile definitions to precede the first native call from this module.
"""

import platform
import unittest

from pythoc import (
    i8, i32, i64, u8, u32, u64, i128, u128, f32, f64, bool, ptr, array, compile,
    void, static,
    atomic_load, atomic_store, atomic_fetch_add, atomic_cas, atomic_fence,
    atomic_fetch_and, atomic_fetch_or, atomic_exchange,
    inf, inff, nan, nanf,
    bswap, popcount, ctlz, cttz,
)
from pythoc.libc.errno import __error, __errno_location
from pythoc.libc.stdio import fputs, stderr


# ============================================================
# i128 / u128
# ============================================================

@compile
def i128_add(a: i128, b: i128) -> i128:
    """Crosses the FFI boundary in both directions (struct carrier)."""
    return a + b


@compile
def i128_sub_negative(a: i128, b: i128) -> i128:
    """Negative 128-bit results reassemble with sign extension."""
    return a - b


@compile
def u128_shr(a: u128, amount: i32) -> u128:
    return a >> amount


@compile
def i128_wide_math() -> i64:
    """Multiply past the 64-bit range, then recover the high word."""
    a: i128 = i128(4611686018427387904) * i128(4)  # 2**62 * 4 = 2**64
    b: i128 = a * i128(2)  # 2**65, beyond i64
    hi: i64 = i64(b >> 64)
    lo: i64 = i64(b & i128(18446744073709551615))
    return hi * i64(1000) + lo  # 2*1000 + 0 = 2000


@compile
def i128_div_mod() -> i64:
    a: i128 = i128(1) << 100
    b: i128 = a / i128(7)
    c: i128 = a % i128(7)
    # (2**100 // 7) * 7 + (2**100 % 7) == 2**100; check the pieces indirectly
    recombined: i128 = b * i128(7) + c
    if recombined == a:
        return i64(11)
    return i64(0)


@compile
def i128_cmp() -> i32:
    a: i128 = i128(1) << 100
    b: i128 = i64(9223372036854775807)  # i64 max, widened
    result: i32 = 0
    if a > b:
        result = result + 1
    if b < a:
        result = result + 2
    if a != b:
        result = result + 4
    neg: i128 = i128(0) - a
    if neg < i128(0):
        result = result + 8
    return result  # 15


# ============================================================
# Generic-width atomics
# ============================================================

@compile
def atomic_generic_ops() -> i64:
    """store/fetch_add/fetch_or/fetch_and/exchange/load/fence on i64."""
    val: i64 = 0
    p: ptr[i64] = ptr[i64](ptr[void](ptr(val)))
    atomic_store(p, i64(41))
    atomic_fetch_add(p, i64(1))      # 42
    atomic_fetch_or(p, i64(256))     # 42 | 256 = 298
    atomic_fetch_and(p, i64(300))    # 298 & 300 = 296
    old: i64 = atomic_exchange(p, i64(7))
    atomic_fence()
    return old * i64(1000) + atomic_load(p)  # 296*1000 + 7 = 296007


@compile
def atomic_generic_cas() -> i64:
    """CAS success and failure paths on i64."""
    val: i64 = 7
    expected: i64 = 7
    ok: i32 = atomic_cas(
        ptr[i64](ptr[void](ptr(val))),
        ptr[i64](ptr[void](ptr(expected))),
        i64(99),
    )
    fail_exp: i64 = 1
    ok2: i32 = atomic_cas(
        ptr[i64](ptr[void](ptr(val))),
        ptr[i64](ptr[void](ptr(fail_exp))),
        i64(5),
    )
    # ok=1, val=99; ok2=0, fail_exp=99
    return i64(ok) * i64(100000) + val * i64(1000) + i64(ok2) * i64(100) + fail_exp
    # 100000 + 99000 + 0 + 99 = 199099


@compile
def atomic_generic_i32() -> i32:
    """Width is derived from the pointer: ptr[i32] gives 32-bit atomics."""
    val: i32 = 100
    p: ptr[i32] = ptr[i32](ptr[void](ptr(val)))
    old: i32 = atomic_fetch_add(p, i32(23))
    return old + val  # 100 + 123 = 223


# ============================================================
# Bit operations
# ============================================================

@compile
def bitops_bswap32() -> u32:
    x: u32 = u32(305419896)  # 0x12345678
    return bswap(x)  # 0x78563412 = 2018915346


@compile
def bitops_bswap64() -> u64:
    x: u64 = u64(81985529216486895)  # 0x0123456789ABCDEF
    return bswap(x)  # 0xEFCDAB8967452301 = 17255913410373128961


@compile
def bitops_count() -> i32:
    a: u64 = u64(17361641481138401520)  # 0xF0F0F0F0F0F0F0F0
    pc: i32 = i32(popcount(a))  # 32
    lz: i32 = i32(ctlz(a))      # 0 (MSB set)
    tz: i32 = i32(cttz(a))      # 4
    zero: u32 = u32(0)
    lz0: i32 = i32(ctlz(zero))  # 32 (well-defined at zero)
    tz0: i32 = i32(cttz(zero))  # 32
    b: u8 = u8(1)
    lz8: i32 = i32(ctlz(b))     # 7
    return pc + lz + tz + lz0 + tz0 + lz8  # 32+0+4+32+32+7 = 107


# ============================================================
# IEEE-754 float constants
# ============================================================

@compile
def float_constants() -> i32:
    big: f64 = inf
    bigf: f32 = inff
    n: f64 = nan
    nf: f32 = nanf
    result: i32 = 0
    limit: f64 = 1e308
    if big > limit:
        result = result + 1
    neginf: f64 = f64(0) - big
    if neginf < -limit:
        result = result + 2
    if n != n:
        result = result + 4
    f32max: f32 = f32(3.0e38)
    if bigf > f32max:
        result = result + 8
    if nf != nf:
        result = result + 16
    nan_from_inf: f64 = big * f64(0)  # inf * 0 == nan
    if nan_from_inf != nan_from_inf:
        result = result + 32
    return result  # 63


# ============================================================
# Incomplete-type pointers and quoted element types
# ============================================================

@compile
def incomplete_ptr_roundtrip() -> i64:
    """ptr["Tag"] with Tag never defined is a pointer to an incomplete
    type (legal in C); it materializes as an opaque identified struct and
    can be cast through ptr[void] to a concrete pointer."""
    x: i64 = 777
    p: ptr["TestOnlyNeverDefinedOpaqueTag"] = ptr["TestOnlyNeverDefinedOpaqueTag"](ptr[void](ptr(x)))
    q: ptr[i64] = ptr[i64](ptr[void](p))
    return q[0]


@compile
class QuotedPoint:
    x: i32
    y: i32


@compile
class QuotedStaticHolder:
    """Class-level static member whose inner type is a quoted forward
    reference; the qualifier must resolve the string lazily (registry)
    instead of crashing attribute forwarding at materialization."""
    box: static["QuotedStaticBox"]


@compile
class QuotedStaticBox:
    v: i64


@compile
def quoted_static_member() -> i64:
    QuotedStaticHolder.box.v = 40
    return QuotedStaticHolder.box.v + i64(2)  # 42


@compile
def quoted_array_element() -> i32:
    """array["T", N] keeps the element type as a lazy name and resolves it
    through the forward-ref registry at compile time."""
    arr: array["QuotedPoint", 2]
    arr[0].x = 10
    arr[0].y = 20
    arr[1].x = 1
    arr[1].y = 2
    return arr[0].x + arr[0].y + arr[1].x + arr[1].y  # 33


# ============================================================
# libc per-platform accessors
# ============================================================

# Platform-specific symbols: defining these unconditionally would leave
# undefined symbols in the module's shared object.  ELF tolerates them at
# link time and lazy binding never resolves them when uncalled, but Windows
# DLLs require every symbol resolved at link time, so each accessor is only
# compiled on its own platform.
if platform.system() == 'Darwin':

    @compile
    def errno_slot_macos() -> i32:
        return __error()[0]


if platform.system() == 'Linux':

    @compile
    def errno_slot_glibc() -> i32:
        return __errno_location()[0]

    @compile
    def stderr_write() -> i32:
        """glibc exposes stderr as an extern FILE* data symbol."""
        return fputs("pythoc stderr smoke\n", stderr)


# ============================================================
# Test cases
# ============================================================

class TestInt128(unittest.TestCase):
    # The FFI carrier (a two-uint64 struct standing in for i128) follows the
    # SysV/AArch64 register-pair ABI.  Windows x64 passes/returns 16-byte
    # aggregates differently (hidden reference), which does not match
    # LLVM's i128 lowering there, so Python<->native marshalling of i128
    # values is not supported on Windows; in-native 128-bit arithmetic
    # works everywhere.
    _FFI_OK = platform.system() != 'Windows'

    @unittest.skipUnless(_FFI_OK, "i128 FFI marshalling is SysV/AArch64-only")
    def test_ffi_roundtrip(self):
        a = (1 << 70) + 5
        b = (1 << 70) + 7
        self.assertEqual(int(i128_add(a, b)), (1 << 71) + 12)

    @unittest.skipUnless(_FFI_OK, "i128 FFI marshalling is SysV/AArch64-only")
    def test_ffi_negative_result(self):
        self.assertEqual(int(i128_sub_negative(5, 1 << 70)), 5 - (1 << 70))

    @unittest.skipUnless(_FFI_OK, "i128 FFI marshalling is SysV/AArch64-only")
    def test_ffi_u128_shift(self):
        self.assertEqual(int(u128_shr(1 << 100, 3)), 1 << 97)

    def test_wide_math(self):
        self.assertEqual(i128_wide_math(), 2000)

    def test_div_mod(self):
        self.assertEqual(i128_div_mod(), 11)

    def test_comparisons(self):
        self.assertEqual(i128_cmp(), 15)


class TestGenericAtomics(unittest.TestCase):
    def test_ops(self):
        self.assertEqual(atomic_generic_ops(), 296007)

    def test_cas(self):
        self.assertEqual(atomic_generic_cas(), 199099)

    def test_i32_width(self):
        self.assertEqual(atomic_generic_i32(), 223)


class TestBitops(unittest.TestCase):
    def test_bswap32(self):
        self.assertEqual(int(bitops_bswap32()), 0x78563412)

    def test_bswap64(self):
        self.assertEqual(int(bitops_bswap64()), 0xEFCDAB8967452301)

    def test_counts(self):
        self.assertEqual(bitops_count(), 107)


class TestFloatConstants(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(float_constants(), 63)


class TestForwardRefForms(unittest.TestCase):
    def test_incomplete_ptr(self):
        self.assertEqual(incomplete_ptr_roundtrip(), 777)

    def test_quoted_array_element(self):
        self.assertEqual(quoted_array_element(), 33)

    def test_quoted_static_member(self):
        self.assertEqual(quoted_static_member(), 42)


class TestLibcAccessors(unittest.TestCase):
    @unittest.skipUnless(platform.system() == 'Darwin', 'macOS accessor')
    def test_errno_macos(self):
        self.assertEqual(errno_slot_macos(), 0)

    @unittest.skipUnless(platform.system() == 'Linux', 'glibc accessor')
    def test_errno_glibc(self):
        self.assertEqual(errno_slot_glibc(), 0)

    @unittest.skipUnless(platform.system() == 'Linux', 'glibc data symbol')
    def test_stderr_data_symbol(self):
        self.assertGreaterEqual(stderr_write(), 0)


if __name__ == "__main__":
    unittest.main()
