# -*- coding: utf-8 -*-
"""
Struct-by-value ABI matrix through cimport.

A header + compiled C source pair where the C functions take and return
structs BY VALUE, exercised from @compile code across the interesting ABI
classes: sub-register, single-register, two-register, and memory (sret /
byval) paths, plus enums as parameters and return values.

Covers:
- 1-field i32 (4-byte) struct by value
- 2-field i32/i32 (8-byte), f64/f64 (16-byte SSE), mixed i32/f64 (16-byte
  INT+SSE) structs by value
- 24-byte struct (sret return / byval parameter)
- struct containing an array, nested struct by value
- struct with i8 fields (offsetof/size correctness, no padding)
- struct return values feeding further @compile arithmetic
- in-out struct transformation (swap) by value
- enum parameters/returns, including negative enum values round-tripping

Note: @compile wrappers are defined at module level because pythoc requires
all @compile definitions to precede the first native call from this module.
"""
from __future__ import annotations

import os
import unittest

from pythoc import compile, i8, i32, i64, f64, offsetof


def _clang_backend_available() -> bool:
    try:
        from pythoc.cimport_clang import is_clang_backend_available
    except Exception:
        return False
    return is_clang_backend_available()


def _cc_available() -> bool:
    try:
        from pythoc.utils.cc_utils import find_available_cc
        find_available_cc()
    except RuntimeError:
        return False
    return True


_BACKEND_AVAILABLE = _clang_backend_available() and _cc_available()

# =============================================================================
# Module-level fixtures: cimport + @compile wrappers
# =============================================================================

_fixture_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'build', 'test', 'cimport_structval'))
os.makedirs(_fixture_dir, exist_ok=True)


def _write_fixture(name: str, content: str) -> str:
    path = os.path.join(_fixture_dir, name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path


if _BACKEND_AVAILABLE:
    from pythoc.cimport import cimport

    _header = _write_fixture('sval.h', '''
struct One { int a; };
struct TwoI { int a; int b; };
struct TwoL { long long a; long long b; };
struct TwoD { double a; double b; };
struct Mix { int a; double b; };
struct Big { long long a, b, c; };
struct WithArr { int v[4]; };
struct SvInner { int x; int y; };
struct SvOuter { struct SvInner in; int tag; };
struct Chars { char a; char b; char c; };

enum Sign { NEG = -7, ZERO = 0, POS = 9 };

struct One mk_one(int a);
struct TwoI mk_twoi(int a, int b);
struct TwoL mk_twol(long long a, long long b);
struct TwoD mk_twod(double a, double b);
struct Mix mk_mix(int a, double b);
struct Big mk_big(long long a, long long b, long long c);
struct WithArr mk_arr(int a, int b, int c, int d);
struct SvOuter mk_outer(int x, int y, int tag);
struct Chars mk_chars(char a, char b, char c);

int sum_one(struct One s);
int sum_twoi(struct TwoI s);
long long sum_twol(struct TwoL s);
double sum_twod(struct TwoD s);
double sum_mix(struct Mix s);
long long sum_big(struct Big s);
int sum_arr(struct WithArr s);
int sum_outer(struct SvOuter s);
int sum_chars(struct Chars s);
struct TwoI swap_twoi(struct TwoI s);
struct Big scale_big(struct Big s, long long k);
enum Sign flip_sign(enum Sign s);
int sign_value(enum Sign s);
''')
    _source = _write_fixture('sval.c', '''
#include "sval.h"

struct One mk_one(int a) { struct One s = {a}; return s; }
struct TwoI mk_twoi(int a, int b) { struct TwoI s = {a, b}; return s; }
struct TwoL mk_twol(long long a, long long b) { struct TwoL s = {a, b}; return s; }
struct TwoD mk_twod(double a, double b) { struct TwoD s = {a, b}; return s; }
struct Mix mk_mix(int a, double b) { struct Mix s = {a, b}; return s; }
struct Big mk_big(long long a, long long b, long long c) {
    struct Big s = {a, b, c};
    return s;
}
struct WithArr mk_arr(int a, int b, int c, int d) {
    struct WithArr s = {{a, b, c, d}};
    return s;
}
struct SvOuter mk_outer(int x, int y, int tag) {
    struct SvOuter s = {{x, y}, tag};
    return s;
}
struct Chars mk_chars(char a, char b, char c) {
    struct Chars s = {a, b, c};
    return s;
}

int sum_one(struct One s) { return s.a; }
int sum_twoi(struct TwoI s) { return s.a + s.b; }
long long sum_twol(struct TwoL s) { return s.a + s.b; }
double sum_twod(struct TwoD s) { return s.a + s.b; }
double sum_mix(struct Mix s) { return s.a + s.b; }
long long sum_big(struct Big s) { return s.a + s.b + s.c; }
int sum_arr(struct WithArr s) { return s.v[0] + s.v[1] + s.v[2] + s.v[3]; }
int sum_outer(struct SvOuter s) { return s.in.x + s.in.y + s.tag; }
int sum_chars(struct Chars s) { return s.a + s.b + s.c; }
struct TwoI swap_twoi(struct TwoI s) { struct TwoI r = {s.b, s.a}; return r; }
struct Big scale_big(struct Big s, long long k) {
    struct Big r = {s.a * k, s.b * k, s.c * k};
    return r;
}
enum Sign flip_sign(enum Sign s) { return s == POS ? NEG : POS; }
int sign_value(enum Sign s) { return (int)s; }
''')
    _mod = cimport(_header, sources=[_source],
                   compile_sources=True, include_dirs=[_fixture_dir])
    One = _mod.One
    TwoI = _mod.TwoI
    TwoL = _mod.TwoL
    TwoD = _mod.TwoD
    Mix = _mod.Mix
    Big = _mod.Big
    WithArr = _mod.WithArr
    SvOuter = _mod.SvOuter
    Chars = _mod.Chars
    mk_one = _mod.mk_one
    mk_twoi = _mod.mk_twoi
    mk_twol = _mod.mk_twol
    mk_twod = _mod.mk_twod
    mk_mix = _mod.mk_mix
    mk_big = _mod.mk_big
    mk_arr = _mod.mk_arr
    mk_outer = _mod.mk_outer
    mk_chars = _mod.mk_chars
    sum_one = _mod.sum_one
    sum_twoi = _mod.sum_twoi
    sum_twol = _mod.sum_twol
    sum_twod = _mod.sum_twod
    sum_mix = _mod.sum_mix
    sum_big = _mod.sum_big
    sum_arr = _mod.sum_arr
    sum_outer = _mod.sum_outer
    sum_chars = _mod.sum_chars
    swap_twoi = _mod.swap_twoi
    scale_big = _mod.scale_big
    flip_sign = _mod.flip_sign
    sign_value = _mod.sign_value
    POS = _mod.POS
    NEG = _mod.NEG
    Sign = _mod.Sign

    # --- 4-byte struct ---
    @compile
    def one_roundtrip(a: i32) -> i32:
        s: One = mk_one(a)
        return sum_one(s)

    @compile
    def one_fields_in_arithmetic(a: i32) -> i32:
        s: One = mk_one(a)
        return s.a * 2 + sum_one(s)

    # --- 8-byte struct ---
    @compile
    def twoi_roundtrip(a: i32, b: i32) -> i32:
        s: TwoI = mk_twoi(a, b)
        return sum_twoi(s)

    @compile
    def twoi_swap(a: i32, b: i32) -> i32:
        s: TwoI = mk_twoi(a, b)
        r: TwoI = swap_twoi(s)
        # sum must be invariant; fields must be swapped
        return sum_twoi(r) * 10000 + r.a * 100 + r.b

    # --- 16-byte structs: INTEGER/INTEGER, SSE/SSE, mixed ---
    @compile
    def twol_roundtrip(a: i64, b: i64) -> i64:
        s: TwoL = mk_twol(a, b)
        return sum_twol(s)

    @compile
    def twod_roundtrip(a: f64, b: f64) -> f64:
        s: TwoD = mk_twod(a, b)
        return sum_twod(s)

    @compile
    def mix_roundtrip(a: i32, b: f64) -> f64:
        s: Mix = mk_mix(a, b)
        return sum_mix(s)

    @compile
    def mix_fields_in_arithmetic() -> i32:
        s: Mix = mk_mix(7, 0.5)
        scaled: f64 = sum_mix(s) * 2.0
        if scaled != 15.0:
            return -1
        return s.a

    # --- 24-byte struct: sret/byval path ---
    @compile
    def big_roundtrip(a: i64, b: i64, c: i64) -> i64:
        s: Big = mk_big(a, b, c)
        return sum_big(s)

    @compile
    def big_scale(a: i64, k: i64) -> i64:
        s: Big = mk_big(a, a + 1, a + 2)
        r: Big = scale_big(s, k)
        return sum_big(r) + r.a + r.b + r.c

    # --- struct containing an array ---
    @compile
    def arr_roundtrip() -> i32:
        s: WithArr = mk_arr(1, 2, 3, 4)
        return sum_arr(s) * 10 + s.v[3]

    # --- nested struct by value ---
    @compile
    def outer_roundtrip(x: i32, y: i32, tag: i32) -> i32:
        s: SvOuter = mk_outer(x, y, tag)
        return sum_outer(s)

    @compile
    def outer_fields() -> i32:
        s: SvOuter = mk_outer(5, 6, 7)
        return s.in_.x * 100 + s.in_.y * 10 + s.tag

    # --- i8-field struct, no padding ---
    @compile
    def chars_roundtrip() -> i32:
        s: Chars = mk_chars(10, 20, 12)
        return sum_chars(s)

    @compile
    def chars_offsets() -> i32:
        return (offsetof(Chars, "a") + offsetof(Chars, "b") * 16
                + offsetof(Chars, "c") * 256)

    @compile
    def mix_offsets() -> i32:
        return offsetof(Mix, "a") + offsetof(Mix, "b") * 16

    # --- enum params/returns ---
    @compile
    def enum_flip_pos() -> i32:
        return sign_value(flip_sign(POS))

    @compile
    def enum_flip_neg() -> i32:
        return sign_value(flip_sign(NEG))

    @compile
    def enum_double_flip() -> i32:
        # enum-typed return values feed straight back into enum params
        return sign_value(flip_sign(flip_sign(POS)))

    @compile
    def enum_construct_from_int(v: i32) -> i32:
        s: Sign = Sign(v)
        return sign_value(s)


@unittest.skipUnless(_BACKEND_AVAILABLE, "clang backend or cc not available")
class TestCimportStructValSmall(unittest.TestCase):
    """Sub-register and single-register structs by value."""

    def test_one_field_i32(self):
        self.assertEqual(one_roundtrip(42), 42)
        self.assertEqual(one_roundtrip(-7), -7)

    def test_return_value_feeds_arithmetic(self):
        self.assertEqual(one_fields_in_arithmetic(10), 30)

    def test_two_i32(self):
        self.assertEqual(twoi_roundtrip(19, 23), 42)
        self.assertEqual(twoi_roundtrip(-5, 5), 0)

    def test_two_i32_swap(self):
        # sum 10 -> 100000 + swapped fields a=7 b=3 -> 703
        self.assertEqual(twoi_swap(3, 7), 100703)

    def test_i8_fields_no_padding(self):
        self.assertEqual(chars_roundtrip(), 42)
        self.assertEqual(Chars.get_size_bytes(), 3)

    def test_i8_field_offsets(self):
        self.assertEqual(chars_offsets(), 0 + 1 * 16 + 2 * 256)


@unittest.skipUnless(_BACKEND_AVAILABLE, "clang backend or cc not available")
class TestCimportStructValMedium(unittest.TestCase):
    """16-byte structs: INTEGER/INTEGER, SSE/SSE, and mixed classes."""

    def test_two_i64(self):
        self.assertEqual(twol_roundtrip(4000000000, 2000000000), 6000000000)

    def test_two_f64(self):
        self.assertEqual(twod_roundtrip(1.5, 2.5), 4.0)
        self.assertEqual(twod_roundtrip(-1.25, 1.25), 0.0)

    def test_mixed_i32_f64(self):
        self.assertEqual(mix_roundtrip(2, 3.5), 5.5)

    def test_mixed_offsets(self):
        # a at 0, b at 8 (i32 padded to f64 alignment)
        self.assertEqual(mix_offsets(), 0 + 8 * 16)
        self.assertEqual(Mix.get_size_bytes(), 16)

    def test_mixed_return_in_arithmetic(self):
        self.assertEqual(mix_fields_in_arithmetic(), 7)


@unittest.skipUnless(_BACKEND_AVAILABLE, "clang backend or cc not available")
class TestCimportStructValLarge(unittest.TestCase):
    """Memory-class structs (>16 bytes): sret return, byval parameter."""

    def test_big_roundtrip(self):
        self.assertEqual(big_roundtrip(100, 200, 300), 600)
        self.assertEqual(big_roundtrip(-1, 0, 1), 0)

    def test_big_in_out_transform(self):
        # s = {a, a+1, a+2}; scaled by k; 2 * k * (3a+3)
        self.assertEqual(big_scale(10, 3), 2 * 3 * 33)

    def test_struct_with_array(self):
        # sum 10 * 10 + v[3] = 104
        self.assertEqual(arr_roundtrip(), 104)
        self.assertEqual(WithArr.get_size_bytes(), 16)

    def test_nested_struct(self):
        self.assertEqual(outer_roundtrip(10, 20, 12), 42)
        self.assertEqual(outer_fields(), 567)


@unittest.skipUnless(_BACKEND_AVAILABLE, "clang backend or cc not available")
class TestCimportStructValEnum(unittest.TestCase):
    """Enum types as call parameters and return values."""

    def test_negative_enum_roundtrip(self):
        self.assertEqual(enum_flip_pos(), -7)

    def test_flip_neg_gives_pos(self):
        self.assertEqual(enum_flip_neg(), 9)

    def test_enum_return_feeds_enum_param(self):
        self.assertEqual(enum_double_flip(), 9)

    def test_enum_constructed_from_runtime_int(self):
        self.assertEqual(enum_construct_from_int(-7), -7)
        self.assertEqual(enum_construct_from_int(9), 9)


if __name__ == "__main__":
    unittest.main()
