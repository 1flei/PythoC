# -*- coding: utf-8 -*-
"""
End-to-end tests for variadic calls through cimport.

printf itself is unsafe to capture in tests, so output goes through
snprintf/sprintf into stack buffers with content assertions.  A compiled
C source provides custom variadic functions (sum of N ints, a double
accumulator, a sentinel counter) called with varying argument counts.

Covers:
- int / double / string / mixed varargs through snprintf
- C default argument promotions at the varargs boundary:
  f32 -> f64, i8/i16 -> i32
- i64/u64 varargs passed unpromoted
- sprintf into a buffer
- variadic calls with zero extra arguments
- custom variadic C functions (va_arg consumers) from a compiled source,
  called with 0..5 extra arguments

Note: @compile wrappers are defined at module level because pythoc requires
all @compile definitions to precede the first native call from this module.
"""
from __future__ import annotations

import os
import sys
import unittest

from pythoc import (
    compile, i8, i16, i32, i64, u64, f32, f64, ptr, array,
)


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
    os.path.dirname(__file__), '..', '..', 'build', 'test', 'cimport_variadic'))
os.makedirs(_fixture_dir, exist_ok=True)


def _write_fixture(name: str, content: str) -> str:
    path = os.path.join(_fixture_dir, name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path


if _BACKEND_AVAILABLE:
    from pythoc.cimport import cimport

    # includes=True: system stdio.h/string.h delegate to private
    # sub-headers on some SDKs (macOS _stdio.h/_string.h).
    _stdio = cimport('stdio.h', lib='c', includes=True)
    _string = cimport('string.h', lib='c', includes=True)
    snprintf = _stdio.snprintf
    sprintf = _stdio.sprintf
    strcmp = _string.strcmp

    # --- Custom variadic C functions (va_arg consumers) ---
    _va_header = _write_fixture('va_lib.h', '''
int va_sum_ints(int n, ...);
double va_sum_doubles(int n, ...);
int va_counted(int first, ...);
double va_weighted(double w, int n, ...);
''')
    _va_source = _write_fixture('va_lib.c', '''
#include <stdarg.h>
#include "va_lib.h"

int va_sum_ints(int n, ...) {
    va_list ap;
    va_start(ap, n);
    int total = 0;
    for (int i = 0; i < n; i++)
        total += va_arg(ap, int);
    va_end(ap);
    return total;
}

double va_sum_doubles(int n, ...) {
    va_list ap;
    va_start(ap, n);
    double total = 0.0;
    for (int i = 0; i < n; i++)
        total += va_arg(ap, double);
    va_end(ap);
    return total;
}

int va_counted(int first, ...) {
    va_list ap;
    va_start(ap, first);
    int v = first;
    int count = 0;
    while (v >= 0) {
        count++;
        v = va_arg(ap, int);
    }
    va_end(ap);
    return count;
}

double va_weighted(double w, int n, ...) {
    va_list ap;
    va_start(ap, n);
    double total = 0.0;
    for (int i = 0; i < n; i++)
        total += va_arg(ap, double);
    va_end(ap);
    return w * total;
}
''')
    _va_mod = cimport(_va_header, sources=[_va_source],
                      compile_sources=True, include_dirs=[_fixture_dir])
    va_sum_ints = _va_mod.va_sum_ints
    va_sum_doubles = _va_mod.va_sum_doubles
    va_counted = _va_mod.va_counted
    va_weighted = _va_mod.va_weighted

    # ------------------------------------------------------------------
    # snprintf matrix
    # ------------------------------------------------------------------

    @compile
    def fmt_ints() -> i32:
        buf: array[i8, 96] = ""
        snprintf(buf, 96, "%d %d %d", 1, -20, 300)
        if strcmp(buf, "1 -20 300") != 0:
            return -1
        return 0

    @compile
    def fmt_doubles() -> i32:
        buf: array[i8, 96] = ""
        snprintf(buf, 96, "%.1f %.2f %g", 1.5, 2.25, 3.0)
        if strcmp(buf, "1.5 2.25 3") != 0:
            return -1
        return 0

    @compile
    def fmt_strings() -> i32:
        buf: array[i8, 96] = ""
        snprintf(buf, 96, "[%s|%s]", "ab", "cd")
        if strcmp(buf, "[ab|cd]") != 0:
            return -1
        return 0

    @compile
    def fmt_mixed() -> i32:
        buf: array[i8, 128] = ""
        n: i32 = snprintf(buf, 128, "%s=%d (%.3f)", "pi", 3, 3.14159)
        if strcmp(buf, "pi=3 (3.142)") != 0:
            return -1
        return n

    @compile
    def fmt_promotions() -> i32:
        # C default argument promotions: i8/i16 -> i32, f32 -> f64.
        buf: array[i8, 96] = ""
        small: i8 = 65
        mid: i16 = -320
        flt: f32 = f32(1.5)
        snprintf(buf, 96, "c=%c d=%d f=%.1f", small, mid, flt)
        if strcmp(buf, "c=A d=-320 f=1.5") != 0:
            return -1
        return 0

    @compile
    def fmt_wide() -> i32:
        buf: array[i8, 96] = ""
        big: i64 = 5000000000
        ubig: u64 = 18000000000000000000
        snprintf(buf, 96, "%lld %llu", big, ubig)
        if strcmp(buf, "5000000000 18000000000000000000") != 0:
            return -1
        return 0

    @compile
    def fmt_no_varargs() -> i32:
        buf: array[i8, 32] = ""
        n: i32 = snprintf(buf, 32, "literal only")
        if strcmp(buf, "literal only") != 0:
            return -1
        return n

    @compile
    def fmt_truncation() -> i32:
        # snprintf returns the would-be length even when truncated.
        buf: array[i8, 8] = ""
        n: i32 = snprintf(buf, 8, "%d", 123456789)
        if strcmp(buf, "1234567") != 0:
            return -1
        return n

    @compile
    def sprintf_into_buffer() -> i32:
        buf: array[i8, 64] = ""
        n: i32 = sprintf(buf, "%x", 255)
        if strcmp(buf, "ff") != 0:
            return -1
        return n

    @compile
    def fmt_runtime_args(x: i32, y: f64, s: ptr[i8]) -> i32:
        buf: array[i8, 96] = ""
        snprintf(buf, 96, "%d:%.1f:%s", x, y, s)
        return strcmp(buf, "7:2.5:ok")

    # ------------------------------------------------------------------
    # Custom variadic functions with varying argument counts
    # ------------------------------------------------------------------

    @compile
    def va_ints_vary() -> i32:
        r0: i32 = va_sum_ints(0)
        r1: i32 = va_sum_ints(1, 42)
        r3: i32 = va_sum_ints(3, 10, 20, 30)
        r5: i32 = va_sum_ints(5, 1, 2, 3, 4, 5)
        if r0 != 0 or r1 != 42 or r3 != 60 or r5 != 15:
            return -1
        return r1 + r3 + r5

    @compile
    def va_doubles_vary() -> i32:
        a: f64 = va_sum_doubles(1, 2.5)
        b: f64 = va_sum_doubles(3, 1.5, 2.25, 0.25)
        if a != 2.5 or b != 4.0:
            return -1
        return i32(a + b * 10.0)

    @compile
    def va_doubles_promoted_from_f32() -> i32:
        # f32 arguments are promoted to f64, which is what va_arg(ap,
        # double) reads back.
        one: f32 = f32(1.0)
        two: f32 = f32(2.0)
        r: f64 = va_sum_doubles(2, one, two)
        if r != 3.0:
            return -1
        return 0

    @compile
    def va_sentinel() -> i32:
        c1: i32 = va_counted(7, -1)
        c3: i32 = va_counted(5, 4, 3, -1)
        if c1 != 1 or c3 != 3:
            return -1
        return c1 + c3

    @compile
    def va_weighted_check() -> i32:
        # fixed double + fixed int before the varargs
        r: f64 = va_weighted(2.0, 3, 1.0, 2.0, 3.0)
        if r != 12.0:
            return -1
        return 0


@unittest.skipUnless(_BACKEND_AVAILABLE, "clang backend or cc not available")
class TestCimportVariadicSnprintf(unittest.TestCase):
    """printf-family varargs asserted through buffer contents."""

    def test_int_args(self):
        self.assertEqual(fmt_ints(), 0)

    def test_double_args(self):
        self.assertEqual(fmt_doubles(), 0)

    def test_string_args(self):
        self.assertEqual(fmt_strings(), 0)

    def test_mixed_args(self):
        self.assertEqual(fmt_mixed(), len("pi=3 (3.142)"))

    def test_default_argument_promotions(self):
        self.assertEqual(fmt_promotions(), 0)

    def test_i64_u64_args(self):
        self.assertEqual(fmt_wide(), 0)

    def test_zero_varargs(self):
        self.assertEqual(fmt_no_varargs(), len("literal only"))

    def test_truncation_return_value(self):
        self.assertEqual(fmt_truncation(), 9)

    def test_sprintf(self):
        self.assertEqual(sprintf_into_buffer(), 2)

    def test_runtime_args_from_python(self):
        self.assertEqual(fmt_runtime_args(7, 2.5, b"ok"), 0)


@unittest.skipUnless(_BACKEND_AVAILABLE, "clang backend or cc not available")
class TestCimportVariadicCustom(unittest.TestCase):
    """Custom variadic C functions called with varying argument counts."""

    def test_sum_ints_varying_counts(self):
        self.assertEqual(va_ints_vary(), 117)

    def test_sum_doubles_varying_counts(self):
        self.assertEqual(va_doubles_vary(), 42)

    def test_f32_promoted_to_double(self):
        self.assertEqual(va_doubles_promoted_from_f32(), 0)

    def test_sentinel_counting(self):
        self.assertEqual(va_sentinel(), 4)

    def test_fixed_args_before_varargs(self):
        self.assertEqual(va_weighted_check(), 0)


if __name__ == "__main__":
    unittest.main()
