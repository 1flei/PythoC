# -*- coding: utf-8 -*-
"""
End-to-end tests for cimport of the real system C library.

Imports the actual platform headers by name (stdlib.h, string.h, stdio.h,
math.h) and calls real libc functions from @compile code.

Covers:
- atoi / atof / strtol / abs round-trips through real libc
- strlen / strcmp / strcpy / strcat / strerror into stack buffers
- malloc / free with real memory reads and writes through pointers
- snprintf into buffers (variadic end-to-end; deeper matrix in
  test_cimport_variadic.py)
- sqrt / sin / pow / fabs with f64 results
- qsort of an i32 array and of a struct array with @compile comparators
- fopen / fwrite / fread / fclose round-trip through an opaque FILE*
  (exercises opaque placeholder types from real system headers)
- numeric macros from real headers (EXIT_SUCCESS / EXIT_FAILURE / EOF)
- a combined scenario: parse -> compute -> format -> compare

Platform notes:
- System headers are imported with includes=True because they delegate
  declarations to private sub-headers (glibc math.h -> bits/mathcalls.h,
  macOS string.h/stdio.h -> _string.h/_stdio.h, stdlib.h -> malloc/_malloc.h
  for malloc/free); the default main-file-only emission would miss them.
- math functions need lib='m' on Linux (libm is separate from libc);
  on macOS libm is folded into libSystem so lib='c' is used.
- plain C char is unsigned on ARM64 Linux and signed on x86/macOS, so
  libc 'char *' parameters bind as ptr[u8] or ptr[i8] depending on the
  target; wrappers that pass typed pointer values across that boundary
  use the derived _cchar_ptr type (see below) to compile either way.
- The whole file is skipped on Windows: the pip libclang cannot parse
  zig's bundled mingw libc headers (any-windows-any/stdlib.h), so real
  libc headers are not importable there.

Note: @compile wrappers are defined at module level because pythoc requires
all @compile definitions to precede the first native call from this module.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

from pythoc import (
    compile, i8, i32, i64, u64, f64, ptr, array, sizeof, nullptr, void,
)

IS_LINUX = sys.platform.startswith('linux')


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


_BACKEND_AVAILABLE = (
    _clang_backend_available()
    and _cc_available()
    # The pip libclang cannot parse zig's bundled mingw libc headers.
    and sys.platform != 'win32'
)

# =============================================================================
# Module-level fixtures: cimport of real system headers + @compile wrappers
# =============================================================================

if _BACKEND_AVAILABLE:
    from pythoc.cimport import cimport

    _stdlib = cimport('stdlib.h', lib='c', includes=True)
    _string = cimport('string.h', lib='c', includes=True)
    _stdio = cimport('stdio.h', lib='c', includes=True)
    # libm is a separate library on Linux; on macOS it is part of libSystem.
    _math = cimport('math.h', lib='m' if IS_LINUX else 'c', includes=True)
    # malloc/free are declared in malloc/_malloc.h in the macOS SDK and
    # reached through stdlib.h via includes=True; glibc declares them in
    # stdlib.h itself.
    _malloc_mod = _stdlib

    atoi = _stdlib.atoi
    atof = _stdlib.atof
    strtol = _stdlib.strtol
    abs_ = _stdlib.abs

    strlen = _string.strlen
    strcmp = _string.strcmp
    strcpy = _string.strcpy
    strcat = _string.strcat
    strerror = _string.strerror

    snprintf = _stdio.snprintf
    fopen = _stdio.fopen
    fwrite = _stdio.fwrite
    fread = _stdio.fread
    fclose = _stdio.fclose
    fseek = _stdio.fseek
    remove = _stdio.remove

    malloc = _malloc_mod.malloc
    free = _malloc_mod.free

    sqrt = _math.sqrt
    sin = _math.sin
    pow = _math.pow
    fabs = _math.fabs

    qsort = _stdlib.qsort

    SEEK_SET_C = _stdio.SEEK_SET

    # Plain C char is unsigned on some targets (ARM64 Linux) and signed on
    # others (x86, macOS); the bindings mirror the target ABI, so a libc
    # 'char *' parameter is ptr[u8] on some targets and ptr[i8] on others.
    # Derive the char pointer type from a real declaration and use it for
    # wrapper parameters/locals that cross a libc 'char *' boundary, so
    # the wrappers compile with either signedness.  (Arrays and string
    # literals adapt to the parameter type on their own; only typed
    # pointer values need this.)
    _cchar_ptr = atoi.param_types[0][1]

    # ------------------------------------------------------------------
    # stdlib: numeric conversions
    # ------------------------------------------------------------------

    @compile
    def atoi_of(s: _cchar_ptr) -> i32:
        return atoi(s)

    @compile
    def atof_doubled(s: _cchar_ptr) -> i32:
        return i32(atof(s) * 2.0)

    @compile
    def strtol_base16(s: _cchar_ptr) -> i32:
        endp: _cchar_ptr = nullptr
        v: i64 = strtol(s, ptr(endp), 16)
        consumed: i64 = i64(endp) - i64(s)
        return i32(v) + i32(consumed)

    @compile
    def abs_of(x: i32) -> i32:
        return abs_(x)

    # ------------------------------------------------------------------
    # string: buffer operations
    # ------------------------------------------------------------------

    @compile
    def string_ops() -> i32:
        buf: array[i8, 64] = ""
        strcpy(buf, "foo")
        strcat(buf, "bar")
        if strcmp(buf, "foobar") != 0:
            return 1
        if i32(strlen(buf)) != 6:
            return 2
        return 0

    @compile
    def strerror_enoent() -> i32:
        msg: _cchar_ptr = strerror(2)  # ENOENT on both macOS and Linux
        if msg == nullptr:
            return 1
        if i32(strlen(msg)) == 0:
            return 2
        if strcmp(msg, "No such file or directory") != 0:
            return 3
        return 0

    @compile
    def strlen_of(s: _cchar_ptr) -> i32:
        return i32(strlen(s))

    # ------------------------------------------------------------------
    # malloc / free
    # ------------------------------------------------------------------

    @compile
    def malloc_squares(n: i32) -> i32:
        p: ptr[i32] = ptr[i32](malloc(u64(n) * 4))
        if p == nullptr:
            return -1
        i: i32 = 0
        while i < n:
            p[i] = i * i
            i += 1
        total: i32 = 0
        i = 0
        while i < n:
            total += p[i]
            i += 1
        free(p)
        return total

    @compile
    def malloc_bytes() -> i32:
        p: ptr[i8] = ptr[i8](malloc(8))
        if p == nullptr:
            return -1
        for i in range(8):
            p[i] = i8(65 + i)  # "ABCDEFGH"
        ok: i32 = 0
        if p[0] == 65 and p[7] == 72:
            ok = 1
        free(p)
        return ok

    # ------------------------------------------------------------------
    # stdio: snprintf + FILE* round-trip
    # ------------------------------------------------------------------

    @compile
    def snprintf_basic() -> i32:
        buf: array[i8, 64] = ""
        n: i32 = snprintf(buf, 64, "v=%d f=%.2f s=%s", 42, 3.25, "xy")
        if strcmp(buf, "v=42 f=3.25 s=xy") != 0:
            return -1
        return n

    @compile
    def file_roundtrip(path: _cchar_ptr) -> i32:
        f = fopen(path, "w+")
        if f == nullptr:
            return 1
        data: array[i8, 6] = "abcde"
        written: u64 = fwrite(data, 1, 5, f)
        if written != 5:
            fclose(f)
            return 2
        if fseek(f, 0, SEEK_SET_C) != 0:
            fclose(f)
            return 3
        out: array[i8, 8] = ""
        got: u64 = fread(out, 1, 5, f)
        if got != 5:
            fclose(f)
            return 4
        if fclose(f) != 0:
            return 5
        if strcmp(out, "abcde") != 0:
            return 6
        return 0

    # ------------------------------------------------------------------
    # math
    # ------------------------------------------------------------------

    @compile
    def math_ops() -> i32:
        if sqrt(16.0) != 4.0:
            return 1
        if pow(2.0, 10.0) != 1024.0:
            return 2
        if fabs(-3.5) != 3.5:
            return 3
        if fabs(2.5) != 2.5:
            return 4
        s: f64 = sin(0.0)
        if s != 0.0:
            return 5
        return 0

    @compile
    def hypot_like(a: f64, b: f64) -> f64:
        return sqrt(pow(a, 2.0) + pow(b, 2.0))

    # ------------------------------------------------------------------
    # qsort with @compile comparators
    # ------------------------------------------------------------------

    @compile
    def cmp_i32(a: ptr[void], b: ptr[void]) -> i32:
        pa: ptr[i32] = ptr[i32](a)
        pb: ptr[i32] = ptr[i32](b)
        if pa[0] < pb[0]:
            return -1
        if pa[0] > pb[0]:
            return 1
        return 0

    @compile
    def qsort_i32_array() -> i32:
        arr: array[i32, 6] = [9, -3, 7, 1, 0, 4]
        qsort(arr, 6, 4, cmp_i32)
        for i in range(5):
            if arr[i] > arr[i + 1]:
                return -1
        return arr[0] * 100 + arr[5]

    @compile
    class Rec:
        key: i32
        val: i32

    @compile
    def cmp_rec(a: ptr[void], b: ptr[void]) -> i32:
        ra: ptr[Rec] = ptr[Rec](a)
        rb: ptr[Rec] = ptr[Rec](b)
        if ra.key < rb.key:
            return -1
        if ra.key > rb.key:
            return 1
        return 0

    @compile
    def qsort_struct_array() -> i32:
        arr: array[Rec, 4]
        arr[0].key = 30
        arr[0].val = 3
        arr[1].key = 10
        arr[1].val = 1
        arr[2].key = 40
        arr[2].val = 4
        arr[3].key = 20
        arr[3].val = 2
        qsort(arr, 4, u64(sizeof(Rec)), cmp_rec)
        for i in range(3):
            if arr[i].key > arr[i + 1].key:
                return -1
        # keys 10,20,30,40 -> vals must ride along with their keys
        return (arr[0].val * 1000 + arr[1].val * 100
                + arr[2].val * 10 + arr[3].val)

    # ------------------------------------------------------------------
    # Combined scenario: parse -> compute -> format -> compare
    # ------------------------------------------------------------------

    @compile
    def pipeline(a: _cchar_ptr, b: _cchar_ptr, expect: _cchar_ptr) -> i32:
        va: i32 = atoi(a)
        vb: i32 = atoi(b)
        product: i32 = va * vb
        buf: array[i8, 64] = ""
        snprintf(buf, 64, "R=%d", product)
        if strcmp(buf, expect) != 0:
            return 1
        return 0


@unittest.skipUnless(_BACKEND_AVAILABLE, "clang backend or cc not available")
class TestCimportLibcStdlib(unittest.TestCase):
    """Numeric conversions through the real libc."""

    def test_atoi(self):
        self.assertEqual(atoi_of(b"123"), 123)
        self.assertEqual(atoi_of(b"-45"), -45)
        self.assertEqual(atoi_of(b"0"), 0)

    def test_atof(self):
        self.assertEqual(atof_doubled(b"2.5"), 5)
        self.assertEqual(atof_doubled(b"2.5e1"), 50)

    def test_strtol_base_and_endptr(self):
        # "ff" is 255 in base 16 and 2 chars are consumed.
        self.assertEqual(strtol_base16(b"ffzz"), 257)

    def test_abs(self):
        self.assertEqual(abs_of(-7), 7)
        self.assertEqual(abs_of(7), 7)


@unittest.skipUnless(_BACKEND_AVAILABLE, "clang backend or cc not available")
class TestCimportLibcString(unittest.TestCase):
    """String functions into stack buffers."""

    def test_strcpy_strcat_strcmp_strlen(self):
        self.assertEqual(string_ops(), 0)

    def test_strerror_enoent(self):
        self.assertEqual(strerror_enoent(), 0)

    def test_strlen_from_python_arg(self):
        self.assertEqual(strlen_of(b"hello world"), 11)
        self.assertEqual(strlen_of(b""), 0)


@unittest.skipUnless(_BACKEND_AVAILABLE, "clang backend or cc not available")
class TestCimportLibcMalloc(unittest.TestCase):
    """malloc/free with real memory traffic."""

    def test_malloc_i32_read_write(self):
        # sum of i*i for i in 0..9
        self.assertEqual(malloc_squares(10), 285)

    def test_malloc_byte_write_read(self):
        self.assertEqual(malloc_bytes(), 1)


@unittest.skipUnless(_BACKEND_AVAILABLE, "clang backend or cc not available")
class TestCimportLibcStdio(unittest.TestCase):
    """snprintf and FILE* round-trip."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_snprintf_basic(self):
        self.assertEqual(snprintf_basic(), len("v=42 f=3.25 s=xy"))

    def test_file_roundtrip(self):
        path = os.path.join(self.temp_dir, "io.bin")
        self.assertEqual(file_roundtrip(path.encode()), 0)
        with open(path, "rb") as f:
            self.assertEqual(f.read(), b"abcde")

    def test_remove(self):
        path = os.path.join(self.temp_dir, "gone.bin")
        self.assertEqual(file_roundtrip(path.encode()), 0)
        self.assertTrue(os.path.exists(path))
        self.assertEqual(remove(path.encode()), 0)
        self.assertFalse(os.path.exists(path))


@unittest.skipUnless(_BACKEND_AVAILABLE, "clang backend or cc not available")
class TestCimportLibcMath(unittest.TestCase):
    """Math functions with f64 results."""

    def test_math_ops(self):
        self.assertEqual(math_ops(), 0)

    def test_math_results_in_arithmetic(self):
        self.assertAlmostEqual(hypot_like(3.0, 4.0), 5.0)
        self.assertAlmostEqual(hypot_like(5.0, 12.0), 13.0)


@unittest.skipUnless(_BACKEND_AVAILABLE, "clang backend or cc not available")
class TestCimportLibcQsort(unittest.TestCase):
    """qsort with @compile comparators (doubles as fn-ptr test)."""

    def test_qsort_i32(self):
        # min * 100 + max = -3 * 100 + 9
        self.assertEqual(qsort_i32_array(), -291)

    def test_qsort_struct_array(self):
        # vals must follow their keys: keys 10,20,30,40 -> vals 1,2,3,4
        self.assertEqual(qsort_struct_array(), 1234)


@unittest.skipUnless(_BACKEND_AVAILABLE, "clang backend or cc not available")
class TestCimportLibcConstants(unittest.TestCase):
    """Numeric macros from the real headers.

    Only asserts values that hold identically on macOS and Linux.
    """

    def test_exit_codes(self):
        self.assertEqual(_stdlib.EXIT_SUCCESS, 0)
        self.assertEqual(_stdlib.EXIT_FAILURE, 1)

    def test_eof(self):
        # `#define EOF (-1)` on both platforms; the parenthesized negative
        # literal must survive the conservative macro rule.
        self.assertEqual(_stdio.EOF, -1)

    def test_seek_set(self):
        self.assertEqual(_stdio.SEEK_SET, 0)

    def test_rand_max_positive(self):
        self.assertGreater(_stdlib.RAND_MAX, 0)


@unittest.skipUnless(_BACKEND_AVAILABLE, "clang backend or cc not available")
class TestCimportLibcPipeline(unittest.TestCase):
    """Combined: parse argv-style strings -> compute -> format -> compare."""

    def test_pipeline(self):
        self.assertEqual(pipeline(b"17", b"3", b"R=51"), 0)
        self.assertEqual(pipeline(b"-6", b"7", b"R=-42"), 0)

    def test_pipeline_mismatch_detected(self):
        self.assertEqual(pipeline(b"17", b"3", b"R=52"), 1)


if __name__ == "__main__":
    unittest.main()
