#!/usr/bin/env python3
"""
Regex benchmark: PythoC (Python-level DFA + compiled native) vs Python re.

Compares four modes:
  1. Python re       -- stdlib re module
  2. PythoC DFA      -- Python-level match/search (CompiledRegex methods)
  3. PythoC native   -- @compile-generated LLVM native code called via ctypes
  4. PythoC pure     -- @compile loop calling regex fn, timed with clock()
                        (no Python/ctypes overhead, pure native speed)

All @compile and generate_*_fn() calls happen at module level.
"""

import ctypes
import re
import signal
import time
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc import compile as pc_compile, u8, u64, i8, i64, ptr
from pythoc.decorators.extern import extern
from pythoc.regex import compile as regex_compile


# ============================================================================
# Benchmark infrastructure
# ============================================================================

_RE_TIMEOUT = 5.0


class _BenchTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _BenchTimeout()


def _bench(fn, *args, warmup=2, repeats=5, timeout=0):
    use_alarm = timeout > 0 and hasattr(signal, 'SIGALRM')
    old_handler = None
    if use_alarm:
        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    try:
        result = None
        for i in range(warmup):
            if use_alarm:
                signal.alarm(int(timeout) + 1)
            result = fn(*args)
        best = float('inf')
        for i in range(repeats):
            if use_alarm:
                signal.alarm(int(timeout) + 1)
            t0 = time.perf_counter()
            result = fn(*args)
            elapsed = time.perf_counter() - t0
            if elapsed < best:
                best = elapsed
        return best, result
    except _BenchTimeout:
        return None, None
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def _bytes_to_cargs(data: bytes):
    n = len(data)
    buf = ctypes.create_string_buffer(data, n)
    ptr_val = ctypes.cast(buf, ctypes.c_void_p).value
    return n, ptr_val, buf


# ============================================================================
# Test data builders
# ============================================================================

def _build_no_match(size):
    return b'x' * size


def _build_match_at_end(size, suffix):
    return b'x' * (size - len(suffix)) + suffix


def _build_match_at_middle(size, needle):
    mid = size // 2
    return b'x' * mid + needle + b'x' * (size - mid - len(needle))


# ============================================================================
# Extern: clock() for pure native timing
# ============================================================================

@extern
def clock() -> i64:
    pass


# ============================================================================
# Compile native regex functions at module level
# ============================================================================

# Pattern 1: literal "needle"
_r_needle = regex_compile("needle")
_needle_match = _r_needle.generate_match_fn()
_needle_search = _r_needle.generate_search_fn()
_needle_search_slots = _r_needle.search_num_slots

# Pattern 2: digit class "\\d+"
_r_digits = regex_compile("\\d+")
_digits_match = _r_digits.generate_match_fn()
_digits_search = _r_digits.generate_search_fn()
_digits_search_slots = _r_digits.search_num_slots

# Pattern 3: alternation "alpha|beta|gamma"
_r_alt = regex_compile("alpha|beta|gamma")
_alt_match = _r_alt.generate_match_fn()
_alt_search = _r_alt.generate_search_fn()
_alt_search_slots = _r_alt.search_num_slots

# Pattern 4: dot-star "a.*z"
_r_dotstar = regex_compile("a.*z")
_dotstar_match = _r_dotstar.generate_match_fn()

# Pattern 5: email-like "[a-z]+@[a-z]+\\.[a-z]+"
_r_email = regex_compile("[a-z]+@[a-z]+\\.[a-z]+")
_email_match = _r_email.generate_match_fn()
_email_search = _r_email.generate_search_fn()
_email_search_slots = _r_email.search_num_slots


# ============================================================================
# Pure-native benchmark loops (timed with clock(), no Python overhead)
#
# Match: (data_n, data) -> ticks
# Search: (data_n, data, out) -> ticks   [out is ptr[i64] for tag slots]
# ============================================================================

_PURE_ITERS = 100


@pc_compile
def _bench_pure_needle_match(data_n: u64, data: ptr[i8]) -> i64:
    t0: i64 = clock()
    k: i64 = i64(0)
    while k < i64(_PURE_ITERS):
        _needle_match(data_n, data)
        k = k + i64(1)
    t1: i64 = clock()
    return t1 - t0


@pc_compile
def _bench_pure_needle_search(data_n: u64, data: ptr[i8], out: ptr[i64]) -> i64:
    t0: i64 = clock()
    k: i64 = i64(0)
    while k < i64(_PURE_ITERS):
        _needle_search(data_n, data, out)
        k = k + i64(1)
    t1: i64 = clock()
    return t1 - t0


@pc_compile
def _bench_pure_digits_match(data_n: u64, data: ptr[i8]) -> i64:
    t0: i64 = clock()
    k: i64 = i64(0)
    while k < i64(_PURE_ITERS):
        _digits_match(data_n, data)
        k = k + i64(1)
    t1: i64 = clock()
    return t1 - t0


@pc_compile
def _bench_pure_digits_search(data_n: u64, data: ptr[i8], out: ptr[i64]) -> i64:
    t0: i64 = clock()
    k: i64 = i64(0)
    while k < i64(_PURE_ITERS):
        _digits_search(data_n, data, out)
        k = k + i64(1)
    t1: i64 = clock()
    return t1 - t0


@pc_compile
def _bench_pure_alt_match(data_n: u64, data: ptr[i8]) -> i64:
    t0: i64 = clock()
    k: i64 = i64(0)
    while k < i64(_PURE_ITERS):
        _alt_match(data_n, data)
        k = k + i64(1)
    t1: i64 = clock()
    return t1 - t0


@pc_compile
def _bench_pure_alt_search(data_n: u64, data: ptr[i8], out: ptr[i64]) -> i64:
    t0: i64 = clock()
    k: i64 = i64(0)
    while k < i64(_PURE_ITERS):
        _alt_search(data_n, data, out)
        k = k + i64(1)
    t1: i64 = clock()
    return t1 - t0


@pc_compile
def _bench_pure_dotstar_match(data_n: u64, data: ptr[i8]) -> i64:
    t0: i64 = clock()
    k: i64 = i64(0)
    while k < i64(_PURE_ITERS):
        _dotstar_match(data_n, data)
        k = k + i64(1)
    t1: i64 = clock()
    return t1 - t0


@pc_compile
def _bench_pure_email_match(data_n: u64, data: ptr[i8]) -> i64:
    t0: i64 = clock()
    k: i64 = i64(0)
    while k < i64(_PURE_ITERS):
        _email_match(data_n, data)
        k = k + i64(1)
    t1: i64 = clock()
    return t1 - t0


@pc_compile
def _bench_pure_email_search(data_n: u64, data: ptr[i8], out: ptr[i64]) -> i64:
    t0: i64 = clock()
    k: i64 = i64(0)
    while k < i64(_PURE_ITERS):
        _email_search(data_n, data, out)
        k = k + i64(1)
    t1: i64 = clock()
    return t1 - t0


# ============================================================================
# Benchmark runner
# ============================================================================

_SIZES = [50_000, 200_000, 1_000_000]

_CLOCKS_PER_SEC = 1_000_000


def _print_header(title):
    print()
    print("=" * 96)
    print(f"  {title}")
    print("=" * 96)
    print(f"  {'Size':>10s}  {'python re':>12s}  {'PC DFA':>12s}  "
          f"{'PC native':>12s}  {'PC pure':>12s}  "
          f"{'re/native':>10s}  {'re/pure':>10s}")
    print("-" * 96)


def _fmt_time(t):
    if t is None:
        return "   TIMEOUT"
    return f"{t*1000:>10.3f}ms"


def _fmt_ratio(t_num, t_den):
    if t_num is None or t_den is None or t_den < 1e-9:
        return "       N/A"
    return f"{t_num / t_den:>9.1f}x"


def _print_row(size, t_re, t_dfa, t_native, t_pure):
    print(f"  {size:>10,d}  {_fmt_time(t_re)}  {_fmt_time(t_dfa)}  "
          f"{_fmt_time(t_native)}  {_fmt_time(t_pure)}  "
          f"{_fmt_ratio(t_re, t_native)}  {_fmt_ratio(t_re, t_pure)}")


def _run_pure_match_bench(pure_fn, n, ptr_val):
    pure_fn(n, ptr_val)
    best_ticks = None
    for _ in range(3):
        ticks = pure_fn(n, ptr_val)
        if best_ticks is None or ticks < best_ticks:
            best_ticks = ticks
    total_sec = best_ticks / _CLOCKS_PER_SEC
    return total_sec / _PURE_ITERS


def _run_pure_search_bench(pure_fn, n, ptr_val, num_slots):
    out_buf = (ctypes.c_int64 * num_slots)()
    out_ptr = ctypes.cast(out_buf, ctypes.c_void_p).value
    pure_fn(n, ptr_val, out_ptr)
    best_ticks = None
    for _ in range(3):
        ticks = pure_fn(n, ptr_val, out_ptr)
        if best_ticks is None or ticks < best_ticks:
            best_ticks = ticks
    total_sec = best_ticks / _CLOCKS_PER_SEC
    return total_sec / _PURE_ITERS


def _make_native_search_caller(native_fn, num_slots, beg_slot):
    """Wrap a native search fn so it returns start position (int) like old API."""
    out_buf = (ctypes.c_int64 * num_slots)()
    out_ptr = ctypes.cast(out_buf, ctypes.c_void_p).value
    def caller(n, ptr_val):
        matched = native_fn(n, ptr_val, out_ptr)
        if matched:
            return int(out_buf[beg_slot])
        return -1
    return caller


def _run_benchmark_match(title, pattern_str, py_re, pc_regex,
                          native_fn, build_data, pure_fn=None):
    _print_header(f"match: /{pattern_str}/  --  {title}")
    for size in _SIZES:
        data = build_data(size)
        n, ptr_val, buf = _bytes_to_cargs(data)

        t_re, res_re = _bench(
            lambda d=data: bool(py_re.match(d)),
            timeout=_RE_TIMEOUT,
        )
        t_dfa, res_dfa = _bench(lambda d=data: pc_regex.match(d)[0])
        t_nat, res_nat = _bench(native_fn, n, ptr_val)

        t_pure = None
        if pure_fn is not None:
            t_pure = _run_pure_match_bench(pure_fn, n, ptr_val)

        res_dfa_bool = bool(res_dfa)
        res_nat_bool = bool(res_nat)
        if t_re is not None:
            res_re_bool = bool(res_re)
            assert res_re_bool == res_dfa_bool == res_nat_bool, (
                f"Result mismatch at size={size}: "
                f"re={res_re_bool} dfa={res_dfa_bool} native={res_nat_bool}"
            )
        else:
            assert res_dfa_bool == res_nat_bool, (
                f"Result mismatch at size={size}: "
                f"dfa={res_dfa_bool} native={res_nat_bool}"
            )

        _print_row(size, t_re, t_dfa, t_nat, t_pure)


def _run_benchmark_search(title, pattern_str, py_re, pc_regex,
                           native_caller, build_data, pure_fn=None,
                           num_slots=0):
    _print_header(f"search: /{pattern_str}/  --  {title}")
    for size in _SIZES:
        data = build_data(size)
        n, ptr_val, buf = _bytes_to_cargs(data)

        def py_search(d=data):
            m = py_re.search(d)
            return m.start() if m else -1

        t_re, res_re = _bench(py_search, timeout=_RE_TIMEOUT)

        def dfa_search(d=data):
            ok, info = pc_regex.search(d)
            return info.get('start', -1) if ok else -1

        t_dfa, res_dfa = _bench(dfa_search)
        t_nat, res_nat = _bench(native_caller, n, ptr_val)

        t_pure = None
        if pure_fn is not None:
            t_pure = _run_pure_search_bench(pure_fn, n, ptr_val, num_slots)

        if t_re is not None:
            assert res_re == res_dfa == res_nat, (
                f"Result mismatch at size={size}: "
                f"re={res_re} dfa={res_dfa} native={res_nat}"
            )
        else:
            assert res_dfa == res_nat, (
                f"Result mismatch at size={size}: "
                f"dfa={res_dfa} native={res_nat}"
            )

        _print_row(size, t_re, t_dfa, t_nat, t_pure)


# ============================================================================
# Build native search callers (wrap 3-arg ABI into position-returning fn)
# ============================================================================

_needle_search_caller = _make_native_search_caller(
    _needle_search, _needle_search_slots,
    _r_needle._search_bma.tag_slots['__pythoc_internal_beg'])

_digits_search_caller = _make_native_search_caller(
    _digits_search, _digits_search_slots,
    _r_digits._search_bma.tag_slots['__pythoc_internal_beg'])

_alt_search_caller = _make_native_search_caller(
    _alt_search, _alt_search_slots,
    _r_alt._search_bma.tag_slots['__pythoc_internal_beg'])

_email_search_caller = _make_native_search_caller(
    _email_search, _email_search_slots,
    _r_email._search_bma.tag_slots['__pythoc_internal_beg'])


# ============================================================================
# Test class (unittest-based, prints benchmark table)
# ============================================================================

class TestRegexBenchmark(unittest.TestCase):

    # --- match benchmarks ---

    def test_literal_no_match(self):
        _run_benchmark_match(
            "no match", "needle",
            re.compile(b"needle"), _r_needle, _needle_match,
            _build_no_match,
            pure_fn=_bench_pure_needle_match,
        )

    def test_literal_match_at_start(self):
        _run_benchmark_match(
            "match at start", "needle",
            re.compile(b"needle"), _r_needle, _needle_match,
            lambda sz: b"needle" + b'x' * (sz - 6),
            pure_fn=_bench_pure_needle_match,
        )

    def test_digits_no_match(self):
        _run_benchmark_match(
            "no match", "\\d+",
            re.compile(rb"\d+"), _r_digits, _digits_match,
            _build_no_match,
            pure_fn=_bench_pure_digits_match,
        )

    def test_dotstar(self):
        _run_benchmark_match(
            "a...z at edges", "a.*z",
            re.compile(b"a.*z"), _r_dotstar, _dotstar_match,
            lambda sz: b'a' + b'm' * (sz - 2) + b'z',
            pure_fn=_bench_pure_dotstar_match,
        )

    def test_email_no_match(self):
        _run_benchmark_match(
            "no match (re: catastrophic backtracking)",
            "[a-z]+@[a-z]+\\.[a-z]+",
            re.compile(rb"[a-z]+@[a-z]+\.[a-z]+"), _r_email, _email_match,
            _build_no_match,
            pure_fn=_bench_pure_email_match,
        )

    # --- search benchmarks ---

    def test_search_literal_no_match(self):
        _run_benchmark_search(
            "no match", "needle",
            re.compile(b"needle"), _r_needle, _needle_search_caller,
            _build_no_match,
            pure_fn=_bench_pure_needle_search,
            num_slots=_needle_search_slots,
        )

    def test_search_literal_match_end(self):
        _run_benchmark_search(
            "match at end", "needle",
            re.compile(b"needle"), _r_needle, _needle_search_caller,
            lambda sz: _build_match_at_end(sz, b"needle"),
            pure_fn=_bench_pure_needle_search,
            num_slots=_needle_search_slots,
        )

    def test_search_digits_match_end(self):
        _run_benchmark_search(
            "match at end", "\\d+",
            re.compile(rb"\d+"), _r_digits, _digits_search_caller,
            lambda sz: _build_match_at_end(sz, b"12345"),
            pure_fn=_bench_pure_digits_search,
            num_slots=_digits_search_slots,
        )

    def test_search_alt_match_middle(self):
        _run_benchmark_search(
            "match in middle", "alpha|beta|gamma",
            re.compile(b"alpha|beta|gamma"), _r_alt, _alt_search_caller,
            lambda sz: _build_match_at_middle(sz, b"gamma"),
            pure_fn=_bench_pure_alt_search,
            num_slots=_alt_search_slots,
        )

    def test_search_email_match_middle(self):
        _run_benchmark_search(
            "match in middle (re: catastrophic backtracking)",
            "[a-z]+@[a-z]+\\.[a-z]+",
            re.compile(rb"[a-z]+@[a-z]+\.[a-z]+"), _r_email, _email_search_caller,
            lambda sz: _build_match_at_middle(sz, b"user@host.com"),
            pure_fn=_bench_pure_email_search,
            num_slots=_email_search_slots,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
