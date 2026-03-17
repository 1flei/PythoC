#!/usr/bin/env python3
"""
Regex benchmark: PythoC (Python-level DFA + compiled native) vs Python re.

Compares three modes:
  1. Python re       -- stdlib re module
  2. PythoC DFA      -- Python-level DFA simulation (CompiledRegex methods)
  3. PythoC native   -- @compile-generated LLVM native code via native executor

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
from pythoc.regex import compile as regex_compile


# ============================================================================
# Benchmark infrastructure
# ============================================================================

# Timeout for a single benchmark call (seconds).
# Python re can exhibit catastrophic backtracking on certain patterns.
_RE_TIMEOUT = 5.0


class _BenchTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _BenchTimeout()


def _bench(fn, *args, warmup=2, repeats=5, timeout=0):
    """Benchmark fn(*args). Returns (best_sec, result).

    If timeout > 0 (seconds), aborts with (None, None) when the first
    call exceeds the deadline. Uses SIGALRM on Unix.
    """
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
    """Convert Python bytes to (n, ptr) ctypes args for native regex fns."""
    n = len(data)
    buf = ctypes.create_string_buffer(data, n)
    ptr_val = ctypes.cast(buf, ctypes.c_void_p).value
    return n, ptr_val, buf  # keep buf alive


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
# Compile native regex functions at module level
# ============================================================================

# Pattern 1: literal "needle"
_r_needle = regex_compile("needle")
_needle_is_match = _r_needle.generate_is_match_fn()
_needle_search = _r_needle.generate_search_fn()

# Pattern 2: digit class "\\d+"
_r_digits = regex_compile("\\d+")
_digits_is_match = _r_digits.generate_is_match_fn()
_digits_search = _r_digits.generate_search_fn()

# Pattern 3: alternation "alpha|beta|gamma"
_r_alt = regex_compile("alpha|beta|gamma")
_alt_is_match = _r_alt.generate_is_match_fn()
_alt_search = _r_alt.generate_search_fn()

# Pattern 4: dot-star "a.*z"
_r_dotstar = regex_compile("a.*z")
_dotstar_is_match = _r_dotstar.generate_is_match_fn()

# Pattern 5: email-like "[a-z]+@[a-z]+\\.[a-z]+"
_r_email = regex_compile("[a-z]+@[a-z]+\\.[a-z]+")
_email_is_match = _r_email.generate_is_match_fn()
_email_search = _r_email.generate_search_fn()


# ============================================================================
# Benchmark runner
# ============================================================================

_SIZES = [50_000, 200_000, 1_000_000]


def _print_header(title):
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)
    print(f"  {'Size':>10s}  {'python re':>12s}  {'PC DFA':>12s}  "
          f"{'PC native':>12s}  {'re/DFA':>8s}  {'re/native':>8s}")
    print("-" * 78)


def _fmt_time(t):
    """Format time in ms, or 'TIMEOUT' if None."""
    if t is None:
        return "   TIMEOUT"
    return f"{t*1000:>10.3f}ms"


def _fmt_ratio(t_num, t_den):
    """Format ratio, handling None (timeout)."""
    if t_num is None or t_den is None or t_den < 1e-9:
        return "     N/A"
    return f"{t_num / t_den:>7.1f}x"


def _print_row(size, t_re, t_dfa, t_native):
    print(f"  {size:>10,d}  {_fmt_time(t_re)}  {_fmt_time(t_dfa)}  "
          f"{_fmt_time(t_native)}  {_fmt_ratio(t_re, t_dfa)}  "
          f"{_fmt_ratio(t_re, t_native)}")


def _run_benchmark_is_match(title, pattern_str, py_re, pc_regex,
                             native_fn, build_data):
    """Run is_match benchmark across all sizes."""
    _print_header(f"is_match: /{pattern_str}/  --  {title}")
    for size in _SIZES:
        data = build_data(size)
        n, ptr_val, buf = _bytes_to_cargs(data)

        t_re, res_re = _bench(
            lambda d=data: bool(py_re.search(d)),
            timeout=_RE_TIMEOUT,
        )
        t_dfa, res_dfa = _bench(pc_regex.is_match, data)
        t_nat, res_nat = _bench(native_fn, n, ptr_val)

        # Verify results agree (skip re if it timed out)
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

        _print_row(size, t_re, t_dfa, t_nat)


def _run_benchmark_search(title, pattern_str, py_re, pc_regex,
                           native_fn, build_data):
    """Run search benchmark across all sizes."""
    _print_header(f"search: /{pattern_str}/  --  {title}")
    for size in _SIZES:
        data = build_data(size)
        n, ptr_val, buf = _bytes_to_cargs(data)

        def py_search(d=data):
            m = py_re.search(d)
            return m.start() if m else -1

        t_re, res_re = _bench(py_search, timeout=_RE_TIMEOUT)
        t_dfa, res_dfa = _bench(pc_regex.search, data)
        t_nat, res_nat = _bench(native_fn, n, ptr_val)

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

        _print_row(size, t_re, t_dfa, t_nat)


# ============================================================================
# Test class (unittest-based, prints benchmark table)
# ============================================================================

class TestRegexBenchmark(unittest.TestCase):
    """Regex benchmark: PythoC vs Python re.

    Each test prints a comparison table and verifies result correctness.
    """

    # --- is_match benchmarks ---

    def test_literal_no_match(self):
        """Literal pattern, no match (worst case for naive O(n^2))."""
        _run_benchmark_is_match(
            "no match", "needle",
            re.compile(b"needle"), _r_needle, _needle_is_match,
            _build_no_match,
        )

    def test_literal_match_at_end(self):
        """Literal pattern, match at very end."""
        _run_benchmark_is_match(
            "match at end", "needle",
            re.compile(b"needle"), _r_needle, _needle_is_match,
            lambda sz: _build_match_at_end(sz, b"needle"),
        )

    def test_digits_no_match(self):
        """Digit class on all-alpha input."""
        _run_benchmark_is_match(
            "no match", "\\d+",
            re.compile(rb"\d+"), _r_digits, _digits_is_match,
            _build_no_match,
        )

    def test_alternation_match_middle(self):
        """Alternation, match in middle."""
        _run_benchmark_is_match(
            "match in middle", "alpha|beta|gamma",
            re.compile(b"alpha|beta|gamma"), _r_alt, _alt_is_match,
            lambda sz: _build_match_at_middle(sz, b"beta"),
        )

    def test_dotstar(self):
        """Dot-star pattern a.*z."""
        _run_benchmark_is_match(
            "a...z at edges", "a.*z",
            re.compile(b"a.*z"), _r_dotstar, _dotstar_is_match,
            lambda sz: b'a' + b'm' * (sz - 2) + b'z',
        )

    def test_email_no_match(self):
        """Email-like pattern, no match.

        This triggers catastrophic backtracking in Python re because
        [a-z]+ matches the all-lowercase input greedily, then backtracks
        on every position looking for '@'. PythoC DFA handles this in O(n).
        """
        _run_benchmark_is_match(
            "no match (re: catastrophic backtracking)",
            "[a-z]+@[a-z]+\\.[a-z]+",
            re.compile(rb"[a-z]+@[a-z]+\.[a-z]+"), _r_email, _email_is_match,
            _build_no_match,
        )

    # --- search benchmarks ---

    def test_search_literal_no_match(self):
        """Literal search, no match."""
        _run_benchmark_search(
            "no match", "needle",
            re.compile(b"needle"), _r_needle, _needle_search,
            _build_no_match,
        )

    def test_search_literal_match_end(self):
        """Literal search, match at end."""
        _run_benchmark_search(
            "match at end", "needle",
            re.compile(b"needle"), _r_needle, _needle_search,
            lambda sz: _build_match_at_end(sz, b"needle"),
        )

    def test_search_digits_match_end(self):
        """Digit search, digits at end."""
        _run_benchmark_search(
            "match at end", "\\d+",
            re.compile(rb"\d+"), _r_digits, _digits_search,
            lambda sz: _build_match_at_end(sz, b"12345"),
        )

    def test_search_alt_match_middle(self):
        """Alternation search, match in middle."""
        _run_benchmark_search(
            "match in middle", "alpha|beta|gamma",
            re.compile(b"alpha|beta|gamma"), _r_alt, _alt_search,
            lambda sz: _build_match_at_middle(sz, b"gamma"),
        )

    def test_search_email_match_middle(self):
        """Email-like search, match in middle.

        Python re backtracks on the leading [a-z]+ prefix before the '@'.
        """
        _run_benchmark_search(
            "match in middle (re: catastrophic backtracking)",
            "[a-z]+@[a-z]+\\.[a-z]+",
            re.compile(rb"[a-z]+@[a-z]+\.[a-z]+"), _r_email, _email_search,
            lambda sz: _build_match_at_middle(sz, b"user@host.com"),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
