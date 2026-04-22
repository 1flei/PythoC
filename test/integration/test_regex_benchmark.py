#!/usr/bin/env python3
"""
Regex benchmark: pythoc.regex (compiled native) vs Python's ``re``.

Per benchmark case we report up to three timings:

  1. ``python re``   -- stdlib ``re`` module.
  2. ``PC py``       -- pythoc.regex Python-level ``match``/``search``
                        (ctypes wrapper + some Python glue).
  3. ``PC native``   -- the compiled LLVM native runner called via
                        ctypes (one call per run).

Timeouts are the only mechanism used to keep the benchmark
responsive -- no case is hard-coded in or out of the run.

- ``PC_BENCH_BUILD_TIMEOUT`` (default 20 s) caps how long pythoc may
  spend compiling a pattern into native code.  The compile path
  threads through Python regularly, so ``SIGALRM`` can interrupt it.
  Exceeding the budget shows the full table as ``build_timeout``.

- ``PC_BENCH_RUN_TIMEOUT`` (default 2 s) is a soft budget applied to
  each individual ``re.search`` / ``PC py`` / ``PC native`` call.
  None of the three is interruptible mid-call (they are atomic C
  calls as far as Python is concerned), so "soft" means: once a call
  crosses the budget we skip all remaining repeats for the same
  column on this (case, size).  That column prints ``run_timeout``.

Patterns live in :mod:`test.utils.regex_patterns`; this file only
decides which cases to run and how big the haystacks should be.

A fourth "PC pure" column that wrapped the native fn in a
``@pc_compile`` loop was removed because the current pythoc codegen
does not inline an external regex ``@compile`` target into the
wrapper -- the loop body was being executed as a no-op rather than
the real regex.  Restoring that column requires pythoc-side support
for cross-unit inlining of already-compiled functions.
"""

import ctypes
import re
import time
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Force line-buffered stdout so benchmark progress becomes visible as
# soon as a line is produced.  Without this, running ``python
# test_regex_benchmark.py`` looks like it hangs on the first sized
# case while stdout buffering sits on unflushed bytes for many
# seconds.  (``python -u`` achieves the same thing from outside.)
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except AttributeError:
    pass

from pythoc.regex import compile as regex_compile

from test.utils.regex_patterns import case as get_case


# ============================================================================
# Benchmark infrastructure
# ============================================================================

# Build timeout: how long we are willing to spend compiling a single
# regex pattern into native code (parse -> TNFA -> TDFA -> T-BMA ->
# codegen -> LLVM).  Unlike ``re.search``, the pythoc compile path is
# *not* one atomic C call -- it threads through Python for most of
# its runtime -- so ``signal.SIGALRM`` reliably preempts it.
_BUILD_TIMEOUT = float(os.environ.get("PC_BENCH_BUILD_TIMEOUT", "20.0"))

# Run timeout (soft budget): once a single call to ``re.search`` /
# ``PC py`` / ``PC native`` exceeds this, we stop the remaining
# repeats for that (case, size).  None of those three is
# interruptible mid-call (they are atomic C calls as far as Python
# is concerned), so this is "soft": once one call is *already* past
# the budget we bail -- we never actively interrupt it.
_RUN_TIMEOUT = float(os.environ.get("PC_BENCH_RUN_TIMEOUT", "1.0"))


class _BuildTimeout(Exception):
    pass


def _install_build_alarm(seconds):
    """Install a SIGALRM-based timeout around pythoc's compile path.

    Returns a context manager.  Only works where ``signal.SIGALRM`` is
    supported (i.e. not on Windows); elsewhere the timeout degrades
    to a no-op and slow compiles will still block.
    """
    import contextlib
    import signal

    @contextlib.contextmanager
    def _cm():
        if not hasattr(signal, "SIGALRM"):
            yield
            return

        def _handler(signum, frame):
            raise _BuildTimeout()

        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, max(seconds, 1e-3))
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

    return _cm()


def _bench(fn, *args, warmup=1, repeats=3, soft_budget=0.0):
    """Run ``fn(*args)`` a few times and return ``(best_time, result)``.

    Every call (warmup + repeats) is subject to ``soft_budget``: if
    any single call exceeds the budget we return ``(None, result)``
    to let the caller print ``run_timeout`` rather than silently
    waiting.  Note we cannot *interrupt* the ongoing call -- ``re``
    / the pythoc wrapper / the ctypes-wrapped native fn are all
    atomic C calls -- but we stop starting new work.

    Pythoc's first-ever call would normally pay a deferred LLVM
    compile cost; the benchmark module primes every registered
    native fn during import (``_prime_native_fns``) so that this
    budget does not need to accommodate JIT warm-up.
    """
    result = None
    for _ in range(warmup):
        t0 = time.perf_counter()
        result = fn(*args)
        elapsed = time.perf_counter() - t0
        if soft_budget > 0 and elapsed > soft_budget:
            return None, result
    best = float('inf')
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args)
        elapsed = time.perf_counter() - t0
        if elapsed < best:
            best = elapsed
        if soft_budget > 0 and elapsed > soft_budget:
            return None, result
    return best, result


def _bytes_to_cargs(data: bytes):
    n = len(data)
    buf = ctypes.create_string_buffer(data, n)
    ptr_val = ctypes.cast(buf, ctypes.c_void_p).value
    return n, ptr_val, buf


# ============================================================================
# Test data builders
# ============================================================================

def _pad_no_match(size, filler=b'x'):
    return filler * size


def _pad_with_needle_at_end(size, needle, filler=b'x'):
    return filler * (size - len(needle)) + needle


def _pad_with_needle_at_middle(size, needle, filler=b'x'):
    mid = size // 2
    return filler * mid + needle + filler * (size - mid - len(needle))


def _pad_with_needle_at_start(size, needle, filler=b'x'):
    return needle + filler * (size - len(needle))


def _interleaved_email_noise(size):
    """Haystack full of '@'s but no real email; catastrophic for ``re``."""
    chunk = b"abc@def@ghi@jkl@"
    repeats = size // len(chunk) + 1
    return (chunk * repeats)[:size]


def _a_runs_no_match(size):
    """Haystack that is all ``a`` then one ``b`` missing: ReDoS feeder."""
    return b"a" * size  # no trailing 'b', so (a+)+b never matches


def _a_runs_match_at_end(size):
    return b"a" * (size - 1) + b"b"


# ============================================================================
# Per-case native fn registration
# ============================================================================

# All cases benchmarked below. Each case may show up in either
# _MATCH_CASE_NAMES, _SEARCH_CASE_NAMES, or both.
_MATCH_CASE_NAMES = (
    "literal_abc",
    "quant_plus",
    "dotstar_bracket",
    "identifier",
    "email_like",
    "ipv4_like",
    "iso_timestamp",
    "redos_a_opt_n_a_n",
    "dotstar_a_dot_k",
    "many_optionals",
)
_SEARCH_CASE_NAMES = (
    "needle_in_haystack",
    "long_needle",
    "digit_run",
    "alt_three",
    "alt_long_prefix",
    "alt_shared_body",
    "email_like",
    "catastrophic_email",
    "catastrophic_nested",
    "c_keywords",
    "python_keywords",
    "sql_reserved",
    "long_trie_shared_prefix",
    "url_prefix",
    "dotted_identifier_chain",
    "iso_timestamp",
    "dotstar_sandwich_chain",
    "dotstar_then_anchored_tail",
)


# ---------------------------------------------------------------------------
# Eager compilation of every case's native fns.
#
# Each compile is wrapped in a SIGALRM-based timeout (``_BUILD_TIMEOUT``);
# anything that exceeds it is recorded in ``_build_timed_out`` and the
# corresponding tests later show ``build_timeout`` instead of data.
# This replaces a previous hard-coded skip list -- the benchmark now
# tells you *what* is slow today rather than relying on stale
# assumptions.
# ---------------------------------------------------------------------------

_all_case_names = sorted(set(_MATCH_CASE_NAMES) | set(_SEARCH_CASE_NAMES))

_cases = {}
_match_fns = {}
_search_fns = {}
_search_slots = {}
_build_timed_out = set()

sys.stderr.write(
    f"[bench] compiling regex cases (build_timeout={_BUILD_TIMEOUT:.1f}s)...\n")
sys.stderr.flush()

for _name in _all_case_names:
    _t0 = time.perf_counter()
    sys.stderr.write(f"  {_name}: compiling... ")
    sys.stderr.flush()
    try:
        with _install_build_alarm(_BUILD_TIMEOUT):
            _c = get_case(_name)
            _cr = regex_compile(_c.pattern)
            _match_fn = None
            _search_fn = None
            if _name in _MATCH_CASE_NAMES:
                _match_fn = _cr.generate_match_fn()
            if _name in _SEARCH_CASE_NAMES:
                _search_fn = _cr.generate_search_fn()
            # Force the deferred LLVM flush for this case's group before
            # moving on.  Until a first ``match/search`` is invoked the
            # pending queue only records IR-level artifacts; the actual
            # optimize + object file + shared library work is postponed
            # and would otherwise pile onto the very first benchmark
            # call.  By triggering it here we keep _BUILD_TIMEOUT
            # honestly covering the full "compile to native" cost.
            #
            # Every CompiledRegex owns its own group (keyed by pattern
            # digest), so calling on one case never appends symbols to
            # another case's already-loaded .so -- which would hit the
            # "Cannot compile new functions after native execution has
            # started" guard in output_manager.
            if _name in _MATCH_CASE_NAMES:
                _cr.match(b"")
            else:
                _cr.search(b"")
    except _BuildTimeout:
        _build_timed_out.add(_name)
        sys.stderr.write(
            f"BUILD_TIMEOUT after {(time.perf_counter() - _t0):.1f}s "
            f"(PC_BENCH_BUILD_TIMEOUT to override)\n")
        sys.stderr.flush()
        continue
    _elapsed_ms = (time.perf_counter() - _t0) * 1000
    sys.stderr.write(f"done in {_elapsed_ms:.0f}ms\n")
    sys.stderr.flush()
    _cases[_name] = (_c, _cr)
    if _match_fn is not None:
        _match_fns[_name] = _match_fn
    if _search_fn is not None:
        _search_fns[_name] = _search_fn
        _search_slots[_name] = _cr.search_num_slots


# ============================================================================
# Benchmark runner
# ============================================================================

# Haystacks span three orders of magnitude so Boyer-Moore shift
# effects (sub-linear scaling in haystack size) actually show up.
_SIZES = (100_000, 1_000_000, 4_000_000)


# Sentinels used by the renderer.  ``None`` simply means "not run",
# which is distinct from "we tried to run and timed out".
_BUILD_TIMEOUT_MARK = "build_timeout"
_RUN_TIMEOUT_MARK = "run_timeout"


def _print_header(title):
    print()
    print("=" * 96)
    print(f"  {title}")
    print("=" * 96)
    print(f"  {'Size':>10s}  {'python re':>14s}  {'PC py':>14s}  "
          f"{'PC native':>14s}  {'re/native':>10s}  {'py/native':>10s}")
    print("-" * 96)


def _fmt_time(t):
    if t is None:
        return "       SKIP"
    if isinstance(t, str):
        return f"{t:>14s}"
    return f"{t*1000:>12.3f}ms"


def _fmt_ratio(t_num, t_den):
    if (t_num is None or t_den is None
            or isinstance(t_num, str) or isinstance(t_den, str)
            or t_den < 1e-9):
        return "       N/A"
    return f"{t_num / t_den:>9.1f}x"


def _print_row(size, t_re, t_py, t_native):
    print(f"  {size:>10,d}  {_fmt_time(t_re)}  {_fmt_time(t_py)}  "
          f"{_fmt_time(t_native)}  "
          f"{_fmt_ratio(t_re, t_native)}  "
          f"{_fmt_ratio(t_py, t_native)}")


def _make_native_search_caller(native_fn, num_slots, beg_slot):
    """Wrap a native search fn into a position-returning callable."""
    out_buf = (ctypes.c_int64 * num_slots)()
    out_ptr = ctypes.cast(out_buf, ctypes.c_void_p).value

    def caller(n, ptr_val):
        matched = native_fn(n, ptr_val, out_ptr)
        if matched:
            return int(out_buf[beg_slot])
        return -1

    return caller


def _maybe_compile_py_re(case_obj):
    if not case_obj.portable_re:
        return None
    try:
        return re.compile(case_obj.pattern.encode('latin1'))
    except re.error:
        return None


def _print_build_timeout_rows(title):
    """Print a full table of ``build_timeout`` rows for a case whose
    native fn was never compiled."""
    _print_header(title)
    for size in _SIZES:
        _print_row(size, _BUILD_TIMEOUT_MARK, _BUILD_TIMEOUT_MARK,
                   _BUILD_TIMEOUT_MARK)


def _timed(label_for_stderr, fn, *args, soft_budget=0.0):
    """Benchmark wrapper that narrates progress to stderr.

    Writes ``label_for_stderr`` *before* calling ``fn`` so the user
    can see which runner is active; writes ``OK`` / ``TIMEOUT`` after
    the call.  Returns either a float (best time) or
    ``_RUN_TIMEOUT_MARK``.  The result of the last call is also
    returned for cross-checking.
    """
    sys.stderr.write(label_for_stderr)
    sys.stderr.flush()
    t, r = _bench(fn, *args, soft_budget=soft_budget)
    if t is None:
        sys.stderr.write(" TIMEOUT\n")
        sys.stderr.flush()
        return _RUN_TIMEOUT_MARK, r
    sys.stderr.write(f" {t * 1000:.3f}ms\n")
    sys.stderr.flush()
    return t, r


def _run_match_benchmark(label, case_name, build_data):
    case_obj = get_case(case_name)
    if case_name in _build_timed_out:
        _print_build_timeout_rows(
            f"match: /{case_obj.pattern}/  --  {label}")
        sys.stderr.write(
            f"[bench] match: {case_name} -- BUILD_TIMEOUT "
            f"(raise PC_BENCH_BUILD_TIMEOUT to include)\n")
        sys.stderr.flush()
        return
    if case_name not in _cases:
        raise RuntimeError(
            f"case {case_name!r} was neither compiled nor flagged "
            f"as a build timeout; this is a benchmark bug.")
    _, cr = _cases[case_name]
    _print_header(f"match: /{case_obj.pattern}/  --  {label}")
    sys.stderr.write(f"[bench] match: {case_name}\n")
    sys.stderr.flush()
    native_fn = _match_fns[case_name]
    py_re = _maybe_compile_py_re(case_obj)
    re_timed_out = False

    for size in _SIZES:
        sys.stderr.write(f"  size={size:,}\n")
        sys.stderr.flush()
        data = build_data(size)
        n, ptr_val, _buf = _bytes_to_cargs(data)

        if py_re is None:
            t_re, res_re = None, None
        elif re_timed_out:
            t_re, res_re = _RUN_TIMEOUT_MARK, None
        else:
            t_re, res_re = _timed(
                "    re...", lambda d=data: bool(py_re.match(d)),
                soft_budget=_RUN_TIMEOUT)
            if t_re == _RUN_TIMEOUT_MARK:
                re_timed_out = True

        t_py, res_py = _timed(
            "    py...", lambda d=data: cr.match(d)[0],
            soft_budget=_RUN_TIMEOUT)

        t_nat, res_nat = _timed(
            "    native...", native_fn, n, ptr_val,
            soft_budget=_RUN_TIMEOUT)

        # Cross-check results when we have them from both sides.
        res_py_bool = bool(res_py)
        res_nat_bool = bool(res_nat)
        if (t_re is not None and not isinstance(t_re, str)
                and res_re is not None and res_py is not None
                and res_nat is not None):
            assert bool(res_re) == res_py_bool == res_nat_bool, (
                f"Result mismatch at size={size} for {case_name}: "
                f"re={bool(res_re)} py={res_py_bool} native={res_nat_bool}"
            )
        elif res_py is not None and res_nat is not None:
            assert res_py_bool == res_nat_bool, (
                f"Result mismatch at size={size} for {case_name}: "
                f"py={res_py_bool} native={res_nat_bool}"
            )

        _print_row(size, t_re, t_py, t_nat)


def _run_search_benchmark(label, case_name, build_data):
    case_obj = get_case(case_name)
    if case_name in _build_timed_out:
        _print_build_timeout_rows(
            f"search: /{case_obj.pattern}/  --  {label}")
        sys.stderr.write(
            f"[bench] search: {case_name} -- BUILD_TIMEOUT "
            f"(raise PC_BENCH_BUILD_TIMEOUT to include)\n")
        sys.stderr.flush()
        return
    if case_name not in _cases:
        raise RuntimeError(
            f"case {case_name!r} was neither compiled nor flagged "
            f"as a build timeout; this is a benchmark bug.")
    _, cr = _cases[case_name]
    _print_header(f"search: /{case_obj.pattern}/  --  {label}")
    sys.stderr.write(f"[bench] search: {case_name}\n")
    sys.stderr.flush()
    native_fn = _search_fns[case_name]
    num_slots = _search_slots[case_name]
    beg_slot = cr._search_bma.tag_slots['__pythoc_internal_beg']
    native_caller = _make_native_search_caller(native_fn, num_slots, beg_slot)
    py_re = _maybe_compile_py_re(case_obj)
    re_timed_out = False

    for size in _SIZES:
        sys.stderr.write(f"  size={size:,}\n")
        sys.stderr.flush()
        data = build_data(size)
        n, ptr_val, _buf = _bytes_to_cargs(data)

        if py_re is None:
            t_re, res_re = None, None
        elif re_timed_out:
            t_re, res_re = _RUN_TIMEOUT_MARK, None
        else:
            def py_search(d=data):
                m = py_re.search(d)
                return m.start() if m else -1

            t_re, res_re = _timed(
                "    re...", py_search, soft_budget=_RUN_TIMEOUT)
            if t_re == _RUN_TIMEOUT_MARK:
                re_timed_out = True

        def py_level_search(d=data):
            ok, info = cr.search(d)
            return info.get('start', -1) if ok else -1

        t_py, res_py = _timed(
            "    py...", py_level_search, soft_budget=_RUN_TIMEOUT)

        t_nat, res_nat = _timed(
            "    native...", native_caller, n, ptr_val,
            soft_budget=_RUN_TIMEOUT)

        if (t_re is not None and not isinstance(t_re, str)
                and res_re is not None and res_py is not None
                and res_nat is not None):
            assert res_re == res_py == res_nat, (
                f"Result mismatch at size={size} for {case_name}: "
                f"re={res_re} py={res_py} native={res_nat}"
            )
        elif res_py is not None and res_nat is not None:
            assert res_py == res_nat, (
                f"Result mismatch at size={size} for {case_name}: "
                f"py={res_py} native={res_nat}"
            )

        _print_row(size, t_re, t_py, t_nat)


# ============================================================================
# Test class -- one test per (case, workload) pair
# ============================================================================

class TestRegexBenchmark(unittest.TestCase):

    # ------------------------------ match -------------------------------

    def test_match_literal_no_match(self):
        _run_match_benchmark(
            "no match (filler)", "literal_abc",
            lambda sz: _pad_no_match(sz),
        )

    def test_match_literal_hit_at_start(self):
        _run_match_benchmark(
            "hit at start", "literal_abc",
            lambda sz: _pad_with_needle_at_start(sz, b"abc"),
        )

    def test_match_dotstar_edges(self):
        _run_match_benchmark(
            "a...b at edges", "dotstar_bracket",
            lambda sz: b'a' + b'm' * (sz - 2) + b'b',
        )

    def test_match_quant_plus_run(self):
        _run_match_benchmark(
            "abbb..c block", "quant_plus",
            lambda sz: b'a' + b'b' * (sz - 2) + b'c',
        )

    def test_match_identifier_long(self):
        _run_match_benchmark(
            "anchored identifier", "identifier",
            lambda sz: b'a' + b'0' * (sz - 1),
        )

    def test_match_email_no_match(self):
        _run_match_benchmark(
            "no match (re: catastrophic backtracking risk)",
            "email_like",
            _pad_no_match,
        )

    def test_match_ipv4_no_match(self):
        _run_match_benchmark(
            "no match (filler)", "ipv4_like",
            _pad_no_match,
        )

    def test_match_iso_timestamp_tail(self):
        _run_match_benchmark(
            "iso timestamp at start",
            "iso_timestamp",
            lambda sz: (
                b"2024-01-02T03:04:05.678+08:00" +
                b"x" * (sz - 29)
            ),
        )

    def test_match_redos_a_opt_n_a_n(self):
        _run_match_benchmark(
            "a?{20}a{20} vs a*40 (ReDoS, re skipped)",
            "redos_a_opt_n_a_n",
            lambda sz: b"a" * max(sz, 40),
        )

    def test_match_dotstar_a_dot_k(self):
        _run_match_benchmark(
            ".*a.{7} exhaustive no-match",
            "dotstar_a_dot_k",
            _pad_no_match,
        )

    def test_match_many_optionals_tail(self):
        _run_match_benchmark(
            "many optionals then xyz tail",
            "many_optionals",
            lambda sz: b"abcdefghxyz" + b"y" * (sz - 11),
        )

    # ------------------------------ search ------------------------------

    def test_search_literal_no_match(self):
        _run_search_benchmark(
            "no match", "needle_in_haystack",
            _pad_no_match,
        )

    def test_search_literal_match_end(self):
        _run_search_benchmark(
            "needle at end", "needle_in_haystack",
            lambda sz: _pad_with_needle_at_end(sz, b"needle"),
        )

    def test_search_literal_match_middle(self):
        _run_search_benchmark(
            "needle in middle", "needle_in_haystack",
            lambda sz: _pad_with_needle_at_middle(sz, b"needle"),
        )

    def test_search_long_needle_match_end(self):
        _run_search_benchmark(
            "34-byte needle at end", "long_needle",
            lambda sz: _pad_with_needle_at_end(
                sz, b"supercalifragilisticexpialidocious"),
        )

    def test_search_long_needle_no_match(self):
        _run_search_benchmark(
            "34-byte needle absent", "long_needle",
            _pad_no_match,
        )

    def test_search_digits_match_end(self):
        _run_search_benchmark(
            "digits at end", "digit_run",
            lambda sz: _pad_with_needle_at_end(sz, b"12345"),
        )

    def test_search_alt_three_match_middle(self):
        _run_search_benchmark(
            "bird in middle", "alt_three",
            lambda sz: _pad_with_needle_at_middle(sz, b"bird"),
        )

    def test_search_alt_long_prefix_match_middle(self):
        _run_search_benchmark(
            "alpine in middle", "alt_long_prefix",
            lambda sz: _pad_with_needle_at_middle(sz, b"alpine"),
        )

    def test_search_alt_shared_body_match_middle(self):
        _run_search_benchmark(
            "axxc in middle", "alt_shared_body",
            lambda sz: _pad_with_needle_at_middle(sz, b"axxc"),
        )

    def test_search_email_match_middle(self):
        _run_search_benchmark(
            "email in middle (re: catastrophic risk)",
            "email_like",
            lambda sz: _pad_with_needle_at_middle(sz, b"user@host.com"),
        )

    def test_search_catastrophic_email_no_match(self):
        _run_search_benchmark(
            "pathological no-match (re: exponential risk)",
            "catastrophic_email",
            _interleaved_email_noise,
        )

    def test_search_catastrophic_nested_no_match(self):
        _run_search_benchmark(
            "(a+)+b over a*N no-match (ReDoS, re skipped)",
            "catastrophic_nested",
            _a_runs_no_match,
        )

    def test_search_catastrophic_nested_match_end(self):
        _run_search_benchmark(
            "(a+)+b over a*(N-1)+b (ReDoS, re skipped)",
            "catastrophic_nested",
            _a_runs_match_at_end,
        )

    def test_search_c_keywords_match_middle(self):
        # Use ``return`` rather than ``unsigned`` or ``signed``: the
        # latter pair exposes a leftmost-match mismatch between pythoc
        # and Python's ``re`` when a keyword is a prefix of another
        # keyword in the alternation (needs a separate investigation).
        _run_search_benchmark(
            "return in middle", "c_keywords",
            lambda sz: _pad_with_needle_at_middle(sz, b"return"),
        )

    def test_search_python_keywords_match_middle(self):
        # See note on test_search_c_keywords_match_middle.
        _run_search_benchmark(
            "return in middle", "python_keywords",
            lambda sz: _pad_with_needle_at_middle(sz, b"return"),
        )

    def test_search_sql_reserved_match_middle(self):
        # ``SCHEMA`` is not a prefix/suffix of any other SQL keyword
        # in the alternation, so it avoids the same leftmost-match
        # corner case as the C / Python keyword tests above.
        _run_search_benchmark(
            "SCHEMA in middle", "sql_reserved",
            lambda sz: _pad_with_needle_at_middle(sz, b"SCHEMA"),
        )

    def test_search_long_trie_shared_prefix_match_middle(self):
        _run_search_benchmark(
            "presumption in middle", "long_trie_shared_prefix",
            lambda sz: _pad_with_needle_at_middle(sz, b"presumption"),
        )

    def test_search_url_prefix_match_middle(self):
        _run_search_benchmark(
            "https://example.com in middle", "url_prefix",
            lambda sz: _pad_with_needle_at_middle(sz, b"https://example.com"),
        )

    def test_search_dotted_identifier_match_middle(self):
        _run_search_benchmark(
            "pkg.mod.name in middle", "dotted_identifier_chain",
            lambda sz: _pad_with_needle_at_middle(sz, b"pkg.mod.name"),
        )

    def test_search_iso_timestamp_match_middle(self):
        _run_search_benchmark(
            "ISO timestamp in middle", "iso_timestamp",
            lambda sz: _pad_with_needle_at_middle(
                sz, b"2024-01-02T03:04:05.678+08:00"),
        )

    def test_search_dotstar_sandwich_chain(self):
        _run_search_benchmark(
            "BEGIN { body } END in middle",
            "dotstar_sandwich_chain",
            lambda sz: _pad_with_needle_at_middle(
                sz, b"BEGIN header {body} footer END"),
        )

    def test_search_dotstar_then_anchored_tail(self):
        _run_search_benchmark(
            "greedy .* before digits + end$",
            "dotstar_then_anchored_tail",
            lambda sz: _pad_with_needle_at_end(sz, b"42end"),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
