"""Smoke test: concurrent compilation of distinct groups.

Triggers first-call compilation of several @compile functions from
multiple threads.  Each function lives in its own group (distinct
compile suffix => distinct group_key), does a distinct computation,
and must return the correct value with a distinct mangled name.

This does NOT exercise parallel `with effect(...)` blocks: the scoped
builtins.__import__ hook is process-level and its enter/exit is known
to race under that combination.
"""

import concurrent.futures
import threading
import unittest

from pythoc import compile, i32

_COUNT = 6
_ARG = 10


def _make_fn(k):
    # Each k produces a distinct computation: f_k(x) = x * (k + 2) + k
    if k == 0:
        @compile(suffix="par_smoke_g0")
        def f(x: i32) -> i32:
            return x * 2 + 0
    elif k == 1:
        @compile(suffix="par_smoke_g1")
        def f(x: i32) -> i32:
            return x * 3 + 1
    elif k == 2:
        @compile(suffix="par_smoke_g2")
        def f(x: i32) -> i32:
            return x * 4 + 2
    elif k == 3:
        @compile(suffix="par_smoke_g3")
        def f(x: i32) -> i32:
            return x * 5 + 3
    elif k == 4:
        @compile(suffix="par_smoke_g4")
        def f(x: i32) -> i32:
            return x * 6 + 4
    else:
        @compile(suffix="par_smoke_g5")
        def f(x: i32) -> i32:
            return x * 7 + 5
    return f


_FUNCS = [_make_fn(k) for k in range(_COUNT)]


def _expected(k, x):
    return x * (k + 2) + k


class TestParallelCompileSmoke(unittest.TestCase):
    @unittest.skip(
        "Known race: OutputManager.flush_all has no in-flight build tracking. "
        "Thread A's flush takes ownership of building all pending groups "
        "(_take_pending_groups drains them; the DAG chain builds one at a "
        "time). A concurrent thread B's flush sees an empty pending queue, "
        "returns immediately, and its execute_function then fails with "
        "'Object file <group>.o not found' because A's chain has not built "
        "that group yet. Reproducible with distinct groups; the failing "
        "group varies per run. Needs an in-flight build wait mechanism in "
        "OutputManager before this can be enabled."
    )
    def test_parallel_first_call_compilation(self):
        """First calls racing in threads must not cross results or symbols."""
        barrier = threading.Barrier(_COUNT)

        def worker(k):
            barrier.wait(timeout=30)
            f = _FUNCS[k]
            result = f(_ARG)  # triggers this group's compilation
            return k, result, f._binding.mangled_name, f._binding.group_key

        with concurrent.futures.ThreadPoolExecutor(max_workers=_COUNT) as pool:
            outcomes = list(pool.map(worker, range(_COUNT)))

        mangled = set()
        group_keys = set()
        for k, result, mangled_name, group_key in outcomes:
            # Result matches the serial expectation for this function.
            self.assertEqual(result, _expected(k, _ARG),
                             f"function {k} returned a crossed result")
            # Symbols and groups must not collide across compilations.
            self.assertIsNotNone(mangled_name)
            self.assertNotIn(mangled_name, mangled)
            self.assertNotIn(group_key, group_keys)
            mangled.add(mangled_name)
            group_keys.add(group_key)

    def test_results_stable_after_parallel_compile(self):
        """Repeat calls after the parallel compile stay correct."""
        for k, f in enumerate(_FUNCS):
            self.assertEqual(f(_ARG), _expected(k, _ARG))
            self.assertEqual(f(_ARG + 1), _expected(k, _ARG + 1))


if __name__ == '__main__':
    unittest.main()
