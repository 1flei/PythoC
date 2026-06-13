"""
High-strength integration tests for instantiate(), aligned with existing
yield loop test coverage.

This maps the full yield test matrix onto the explicit instantiate API:
  it: api.Iter; api.init(ptr(it))
  while api.next(ptr(it)):
      v = api.value(ptr(it))
      ... use v ...

Scenes mapped from yield tests:
  A. basic (already in test_instantiate.py)
  B. yield with while loop (already in test_instantiate.py)
  C. multi-param (already in test_instantiate_extended.py)
  D. complex control flow (if-elif-else, nested if, multi-yield per iter, dispatch)
  E. break/continue equivalents (in while over next())
  F. nested loops
  G. cross-module call inside yield fn
  H. empty / single yield
  I. conditional type yields (f64, bool)
  J. generator expressions with filter / nested range
  K. partial iteration
  L. while-else equivalents (for-else mapped to while-else)
  M. break/continue with no internal loop (sequential yields)
  N. conditional-yield break/continue
  O. multi-local update (Fibonacci)
"""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import unittest
from pythoc import compile, i32, f64, bool, ptr
from pythoc.builtin_entities.instantiate import instantiate


# =====================================================================
# D. Complex control flow (ported from test_yield_complex_control_flow)
# =====================================================================

@compile
def icf_if_else(n: i32) -> i32:
    i: i32 = 0
    while i < n:
        if i % 2 == 0:
            yield i
        else:
            yield i * 2
        i = i + 1


@compile
def icf_if_elif_else(code: i32) -> i32:
    i: i32 = 0
    while i < 5:
        if code == 1:
            yield i
        elif code == 2:
            yield i * 2
        else:
            yield i * 3
        i = i + 1


@compile
def icf_multi_yield(n: i32) -> i32:
    i: i32 = 0
    while i < n:
        yield i
        yield i + 100
        i = i + 1


@compile
def icf_nested_if(n: i32) -> i32:
    i: i32 = 0
    while i < n:
        if i > 0:
            if i < 3:
                yield i
            else:
                yield i * 2
        i = i + 1


@compile
def icf_dispatch(mode_unused: i32) -> i32:
    i: i32 = 0
    while i < 3:
        tok: i32 = i % 4
        if tok == 0:
            yield 10
        elif tok == 1:
            yield 20
        elif tok == 2:
            yield 30
        else:
            yield 40
        i = i + 1


@compile
def run_icf_if_else() -> i32:
    api = instantiate(icf_if_else(4))
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        total = total + api.value(ptr(it))
    return total  # 0 + 2 + 2 + 6 = 10


@compile
def run_icf_if_elif_else() -> i32:
    api = instantiate(icf_if_elif_else(1))
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        total = total + api.value(ptr(it))
    return total  # 0+1+2+3+4 = 10


@compile
def run_icf_multi_yield() -> i32:
    api = instantiate(icf_multi_yield(3))
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        total = total + api.value(ptr(it))
    return total  # (0+100)+(1+101)+(2+102) = 306


@compile
def run_icf_nested_if() -> i32:
    api = instantiate(icf_nested_if(5))
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        total = total + api.value(ptr(it))
    return total  # 1+2+6+8 = 17


@compile
def run_icf_dispatch() -> i32:
    api = instantiate(icf_dispatch(0))
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        total = total + api.value(ptr(it))
    return total  # 10+20+30 = 60


# =====================================================================
# E. Break/continue equivalents
# =====================================================================

@compile
def gen_0_to_n(n: i32) -> i32:
    i: i32 = 0
    while i < n:
        yield i
        i = i + 1


@compile
def gen_multi_yield(n: i32) -> i32:
    i: i32 = 0
    while i < n:
        yield i
        yield i * 10
        i = i + 1


@compile
def run_break_simple() -> i32:
    api = instantiate(gen_0_to_n(10))
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        v = api.value(ptr(it))
        if v == 5:
            break
        total = total + v
    return total  # 0+1+2+3+4 = 10


@compile
def run_break_first() -> i32:
    api = instantiate(gen_0_to_n(10))
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        break
        total = total + api.value(ptr(it))
    return total  # 0


@compile
def run_continue_simple() -> i32:
    api = instantiate(gen_0_to_n(10))
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        v = api.value(ptr(it))
        if v == 5:
            continue
        total = total + v
    return total  # 0+1+2+3+4+6+7+8+9 = 40


@compile
def run_continue_all() -> i32:
    api = instantiate(gen_0_to_n(5))
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        continue
        total = total + api.value(ptr(it))
    return total  # 0 (never accumulates)


@compile
def run_break_continue_mix() -> i32:
    api = instantiate(gen_0_to_n(10))
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        v = api.value(ptr(it))
        if v % 2 == 0:
            continue
        if v >= 7:
            break
        total = total + v
    return total  # 1+3+5 = 9


@compile
def run_multi_yield_break() -> i32:
    api = instantiate(gen_multi_yield(5))
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        v = api.value(ptr(it))
        if v >= 20:
            break
        total = total + v
    return total  # 0+0+1+10+2 = 13


@compile
def run_multi_yield_continue() -> i32:
    api = instantiate(gen_multi_yield(3))
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        v = api.value(ptr(it))
        if v >= 10:
            continue
        total = total + v
    return total  # 0+0+1+2 = 3


# =====================================================================
# F. Nested loops (ported from test_yield_break_continue nested cases)
# =====================================================================

@compile
def run_nested_break_inner() -> i32:
    api_i = instantiate(gen_0_to_n(3))
    it_i: api_i.Iter; api_i.init(ptr(it_i))
    total: i32 = 0
    while api_i.next(ptr(it_i)):
        api_j = instantiate(gen_0_to_n(5))
        it_j: api_j.Iter; api_j.init(ptr(it_j))
        while api_j.next(ptr(it_j)):
            v = api_j.value(ptr(it_j))
            if v == 2:
                break
            total = total + v
        total = total + 100
    return total  # (0+1+100)*3 = 303


@compile
def run_nested_continue_inner() -> i32:
    api_i = instantiate(gen_0_to_n(3))
    it_i: api_i.Iter; api_i.init(ptr(it_i))
    total: i32 = 0
    while api_i.next(ptr(it_i)):
        api_j = instantiate(gen_0_to_n(5))
        it_j: api_j.Iter; api_j.init(ptr(it_j))
        while api_j.next(ptr(it_j)):
            v = api_j.value(ptr(it_j))
            if v == 2:
                continue
            total = total + v
    return total  # (0+1+3+4)*3 = 24


# =====================================================================
# G. Cross-module call inside yield fn
# =====================================================================

from test.integration.test_yield_cross_module_helper import external_add


@compile
def gen_with_external(n: i32) -> i32:
    i: i32 = 0
    while i < n:
        yield external_add(i, 10)
        i = i + 1


@compile
def run_cross_module() -> i32:
    api = instantiate(gen_with_external(3))
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        total = total + api.value(ptr(it))
    return total  # 10+11+12 = 33


# =====================================================================
# H. Empty / single yield
# =====================================================================

@compile
def gen_empty() -> i32:
    i: i32 = 0
    while i < 0:
        yield i
        i = i + 1


@compile
def run_empty() -> i32:
    api = instantiate(gen_empty())
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        total = total + api.value(ptr(it))
    return total  # 0


@compile
def gen_single() -> i32:
    yield 42


@compile
def run_single() -> i32:
    api = instantiate(gen_single())
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        total = total + api.value(ptr(it))
    return total  # 42


# =====================================================================
# I. Conditional / type variants (f64, bool) – extended
# =====================================================================

@compile
def gen_f64_seq() -> f64:
    yield 1.5
    yield 2.5
    yield 3.5


@compile
def run_f64_sum() -> i32:
    api = instantiate(gen_f64_seq())
    it: api.Iter; api.init(ptr(it))
    total: f64 = 0.0
    while api.next(ptr(it)):
        total = total + api.value(ptr(it))
    return i32(total)  # 1.5+2.5+3.5 = 7.5 -> 7


@compile
def gen_bool_seq() -> bool:
    yield True
    yield False
    yield True


@compile
def run_bool_encode() -> i32:
    api = instantiate(gen_bool_seq())
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        if api.value(ptr(it)):
            total = total + 1
    return total  # 2


# =====================================================================
# J. Generator expressions (already in test_instantiate_extended.py,
#    plus more complex cases)
# =====================================================================

@compile
def run_genexpr_filter_squares() -> i32:
    api = instantiate(i32(x * x) for x in range(10) if x % 2 == 1)
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        total = total + api.value(ptr(it))
    return total  # 1+9+25+49+81 = 165


@compile
def run_genexpr_range_filter() -> i32:
    api = instantiate(i32(x) for x in range(20) if x < 5)
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        total = total + api.value(ptr(it))
    return total  # 0+1+2+3+4 = 10


@compile
def run_genexpr_nested() -> i32:
    api = instantiate(i32(x + y) for x in range(3) for y in range(2))
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        total = total + api.value(ptr(it))
    return total  # (0+0)+(0+1)+(1+0)+(1+1)+(2+0)+(2+1) = 9


# =====================================================================
# K. Partial iteration (caller stops early)
# =====================================================================

@compile
def run_partial_3_of_10() -> i32:
    api = instantiate(gen_0_to_n(10))
    it: api.Iter; api.init(ptr(it))
    api.next(ptr(it))
    api.next(ptr(it))
    api.next(ptr(it))
    return api.value(ptr(it))  # third value = 2


@compile
def run_partial_no_value_after_done() -> i32:
    api = instantiate(gen_single())
    it: api.Iter; api.init(ptr(it))
    api.next(ptr(it))      # consume the only value
    r: i32 = api.value(ptr(it))
    api.next(ptr(it))      # should return 0 (done)
    return r  # 42


# =====================================================================
# L. while-else equivalents (mapped from test_yield_break_continue for-else)
# =====================================================================
#
# The instantiate manual API uses ``while sealed.next(ptr(it)):`` which
# supports ``else`` identically to a ``for`` loop.

@compile
def run_while_else_normal() -> i32:
    """Else executes when loop completes normally (no break)."""
    api = instantiate(gen_0_to_n(3))
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        total = total + api.value(ptr(it))
    else:
        total = total + 100
    return total  # 0+1+2+100 = 103


@compile
def run_while_else_break() -> i32:
    """Else skipped when break fires."""
    api = instantiate(gen_0_to_n(5))
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        v = api.value(ptr(it))
        if v == 3:
            break
        total = total + v
    else:
        total = total + 100  # should NOT execute
    return total  # 0+1+2 = 3


@compile
def run_while_else_empty() -> i32:
    """Else executes for empty iterator."""
    api = instantiate(gen_empty())
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        total = total + api.value(ptr(it))
    else:
        total = total + 100
    return total  # 100


# =====================================================================
# M. Break/continue with NO internal loop (sequential yields)
# =====================================================================
# Ported from test_yield_break_no_loop.py.  The yield function body is a
# flat sequence of ``yield`` statements without a containing ``while``.
# This exercises different state-machine entry/exit paths.

@compile
def gen_three_seq() -> i32:
    """Sequential yields without a loop."""
    yield 1
    yield 2
    yield 3


@compile
def run_seq_break() -> i32:
    """Break in sequential-yield iterator."""
    api = instantiate(gen_three_seq())
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        v = api.value(ptr(it))
        if v == 2:
            break
        total = total + v
    return total  # 1


@compile
def run_seq_continue() -> i32:
    """Continue in sequential-yield iterator."""
    api = instantiate(gen_three_seq())
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        v = api.value(ptr(it))
        if v == 2:
            continue
        total = total + v
    return total  # 1+3 = 4


@compile
def run_seq_else_no_break() -> i32:
    """While-else with sequential yields, no break."""
    api = instantiate(gen_three_seq())
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        total = total + api.value(ptr(it))
    else:
        total = total + 100
    return total  # 1+2+3+100 = 106


@compile
def run_seq_else_break() -> i32:
    """While-else with sequential yields, break skips else."""
    api = instantiate(gen_three_seq())
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        v = api.value(ptr(it))
        if v == 2:
            break
        total = total + v
    else:
        total = total + 100  # should NOT execute
    return total  # 1


# =====================================================================
# N. Conditional-yield break/continue
# =====================================================================
# Ported from test_yield_break_continue conditional_yield cases.
# test_conditional_yield_break / test_conditional_yield_continue.

@compile
def gen_conditional_yield(n: i32) -> i32:
    """Yield values conditionally: even i -> i, odd i -> i*100."""
    i: i32 = 0
    while i < n:
        if i % 2 == 0:
            yield i
        else:
            yield i * 100
        i = i + 1


@compile
def run_cond_yield_break() -> i32:
    """Break when yield value >= 300."""
    api = instantiate(gen_conditional_yield(6))
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        v = api.value(ptr(it))
        if v >= 300:
            break
        total = total + v
    return total  # 0 + 100 + 2 = 102


@compile
def run_cond_yield_continue() -> i32:
    """Continue when yield value >= 100."""
    api = instantiate(gen_conditional_yield(6))
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        v = api.value(ptr(it))
        if v >= 100:
            continue
        total = total + v
    return total  # 0 + 2 + 4 = 6


# =====================================================================
# O. Multi-local update (Fibonacci)
# =====================================================================
# Ported from test_yield.py test_fibonacci.  Exercises correctness of
# multiple locals promoted to state struct and updated across yield points.

@compile
def gen_fib(limit: i32) -> i32:
    """Fibonacci numbers less than limit."""
    a: i32 = 0
    b: i32 = 1
    while a < limit:
        yield a
        new_a: i32 = b
        new_b: i32 = a + b
        a = new_a
        b = new_b


@compile
def run_fib_sum() -> i32:
    api = instantiate(gen_fib(100))
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        total = total + api.value(ptr(it))
    return total  # 0+1+1+2+3+5+8+13+21+34+55+89 = 232


# =====================================================================
# Test class
# =====================================================================

class TestInstantiateStrength(unittest.TestCase):
    """High-strength tests aligned with existing yield loop coverage."""

    # ---- D. Complex control flow ----
    def test_icf_if_else(self):
        self.assertEqual(run_icf_if_else(), 10)

    def test_icf_if_elif_else(self):
        self.assertEqual(run_icf_if_elif_else(), 10)

    def test_icf_multi_yield(self):
        self.assertEqual(run_icf_multi_yield(), 306)

    def test_icf_nested_if(self):
        self.assertEqual(run_icf_nested_if(), 17)

    def test_icf_dispatch(self):
        self.assertEqual(run_icf_dispatch(), 60)

    # ---- E. Break/continue equivalents ----
    def test_break_simple(self):
        self.assertEqual(run_break_simple(), 10)

    def test_break_first(self):
        self.assertEqual(run_break_first(), 0)

    def test_continue_simple(self):
        self.assertEqual(run_continue_simple(), 40)

    def test_continue_all(self):
        self.assertEqual(run_continue_all(), 0)

    def test_break_continue_mix(self):
        self.assertEqual(run_break_continue_mix(), 9)

    def test_multi_yield_break(self):
        self.assertEqual(run_multi_yield_break(), 13)

    def test_multi_yield_continue(self):
        self.assertEqual(run_multi_yield_continue(), 3)

    # ---- F. Nested loops ----
    def test_nested_break_inner(self):
        self.assertEqual(run_nested_break_inner(), 303)

    def test_nested_continue_inner(self):
        self.assertEqual(run_nested_continue_inner(), 24)

    # ---- G. Cross-module ----
    def test_cross_module(self):
        self.assertEqual(run_cross_module(), 33)

    # ---- H. Empty / single ----
    def test_empty(self):
        self.assertEqual(run_empty(), 0)

    def test_single(self):
        self.assertEqual(run_single(), 42)

    # ---- I. Type variants ----
    def test_f64_sum(self):
        self.assertEqual(run_f64_sum(), 7)

    def test_bool_encode(self):
        self.assertEqual(run_bool_encode(), 2)

    # ---- J. Generator expressions ----
    def test_genexpr_filter_squares(self):
        self.assertEqual(run_genexpr_filter_squares(), 165)

    def test_genexpr_range_filter(self):
        self.assertEqual(run_genexpr_range_filter(), 10)

    def test_genexpr_nested(self):
        self.assertEqual(run_genexpr_nested(), 9)

    # ---- K. Partial iteration ----
    def test_partial_3_of_10(self):
        self.assertEqual(run_partial_3_of_10(), 2)

    def test_partial_no_value_after_done(self):
        self.assertEqual(run_partial_no_value_after_done(), 42)

    # ---- L. while-else equivalents ----
    def test_while_else_normal(self):
        self.assertEqual(run_while_else_normal(), 103)

    def test_while_else_break(self):
        self.assertEqual(run_while_else_break(), 3)

    def test_while_else_empty(self):
        self.assertEqual(run_while_else_empty(), 100)

    # ---- M. Break/continue with no internal loop ----
    def test_seq_break(self):
        self.assertEqual(run_seq_break(), 1)

    def test_seq_continue(self):
        self.assertEqual(run_seq_continue(), 4)

    def test_seq_else_no_break(self):
        self.assertEqual(run_seq_else_no_break(), 106)

    def test_seq_else_break(self):
        self.assertEqual(run_seq_else_break(), 1)

    # ---- N. Conditional-yield break/continue ----
    def test_cond_yield_break(self):
        self.assertEqual(run_cond_yield_break(), 102)

    def test_cond_yield_continue(self):
        self.assertEqual(run_cond_yield_continue(), 6)

    # ---- O. Multi-local update (Fibonacci) ----
    def test_fib_sum(self):
        self.assertEqual(run_fib_sum(), 232)


if __name__ == '__main__':
    unittest.main()
