#!/usr/bin/env python3
"""
Integration test for the yield-inline linear-argument ownership fix
(defer_linear_transfer on yield placeholders).

BUG (before the fix):
    Any generator taking a LINEAR argument failed CFG checking with
    "Linear token ... already consumed": ownership was transferred twice --
    once at the for-iter call node (handle_call on the yield placeholder) and
    once again inside the inlined param-binding template (prf_inline_N =
    move(arg)). Non-linear arguments were unaffected, which is why the bug hid
    in the yield suite for so long.

FIX:
    Both yield placeholders (module-level and closure) now set
    defer_linear_transfer = True, so the call site transfers NOTHING and
    ownership moves exactly once -- inside the param-binding move().

WHAT THIS TEST COVERS:
    - Full-iteration caller: generator defers consume(prf); proof consumed
      exactly once at caller function exit. Witness flag = 1 (read by an
      outer function, since return values evaluate before defers run).
    - While-loop generator variant (state machine + yields in a loop).
    - Break-path caller: defer still fires when the loop is abandoned early.
    - Failure scenario 1: same proof reused in a SECOND yield loop is
      REJECTED at compile time ("already consumed").
    - Failure scenario 2: proof consumed twice inside the generator body is
      REJECTED at compile time.
    - Failure scenario 3 (regression shape of the original bug): using the
      caller's proof again AFTER passing it to a generator is REJECTED --
      the for-loop param binding move() owns it from that point on. Pre-fix,
      the token died at the call node and even the FIRST use failed; post-fix
      the first use compiles and only genuine double-ownership is rejected.
    - Failure scenario 4: explicit consume at the END of the generator body
      is NOT a sound ownership strategy -- whether it runs depends on the
      caller's loop structure. A break-exit caller leaks the token and the
      compile is REJECTED ("not consumed before function exit"). defer is
      the only single-point finalizer; body-consume imposes an invisible
      "iterate to exhaustion" obligation on every caller.

Run: python3 test/integration/test_yield_linear_arg.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import unittest

from pythoc.decorators.compile import compile
from pythoc.builtin_entities import void, i32, linear, consume, ptr
from pythoc import defer
from pythoc.build.output_manager import flush_all_pending_outputs

from test.utils.test_utils import DeferredTestCase, expect_error


# =============================================================================
# Shared @compile helpers (suffixes avoid cross-test-file symbol clashes)
# =============================================================================

# Consumption witness: consume() plus an observable side effect written
# through a pointer. Return-before-defer semantics mean a function cannot
# observe its own defers, so the witness must be read by an OUTER function.
@compile(suffix="yl_consume_witness")
def consume_witness(p: ptr[i32], t: linear) -> void:
    consume(t)
    p[0] = p[0] + 1


@compile(suffix="yl_gen_defer")
def gen_defer(t: linear, p: ptr[i32]) -> i32:
    """Generator that releases its linear proof via defer."""
    defer(consume_witness, p, t)
    yield 1
    yield 2


@compile(suffix="yl_gen_defer_loop")
def gen_defer_loop(t: linear, p: ptr[i32], n: i32) -> i32:
    """Generator with a while-loop + yields, proof released via defer."""
    defer(consume_witness, p, t)
    i: i32 = 0
    while i < n:
        yield i
        i = i + 1


@compile(suffix="yl_inner_full")
def inner_full(p: ptr[i32]) -> i32:
    t = linear()
    total: i32 = 0
    for v in gen_defer(t, p):
        total = total + v
    return total


@compile(suffix="yl_caller_full_witness")
def caller_full_witness() -> i32:
    """Full iteration: defer fires at inner exit; witness flag must be 1."""
    flag: i32 = 0
    got: i32 = inner_full(ptr(flag))
    return got + flag


@compile(suffix="yl_inner_break")
def inner_break(p: ptr[i32]) -> i32:
    t = linear()
    total: i32 = 0
    for v in gen_defer(t, p):
        total = total + v
        break
    return total


@compile(suffix="yl_caller_break_witness")
def caller_break_witness() -> i32:
    """Break abandons the loop; defer must still fire (flag must be 1)."""
    flag: i32 = 0
    got: i32 = inner_break(ptr(flag))
    return got + flag


@compile(suffix="yl_inner_while")
def inner_while(p: ptr[i32]) -> i32:
    t = linear()
    total: i32 = 0
    for v in gen_defer_loop(t, p, 4):
        total = total + v
    return total


@compile(suffix="yl_caller_while_witness")
def caller_while_witness() -> i32:
    """While-loop generator: defer fires at inner exit (flag must be 1)."""
    flag: i32 = 0
    got: i32 = inner_while(ptr(flag))
    return got + flag


# =============================================================================
# Failure scenarios (must be REJECTED at compile time)
# =============================================================================

# FAILURE 1: reuse of the same proof in a second yield loop. The first
# param-binding move() consumed the token; the second loop's move() must
# fail. (Pre-fix code rejected even the FIRST use; now the first use
# compiles and only genuine double-ownership is rejected.)
@expect_error(["consumed"], suffix="yl_two_loops")
def run_error_second_loop():
    @compile(suffix="yl_two_loops")
    def two_loops() -> i32:
        t = linear()
        total: i32 = 0
        for v in gen_defer(t, ptr(total)):
            total = total + v
        for w in gen_defer(t, ptr(total)):
            total = total + w
        return total


# FAILURE 2: consume twice inside the generator body. The generator body is
# checked when the CALLER is compiled (the body is spliced in at inline
# time), so the error-test caller is what triggers the CFG check.
@expect_error(["consumed"], suffix="yl_gen_double")
def run_error_double_consume_in_body():
    @compile(suffix="yl_gen_double")
    def gen_double(t: linear) -> i32:
        yield 1
        consume(t)
        consume(t)  # ERROR: second consume

    @compile(suffix="yl_gen_double_caller")
    def gen_double_caller() -> i32:
        t = linear()
        total: i32 = 0
        for v in gen_double(t):
            total = total + v
        return total


# FAILURE 3 (regression shape of the original bug): the caller keeps using
# the proof after passing it to the generator. The param-binding move() is
# the single owner from that point; any later use must be rejected. Pre-fix
# the token died at the call node instead -- same error class, wrong point;
# this pins the post-fix ownership boundary.
@expect_error(["consumed"], suffix="yl_use_after_gen")
def run_error_use_after_gen():
    @compile(suffix="yl_use_after_gen")
    def use_after_gen() -> i32:
        t = linear()
        total: i32 = 0
        for v in gen_defer(t, ptr(total)):
            total = total + v
        consume(t)  # ERROR: proof already owned by the generator's binding
        return total


# FAILURE 4: explicit consume at the END of the generator body is not a
# sound ownership strategy -- whether the consume runs depends on how the
# caller drives the loop. A break-exit caller never reaches the body-end
# consume, so the token leaks and the compile is rejected. This pins the
# design decision: defer is the single-point finalizer; body-end consume
# silently obliges every caller to iterate to exhaustion.
@expect_error(["not consumed"], suffix="yl_body_consume_break")
def run_error_body_consume_break():
    @compile(suffix="yl_body_consume")
    def gen_body_consume(t: linear) -> i32:
        yield 1
        yield 2
        consume(t)

    @compile(suffix="yl_body_consume_break_caller")
    def body_consume_break_caller() -> i32:
        t = linear()
        total: i32 = 0
        for v in gen_body_consume(t):
            total = total + v
            break  # ERROR: body-end consume never runs; token leaks
        return total


# =============================================================================
# Test class
# =============================================================================

class TestYieldLinearArg(DeferredTestCase):
    """yield generator + linear argument ownership"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        flush_all_pending_outputs()

    # -- valid cases ---------------------------------------------------------

    def test_full_iteration_defer_fires_once(self):
        """Full iteration: defer(consume) fires exactly once at inner exit."""
        # inner returns 1+2=3; defer adds flag=1 -> 4
        self.assertEqual(caller_full_witness(), 4)

    def test_break_path_defer_still_fires(self):
        """Break abandons the loop; defer must still release the proof."""
        # inner returns 1 (first yield only, then break); flag=1 -> 2
        self.assertEqual(caller_break_witness(), 2)

    def test_while_generator_defer(self):
        """While-loop state machine generator: defer fires at inner exit."""
        # yields 0+1+2+3=6; flag=1 -> 7
        self.assertEqual(caller_while_witness(), 7)

    # -- failure cases -------------------------------------------------------

    def test_error_second_loop_rejected(self):
        passed, msg = run_error_second_loop()
        self.assertTrue(passed, msg)

    def test_error_double_consume_in_body_rejected(self):
        passed, msg = run_error_double_consume_in_body()
        self.assertTrue(passed, msg)

    def test_error_use_after_gen_rejected(self):
        passed, msg = run_error_use_after_gen()
        self.assertTrue(passed, msg)

    def test_error_body_consume_break_rejected(self):
        passed, msg = run_error_body_consume_break()
        self.assertTrue(passed, msg)


if __name__ == '__main__':
    unittest.main()
