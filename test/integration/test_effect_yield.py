#!/usr/bin/env python3
"""
Integration tests for effect-yield unification:
for loops over compile-level yield handlers and effect-resolved yield
functions, plus `yield effect.perform(...)` inside producers.

Model: handlers are continuation transformers (compile-level yield
functions). 'for x in P(args):' is the default loop-effect handler form -
P resumes the body once per yield, binding the payload to x; 'else' is
the return clause. Handler installation/binding is spelled 'with
effect(x=impl):' (compile-time DI); 'with' never captures continuations -
a with block always runs normally, exactly once, in place.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from types import SimpleNamespace

from pythoc.decorators.compile import compile
from pythoc.builtin_entities import void, i8, i32, ptr, static
from pythoc.effect import effect
from pythoc.logger import set_raise_on_error
from pythoc.build.output_manager import flush_all_pending_outputs, clear_failed_group

set_raise_on_error(True)


# ============================================================================
# Handlers (compile-level yield functions)
# ============================================================================
@compile
def _bracket(p: ptr[i32]) -> void:
    p[0] = p[0] * 10 + 1      # prologue
    yield                      # resume the block
    p[0] = p[0] * 10 + 3      # epilogue


@compile
def _produce_two() -> i32:
    yield 10
    yield 20


@compile
def _maybe(x: i32) -> i32:
    if x > 0:
        yield x


@compile
def _pred_bind(x: i32) -> i32:
    if x > 0:
        yield x


@compile
def _pred_bind_alt(x: i32) -> i32:
    if x > 0:
        yield x * 100


effect.default(pred=SimpleNamespace(bind=_pred_bind))


# ============================================================================
# Test 1: bracket handler - prologue/body/epilogue order
# ============================================================================
@compile(suffix="with_bracket")
def with_bracket() -> i32:
    r: i32 = 0
    for _sink in _bracket(ptr(r)):
        r = r * 10 + 2
    return r


def test_with_bracket():
    assert with_bracket() == 123
    print("OK test_with_bracket passed")


# ============================================================================
# Test 2: payload binding per resume
# ============================================================================
@compile(suffix="with_as_payload")
def with_as_payload() -> i32:
    total: i32 = 0
    for v in _produce_two():
        total = total + v
    return total


def test_with_as_payload():
    assert with_as_payload() == 30
    print("OK test_with_as_payload passed")


# ============================================================================
# Test 3: short-circuit handler - body skipped when handler does not yield
# ============================================================================
@compile(suffix="with_short_circuit")
def with_short_circuit() -> i32:
    hit: i32 = 0
    for _s in _maybe(-1):
        hit = 1
    return hit


def test_with_short_circuit():
    assert with_short_circuit() == 0
    print("OK test_with_short_circuit passed")


# ============================================================================
# Test 4: for over effect-resolved yield op, with else and break
# ============================================================================
@compile(suffix="effect_for_bind")
def effect_for_bind(n: i32) -> i32:
    total: i32 = 0
    for v in effect.pred.bind(n):
        total = v
        break
    else:
        total = -1
    return total


def test_effect_for_bind():
    assert effect_for_bind(5) == 5
    assert effect_for_bind(-1) == -1
    print("OK test_effect_for_bind passed")


# ============================================================================
# Test 5: override a yield handler via the effect system
# ============================================================================
with effect(pred=SimpleNamespace(bind=_pred_bind_alt), suffix="alt"):
    @compile(suffix="alt")
    def effect_for_bind_alt(n: i32) -> i32:
        total: i32 = 0
        for v in effect.pred.bind(n):
            total = v
            break
        else:
            total = -1
        return total


def test_effect_for_override():
    assert effect_for_bind_alt(5) == 500
    assert effect_for_bind_alt(-1) == -1
    # default group unaffected
    assert effect_for_bind(5) == 5
    print("OK test_effect_for_override passed")


# ============================================================================
# Test 6: with a non-yield context expression is a clear error
# ============================================================================
def test_with_non_yield_error():
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'with_non_yield')
    try:
        @compile(suffix="with_non_yield")
        def bad() -> i32:
            x: i32 = 0
            with x:
                x = 1
            return x

        flush_all_pending_outputs()
        print("FAIL test_with_non_yield_error failed - should have raised")
    except NotImplementedError as e:
        if "with statement" in str(e):
            print(f"OK test_with_non_yield_error passed: {e}")
        else:
            print(f"FAIL test_with_non_yield_error failed - wrong error: {e}")
    finally:
        clear_failed_group(group_key)


# ============================================================================
# Step 2: yield effect.perform(...) inside producers
# ============================================================================

@compile
def _add_chain(a: i32, b: i32) -> i32:
    v1 = yield effect.perform(_maybe, a)
    v2 = yield effect.perform(_maybe, b)
    yield v1 + v2


@compile(suffix="perform_chain")
def perform_chain(a: i32, b: i32) -> i32:
    total: i32 = 0
    for v in _add_chain(a, b):
        total = v
        break
    else:
        total = -1
    return total


def test_perform_chain():
    assert perform_chain(3, 4) == 7
    assert perform_chain(-1, 4) == -1   # first perform short-circuits
    assert perform_chain(3, -1) == -1   # second perform short-circuits
    print("OK test_perform_chain passed")


# ============================================================================
# Test: statement form relays the performed value onward
# ============================================================================
@compile
def _relay(x: i32) -> i32:
    yield effect.perform(_maybe, x)


@compile(suffix="perform_stmt_form")
def perform_stmt_form(n: i32) -> i32:
    total: i32 = 0
    for v in _relay(n):
        total = v
        break
    else:
        total = -1
    return total


def test_perform_stmt_form():
    assert perform_stmt_form(5) == 5
    assert perform_stmt_form(-1) == -1
    print("OK test_perform_stmt_form passed")


# ============================================================================
# Test: perform target resolved through an effect namespace
# ============================================================================
@compile
def _chain_via_effect(a: i32, b: i32) -> i32:
    v1 = yield effect.perform(effect.pred.bind, a)
    v2 = yield effect.perform(effect.pred.bind, b)
    yield v1 + v2


@compile(suffix="perform_effect_ns")
def perform_effect_ns(a: i32, b: i32) -> i32:
    total: i32 = 0
    for v in _chain_via_effect(a, b):
        total = v
        break
    else:
        total = -1
    return total


def test_perform_effect_ns():
    assert perform_effect_ns(3, 4) == 7
    assert perform_effect_ns(3, -1) == -1
    print("OK test_perform_effect_ns passed")


# ============================================================================
# Test: err side channel across a short-circuiting perform
# ============================================================================
@compile(suffix="err_slot")
def _err_slot() -> ptr[i32]:
    s: static[i32]
    return ptr(s)


@compile
def _bind_sc(x: i32) -> i32:
    if x >= 0:
        yield x
    else:
        _err_slot()[0] = x


@compile
def _chain_sc(a: i32, b: i32) -> i32:
    v1 = yield effect.perform(_bind_sc, a)
    v2 = yield effect.perform(_bind_sc, b)
    yield v1 + v2


@compile(suffix="perform_side_channel")
def perform_side_channel(a: i32, b: i32) -> i32:
    for v in _chain_sc(a, b):
        return v
    return _err_slot()[0]


def test_perform_side_channel():
    assert perform_side_channel(3, 4) == 7
    assert perform_side_channel(-7, 4) == -7
    assert perform_side_channel(3, -9) == -9
    print("OK test_perform_side_channel passed")


# ============================================================================
# Step 3: direct-style do-notation with real Result types + errno override
# ============================================================================
from pythoc.std.result import result_wrap

R_type, R_api = result_wrap(i32, i32, name="Result_i32_i32")


@compile
def _nonneg(a: i32) -> R_type:
    if a >= 0:
        return R_api.ok(a)
    return R_api.err(a * 10)


@compile
def _sum_oks(a: i32, b: i32) -> i32:
    v1 = yield effect.perform(R_api.bind, _nonneg(a))
    v2 = yield effect.perform(R_api.bind, _nonneg(b))
    yield v1 + v2


@compile(suffix="result_do_direct")
def result_do_direct(a: i32, b: i32) -> i32:
    for v in _sum_oks(a, b):
        return v
    return -1


def test_result_do_direct():
    assert result_do_direct(3, 4) == 7
    assert result_do_direct(-1, 4) == -1   # first bind short-circuits
    assert result_do_direct(3, -1) == -1   # second bind short-circuits
    print("OK test_result_do_direct passed")


# ---------------------------------------------------------------------------
# errno side channel override: channel storage moved to Python-managed memory
# ---------------------------------------------------------------------------
class PyBufferProvider:
    """ErrnoSlotProvider whose slots live in Python-managed ctypes memory.

    Demonstrates that the err side channel is an ordinary effect impl:
    swapping it moves the channel's storage without touching bind/do.
    """

    def __init__(self):
        self._slots = {}

    def get_slot(self, err_type):
        size = err_type.get_size_bytes()
        if size in self._slots:
            return self._slots[size][1]

        import ctypes
        buf = (ctypes.c_char * size)()
        addr = ctypes.addressof(buf)

        @compile(suffix=("pybuf_slot", size))
        def _slot_fn() -> ptr[i8]:
            return ptr[i8](addr)

        self._slots[size] = (buf, _slot_fn)
        return _slot_fn

    def read_i32(self) -> int:
        import ctypes
        buf = self._slots[i32.get_size_bytes()][0]
        return ctypes.c_int32.from_address(ctypes.addressof(buf)).value


_pybuf_provider = PyBufferProvider()
with effect(errno=_pybuf_provider, suffix="pybuf"):
    R2_type, R2_api = result_wrap(i32, i32, name="Result_pybuf")


@compile
def _nonneg2(a: i32) -> R2_type:
    if a >= 0:
        return R2_api.ok(a)
    return R2_api.err(a * 10)


# The provider override still applies to direct bind usage (outside do) -
# do now owns its channel (frame-local), so the provider is only consulted
# on paths that do not install their own block binding.
with effect(errno=_pybuf_provider, suffix="pybuf"):
    @compile(suffix="pybuf_direct_bind")
    def pybuf_direct_bind(a: i32) -> i32:
        hit: i32 = -1
        for v in R2_api.bind(_nonneg2(a)):
            hit = v
            break
        return hit


@compile
def _sum_oks2(a: i32, b: i32) -> i32:
    v1 = yield effect.perform(R2_api.bind, _nonneg2(a))
    v2 = yield effect.perform(R2_api.bind, _nonneg2(b))
    yield v1 + v2


@compile(suffix="result_do_pybuf")
def result_do_pybuf(a: i32, b: i32) -> i32:
    for v in _sum_oks2(a, b):
        return v
    return -999


def test_result_do_errno_override():
    # do path: works with its own frame-local channel (provider unused)
    assert result_do_pybuf(2, 3) == 5
    assert result_do_pybuf(-3, 4) == -999
    assert result_do_pybuf(2, -7) == -999
    # direct bind path: provider override still decides the channel
    assert pybuf_direct_bind(2) == 2
    assert pybuf_direct_bind(-3) == -1
    assert _pybuf_provider.read_i32() == -30
    print("OK test_result_do_errno_override passed")


# ============================================================================
# Zero-cost side channel: closure handler capturing a frame-local channel
#
# A nested def with yield is a for-loop producer (closure-with-yield).
# Its by-ref captures make the err channel an ordinary frame local: no
# global slot, no runtime cost, reentrant by construction. The producer
# reifies the channel into a Result through the normal yield channel.
# ============================================================================
@compile
def _yield_bind_capture(x: i32) -> i32:
    base: i32 = 100
    def gen(v: i32) -> i32:
        if v > 0:
            yield v + base
    for r in gen(x):
        yield r


@compile(suffix="yield_closure_basic")
def yield_closure_basic(x: i32) -> i32:
    total: i32 = 0
    for v in _yield_bind_capture(x):
        total = v
        break
    else:
        total = -1
    return total


def test_yield_closure_basic():
    assert yield_closure_basic(5) == 105
    assert yield_closure_basic(-1) == -1
    print("OK test_yield_closure_basic passed")


@compile
def _sum_channel(a: i32, b: i32) -> R_type:
    err: i32 = 0
    err_set: i32 = 0

    def bind(x: i32) -> i32:
        if x >= 0:
            yield x
        else:
            err = x * 10
            err_set = 1

    for v1 in bind(a):
        for v2 in bind(b):
            yield R_api.ok(v1 + v2)
    if err_set == 1:
        yield R_api.err(err)


@compile(suffix="channel_unwrap")
def channel_unwrap(r: R_type) -> i32:
    match r:
        case (R_type.Ok, v):
            return v
        case (R_type.Err, e):
            return -(1000000 - e)


@compile(suffix="zero_cost_channel")
def zero_cost_channel(a: i32, b: i32) -> i32:
    for res in _sum_channel(a, b):
        return channel_unwrap(res)
    return -999


def test_zero_cost_channel():
    assert zero_cost_channel(3, 4) == 7
    # err = -3 * 10 = -30 -> -(1000000 - (-30)) = -1000030
    assert zero_cost_channel(-3, 4) == -1000030
    # err = -7 * 10 = -70 -> -(1000000 - (-70)) = -1000070
    assert zero_cost_channel(3, -7) == -1000070
    print("OK test_zero_cost_channel passed")


# ============================================================================
# Block-scoped effect binding: with effect(x=impl): in compiled code
# ============================================================================
@compile(suffix="block_binding")
def block_binding(n: i32) -> i32:
    total: i32 = 0
    # module default: bind yields x when x > 0
    for v in effect.pred.bind(n):
        total = total + v
        break
    else:
        total = -1
    with effect(pred=SimpleNamespace(bind=_pred_bind_alt)):
        # block binding wins: bind yields x * 100
        for w in effect.pred.bind(n):
            total = total + w
            break
    return total


@compile(suffix="block_binding_scope")
def block_binding_scope(n: i32) -> i32:
    total: i32 = 0
    with effect(pred=SimpleNamespace(bind=_pred_bind_alt)):
        for v in effect.pred.bind(n):
            total = v
            break
    # past the block: back to the module default
    for w in effect.pred.bind(n):
        total = total + w
        break
    return total


def test_block_binding():
    assert block_binding(5) == 505      # 5 + 500
    assert block_binding(-1) == -1      # default else, then no yield
    print("OK test_block_binding passed")


def test_block_binding_scope():
    # inside block: 500 (alt); outside: + 5 (default) - not 500 + 500
    assert block_binding_scope(5) == 505
    print("OK test_block_binding_scope passed")


def main():
    print("Effect-Yield Integration Tests (step 1)")
    print("=" * 60)

    try:
        test_with_bracket()
        test_with_as_payload()
        test_with_short_circuit()
        test_effect_for_bind()
        test_effect_for_override()
        test_with_non_yield_error()
        test_perform_chain()
        test_perform_stmt_form()
        test_perform_effect_ns()
        test_perform_side_channel()
        test_result_do_direct()
        test_result_do_errno_override()
        test_yield_closure_basic()
        test_zero_cost_channel()
        test_block_binding()
        test_block_binding_scope()

        print()
        print("=" * 60)
        print("All effect-yield tests passed! OK")
        return 0
    except Exception as e:
        print(f"\nFAIL Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
