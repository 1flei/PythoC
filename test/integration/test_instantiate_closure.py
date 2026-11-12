"""Integration tests for materialized closures.

Runtime captures and default values live in State. Calls preserve the source
parameter binding rules through the shared closure and call machinery.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from pythoc import (
    CompileSession, array, bool, compile, consume, defer, f64, i32, i64, linear, move, ptr,
    struct, union, void,
)
from pythoc.builtin_entities.instantiate import instantiate
from pythoc.logger import logger


# =====================================================================
# Compile-time Python constants used for capture tests
# =====================================================================
CAP_10 = 10
CAP_3 = 3
CAP_7 = 7
CAP_100 = 100
SCALE_F64 = 1.5
THRESHOLD = 5
FLAG = 1
LIMIT = 4
MAGIC = 42
VAL_99 = 99


# =====================================================================
# A. Simple single capture (i32 constant)
# =====================================================================

@compile
def run_closure_single_capture() -> i32:
    def closure_sum(x: i32) -> i32:
        return x + CAP_10
    api = instantiate(closure_sum)
    o = api.init()
    return api.call(ptr(o), i32(5))


# =====================================================================
# B. Multiple captures
# =====================================================================

@compile
def run_closure_multi_capture() -> i32:
    def closure_combined(x: i32) -> i32:
        return CAP_3 * x + CAP_7
    api = instantiate(closure_combined)
    o = api.init()
    return api.call(ptr(o), i32(3))


# =====================================================================
# C. No capture (pure closure)
# =====================================================================

@compile
def run_closure_no_capture() -> i32:
    def closure_pure(x: i32) -> i32:
        return x * 2
    api = instantiate(closure_pure)
    o = api.init()
    return api.call(ptr(o), i32(7))


# =====================================================================
# D. Multi-parameter closure
# =====================================================================

@compile
def run_closure_multi_param() -> i32:
    def closure_multi(a: i32, b: i32) -> i32:
        return CAP_100 + a + b
    api = instantiate(closure_multi)
    o = api.init()
    return api.call(ptr(o), i32(1), i32(2))


# =====================================================================
# E. f64 return type
# =====================================================================

@compile
def run_closure_f64() -> i32:
    def closure_scaled(x: i32) -> f64:
        return f64(x) * SCALE_F64
    api = instantiate(closure_scaled)
    o = api.init()
    return i32(api.call(ptr(o), i32(4)))


# =====================================================================
# F. bool return type
# =====================================================================

@compile
def run_closure_bool() -> i32:
    def closure_gt(x: i32) -> bool:
        return x > THRESHOLD
    api = instantiate(closure_gt)
    o = api.init()
    total: i32 = 0
    if api.call(ptr(o), i32(10)):
        total = total + 1
    if not api.call(ptr(o), i32(3)):
        total = total + 1
    return total


# =====================================================================
# G. If-else multi-return paths
# =====================================================================

@compile
def run_closure_if_else() -> i32:
    def closure_dispatch(x: i32) -> i32:
        if FLAG == 1:
            return x * 10
        else:
            return x + 10
    api = instantiate(closure_dispatch)
    o = api.init()
    return api.call(ptr(o), i32(6))


# =====================================================================
# H. Nested if inside while with return
# =====================================================================

@compile
def run_closure_nested_control() -> i32:
    def closure_nested(n: i32) -> i32:
        i: i32 = 0
        while i < n:
            if i % 2 == 0:
                return i * 100
            i = i + 1
        return LIMIT
    api = instantiate(closure_nested)
    o = api.init()
    return api.call(ptr(o), i32(5))


# =====================================================================
# I. Multiple sequential returns (first wins)
# =====================================================================

@compile
def run_closure_sequential() -> i32:
    def closure_seq(x: i32) -> i32:
        return MAGIC + x
    api = instantiate(closure_seq)
    o = api.init()
    return api.call(ptr(o), i32(8))


# =====================================================================
# J. Idempotent repeated calls
# =====================================================================

@compile
def run_closure_repeated() -> i32:
    def closure_pure(x: i32) -> i32:
        return x * 3
    api = instantiate(closure_pure)
    o = api.init()
    a: i32 = api.call(ptr(o), i32(2))
    b: i32 = api.call(ptr(o), i32(3))
    return a + b


# =====================================================================
# K. Closure with no-arg returning captured constant
# =====================================================================

@compile
def run_closure_no_arg() -> i32:
    def closure_const() -> i32:
        return VAL_99
    api = instantiate(closure_const)
    o = api.init()
    return api.call(ptr(o))


# =====================================================================
# L. Local variable mutated across calls
# =====================================================================

@compile
def run_closure_local_mutation() -> i32:
    def closure_accum(x: i32) -> i32:
        total: i32 = 0
        total = total + x
        return total
    api = instantiate(closure_accum)
    o = api.init()
    a: i32 = api.call(ptr(o), i32(10))
    b: i32 = api.call(ptr(o), i32(20))
    return a + b


# =====================================================================
# M. Runtime capture of a local i32 variable
# =====================================================================

@compile
def run_closure_runtime_capture() -> i32:
    i: i32 = 10
    def closure_add(x: i32) -> i32:
        return x + i
    api = instantiate(closure_add)
    o = api.init()
    return api.call(ptr(o), i32(5))


# =====================================================================
# N. Loop variable runtime capture (the primary use-case)
# =====================================================================

@compile
def run_closure_loop_capture() -> i32:
    total: i32 = 0
    i: i32 = 0
    while i < 3:
        def closure_add(x: i32) -> i32:
            return x + i
        api = instantiate(closure_add)
        o = api.init()
        total = total + api.call(ptr(o), i32(1))
        i = i + 1
    return total


# =====================================================================
# O. Multiple runtime captures
# =====================================================================

@compile
def run_closure_multi_runtime_capture() -> i32:
    a: i32 = 3
    b: i32 = 4
    def closure_sum(x: i32) -> i32:
        return x + a + b
    api = instantiate(closure_sum)
    o = api.init()
    return api.call(ptr(o), i32(2))


# =====================================================================
# P. Mixed compile-time and runtime capture
# =====================================================================

@compile
def run_closure_mixed_capture() -> i32:
    runtime_val: i32 = 7
    def closure_mixed(x: i32) -> i32:
        return x + CAP_10 + runtime_val
    api = instantiate(closure_mixed)
    o = api.init()
    return api.call(ptr(o), i32(2))


# =====================================================================
# Q. Nested closure inside instantiated closure
# =====================================================================

@compile
def run_closure_nested_closure() -> i32:
    offset: i32 = 10
    def closure_outer(x: i32) -> i32:
        local: i32 = 3
        def closure_inner(y: i32) -> i32:
            return y + local + offset
        return closure_inner(x)
    api = instantiate(closure_outer)
    o = api.init()
    return api.call(ptr(o), i32(2))


# =====================================================================
# R. Runtime capture of composite and pointer values
# =====================================================================

@compile
def run_closure_struct_capture() -> i32:
    pair: struct[i32, i32] = (3, 4)
    def closure_struct(x: i32) -> i32:
        return x + pair[0] + pair[1]
    api = instantiate(closure_struct)
    o = api.init()
    return api.call(ptr(o), i32(5))


@compile
def run_closure_union_capture() -> i32:
    data: union[i32, f64] = union[i32, f64]()
    data[0] = 11
    def closure_union(x: i32) -> i32:
        return x + data[0]
    api = instantiate(closure_union)
    o = api.init()
    return api.call(ptr(o), i32(6))


@compile
def run_closure_ptr_capture() -> i32:
    value: i32 = 41
    p: ptr[i32] = ptr(value)
    def closure_ptr(x: i32) -> i32:
        return x + p[0]
    api = instantiate(closure_ptr)
    o = api.init()
    return api.call(ptr(o), i32(1))


@compile
def run_closure_array_capture() -> i32:
    values: array[i32, 3] = array[i32, 3]()
    values[0] = 2
    values[1] = 4
    values[2] = 8
    def closure_array(x: i32) -> i32:
        return x + values[0] + values[1] + values[2]
    api = instantiate(closure_array)
    o = api.init()
    return api.call(ptr(o), i32(1))


# =====================================================================
# Test classes
# =====================================================================

class TestClosureSingleCapture(unittest.TestCase):
    def test_all(self):
        # 5 + 10 = 15
        self.assertEqual(run_closure_single_capture(), 15)


class TestClosureMultiCapture(unittest.TestCase):
    def test_all(self):
        # 3 * 3 + 7 = 16
        self.assertEqual(run_closure_multi_capture(), 16)


class TestClosureNoCapture(unittest.TestCase):
    def test_all(self):
        # 7 * 2 = 14
        self.assertEqual(run_closure_no_capture(), 14)


class TestClosureMultiParam(unittest.TestCase):
    def test_all(self):
        # 100 + 1 + 2 = 103
        self.assertEqual(run_closure_multi_param(), 103)


class TestClosureF64(unittest.TestCase):
    def test_all(self):
        # f64(4) * 1.5 = 6.0 -> i32 = 6
        self.assertEqual(run_closure_f64(), 6)


class TestClosureBool(unittest.TestCase):
    def test_all(self):
        # 10 > 5 is True -> +1; 3 > 5 is False -> +1; total = 2
        self.assertEqual(run_closure_bool(), 2)


class TestClosureIfElse(unittest.TestCase):
    def test_all(self):
        # FLAG==1 so x*10; 6*10 = 60
        self.assertEqual(run_closure_if_else(), 60)


class TestClosureNestedControl(unittest.TestCase):
    def test_all(self):
        # i=0 is even -> return 0*100 = 0
        self.assertEqual(run_closure_nested_control(), 0)


class TestClosureSequential(unittest.TestCase):
    def test_all(self):
        # 42 + 8 = 50
        self.assertEqual(run_closure_sequential(), 50)


class TestClosureRepeated(unittest.TestCase):
    def test_repeated_calls(self):
        # 2*3 + 3*3 = 6 + 9 = 15
        self.assertEqual(run_closure_repeated(), 15)


class TestClosureNoArg(unittest.TestCase):
    def test_all(self):
        # returns 99
        self.assertEqual(run_closure_no_arg(), 99)


class TestClosureLocalMutation(unittest.TestCase):
    def test_local_reset_per_call(self):
        # total is local (not captured) so it is reset each call
        # call(10) -> 10; call(20) -> 20; total = 30
        self.assertEqual(run_closure_local_mutation(), 30)


# =====================================================================
# Test classes (runtime capture)
# =====================================================================

class TestClosureRuntimeCapture(unittest.TestCase):
    def test_all(self):
        # 5 + 10 = 15
        self.assertEqual(run_closure_runtime_capture(), 15)


class TestClosureLoopCapture(unittest.TestCase):
    def test_all(self):
        # loop i=0,1,2: each call returns 1+i
        # total = 1 + 2 + 3 = 6
        self.assertEqual(run_closure_loop_capture(), 6)


class TestClosureMultiRuntimeCapture(unittest.TestCase):
    def test_all(self):
        # 2 + 3 + 4 = 9
        self.assertEqual(run_closure_multi_runtime_capture(), 9)


class TestClosureMixedCapture(unittest.TestCase):
    def test_all(self):
        # 2 + 10 + 7 = 19
        self.assertEqual(run_closure_mixed_capture(), 19)


class TestClosureNestedClosure(unittest.TestCase):
    def test_all(self):
        self.assertEqual(run_closure_nested_closure(), 15)


class TestClosureCompositeCapture(unittest.TestCase):
    def test_struct_capture(self):
        self.assertEqual(run_closure_struct_capture(), 12)

    def test_union_capture(self):
        self.assertEqual(run_closure_union_capture(), 17)

    def test_ptr_capture(self):
        self.assertEqual(run_closure_ptr_capture(), 42)

    def test_array_capture(self):
        self.assertEqual(run_closure_array_capture(), 15)


@compile
def float_lambda() -> i32:
    api = instantiate(lambda value: value + 0.25)
    state = api.init()
    return i32(api.call(ptr(state), f64(1.0)) * 100.0)


@compile
def constant_capture() -> i32:
    offset = 0.25
    api = instantiate(lambda value: value + offset)
    state = api.init()
    offset = 99.0
    return i32(api.call(ptr(state), f64(1.0)) * 100.0)


@compile
def wide_lambda() -> i64:
    api = instantiate(lambda value: value + 1)
    state = api.init()
    return api.call(ptr(state), i64(1099511627776))


@compile
def inferred_return() -> i32:
    offset: f64 = 0.25

    def work(value: f64):
        return value + offset

    api = instantiate(work)
    state = api.init()
    return i32(api.call(ptr(state), 1.0) * 100.0)


@compile
def keyword_return() -> i32:
    def work(value: f64):
        return value + 0.25

    api = instantiate(work)
    state = api.init()
    return i32(api.call(ptr(state), value=1.0) * 100.0)


@compile
def dependent_signature() -> i32:
    def work(T, value: 'T') -> 'T':
        return value + 0.25

    api = instantiate(work)
    state = api.init()
    return i32(api.call(ptr(state), f64, 1.0) * 100.0)


@compile
def prefix_return() -> i32:
    def work(value: f64):
        amount = value + 0.25
        return amount

    api = instantiate(work)
    state = api.init()
    return i32(api.call(ptr(state), 1.0) * 100.0)


@compile
def increment(target: ptr[i32]) -> void:
    target[0] += 1


@compile
def return_before_cleanup() -> i32:
    def work(target: ptr[i32]):
        defer(increment, target)
        return target[0]

    api = instantiate(work)
    state = api.init()
    count: i32 = 1
    before = api.call(ptr(state), ptr(count))
    return before * 10 + count


@compile
def linear_return_value() -> i32:
    def work(token):
        return move(token)

    api = instantiate(work)
    state = api.init()
    token = linear()
    returned = api.call(ptr(state), move(token))
    consume(returned)
    return 42


@compile
def read_once(count: ptr[i32]) -> f64:
    count[0] += 1
    return 1.25


@compile
def call_argument_once() -> i32:
    count: i32 = 0
    api = instantiate(lambda value: value)
    state = api.init()
    value = api.call(ptr(state), read_once(ptr(count)))
    return count * 1000 + i32(value * 100.0)


@compile
def implicit_void() -> i32:
    def work(target: ptr[i32]):
        target[0] += 1

    api = instantiate(work)
    state = api.init()
    count: i32 = 0
    api.call(ptr(state), ptr(count))
    return count


@compile
def optional_argument_calls() -> i32:
    def choose(value: i32 = 7) -> i32:
        return value

    api = instantiate(choose)
    state = api.init()
    return (api.call(ptr(state)) * 10000
            + api.call(ptr(state), 19) * 100
            + api.call(ptr(state), value=23))


@compile
def runtime_default_calls() -> i32:
    count: i32 = 0

    def choose(value: f64 = read_once(ptr(count))) -> f64:
        return value

    api = instantiate(choose)
    state = api.init()
    override = api.call(ptr(state), 2.5)
    first = api.call(ptr(state))
    second = api.call(ptr(state))
    return count * 10000 + i32((override + first + second) * 100.0)


@compile
def generic_runtime_default(seed: f64) -> i32:
    def twice(value=seed):
        return value * 2.0

    api = instantiate(twice)
    state = api.init()
    return i32(api.call(ptr(state)) * 100.0) + i32(api.call(ptr(state), 2.5) * 100.0)


@compile
def mixed_parameter_calls() -> i32:
    def combine(head: i32, /, *values: struct[i32, i32], tail: i32 = 5,
                **named: struct['bonus': i32]) -> i32:
        return head + values[0] + values[1] + tail + named.bonus

    api = instantiate(combine)
    state = api.init()
    defaulted = api.call(ptr(state), 1, 19, 23, bonus=9)
    explicit = api.call(ptr(state), 1, 19, 23, tail=7, bonus=9)
    return defaulted * 100 + explicit


@compile
def default_parameter_name_collisions(seed: i32) -> i32:
    def choose(s: i32 = seed, _default_s: i32 = 2) -> i32:
        return s * 10 + _default_s

    api = instantiate(choose)
    state = api.init()
    return api.call(ptr(state)) * 100 + api.call(ptr(state), 3, 4)


@compile
def runtime_default_state_instances(seed: i32) -> i32:
    def choose(value: i32 = seed) -> i32:
        return value

    api = instantiate(choose)
    first = api.init()
    second: api.State
    api.init(ptr(second), seed + 1)
    return api.call(ptr(first)) * 100 + api.call(ptr(second))


class TestClosureCallSignature(unittest.TestCase):
    def test_keyword_only_parameter_rejects_positional_arguments(self):
        with CompileSession(), patch.object(logger, 'raise_on_error', True):
            @compile(suffix='materialized_keyword_only_rejection')
            def bad() -> i32:
                def choose(*, value: i32) -> i32:
                    return value

                api = instantiate(choose)
                state = api.init()
                return api.call(ptr(state), 99)

            with self.assertRaisesRegex(TypeError, 'too many positional'):
                bad()

    def test_positional_only_parameter_rejects_keyword_arguments(self):
        with CompileSession(), patch.object(logger, 'raise_on_error', True):
            @compile(suffix='materialized_positional_only_rejection')
            def bad() -> i32:
                def choose(value: i32, /) -> i32:
                    return value

                api = instantiate(choose)
                state = api.init()
                return api.call(ptr(state), value=99)

            with self.assertRaisesRegex(TypeError, 'positional-only'):
                bad()

    def test_optional_arguments_remain_overridable(self):
        self.assertEqual(optional_argument_calls(), 71923)

    def test_runtime_default_is_evaluated_once_at_definition(self):
        self.assertEqual(runtime_default_calls(), 10500)

    def test_generic_call_reads_the_materialized_default(self):
        self.assertEqual(generic_runtime_default(1.25), 750)
        self.assertEqual(generic_runtime_default(3.0), 1100)

    def test_collectors_and_parameter_kinds_survive_materialization(self):
        self.assertEqual(mixed_parameter_calls(), 5759)

    def test_state_fields_cannot_collide_with_parameter_names(self):
        self.assertEqual(default_parameter_name_collisions(7), 7234)
        self.assertEqual(default_parameter_name_collisions(9), 9234)

    def test_runtime_default_is_read_from_the_supplied_state(self):
        self.assertEqual(runtime_default_state_instances(7), 708)
        self.assertEqual(runtime_default_state_instances(11), 1112)


class TestClosureReturnTypes(unittest.TestCase):
    def test_lambda_float_return_is_not_truncated(self):
        self.assertEqual(float_lambda(), 125)

    def test_compiletime_capture_uses_definition_bindings(self):
        self.assertEqual(constant_capture(), 125)

    def test_lambda_wide_return_is_not_truncated(self):
        self.assertEqual(wide_lambda(), 1099511627777)

    def test_unannotated_return_reuses_inline_types(self):
        self.assertEqual(inferred_return(), 125)

    def test_keyword_arguments_use_normal_closure_binding(self):
        self.assertEqual(keyword_return(), 125)

    def test_dependent_annotations_resolve_after_argument_binding(self):
        self.assertEqual(dependent_signature(), 125)

    def test_return_expression_sees_callee_locals(self):
        self.assertEqual(prefix_return(), 125)

    def test_return_value_is_evaluated_before_defer_cleanup(self):
        self.assertEqual(return_before_cleanup(), 12)

    def test_linear_return_transfers_ownership_once(self):
        self.assertEqual(linear_return_value(), 42)

    def test_call_argument_is_evaluated_once(self):
        self.assertEqual(call_argument_once(), 1125)

    def test_implicit_void_has_no_synthetic_integer_return(self):
        self.assertEqual(implicit_void(), 1)


if __name__ == '__main__':
    unittest.main()
