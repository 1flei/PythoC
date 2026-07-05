#!/usr/bin/env python3
"""Integration tests for parametric polymorphism via ``param``."""

from __future__ import annotations

import unittest

from pythoc import i32, i64, f64, u64, ptr, array, void, compile, param, sizeof
from pythoc.libc import malloc


# ---------------------------------------------------------------------------
# Success cases
# ---------------------------------------------------------------------------

@compile
def identity(T: param, x: T) -> T:
    return x


@compile
def cast_to(T: param, x: i32) -> T:
    return T(x)


@compile
def make_array(T: param, n: param) -> ptr[T]:
    return malloc(sizeof(array[T, n]))


@compile
def multi_param(T: param, x: i32, U: param, y: ptr[U]) -> T:
    if x > 0:
        return T(x)
    return y[0]


@compile
def all_param(T: param, N: param) -> i32:
    """Every parameter is a compile-time parameter; the call site supplies
    only parametric arguments and receives a zero-argument callable.
    """
    return sizeof(array[T, N])


@compile
def param_at_end(x: i32, T: param) -> T:
    return T(x)


@compile
def param_in_middle(x: i32, T: param, y: i32) -> i32:
    z: T = T(x + y)
    return i32(z)


@compile
def square(T: param, x: T) -> T:
    return x * x


@compile
def make_single(T: param) -> T:
    p: ptr[T] = malloc(sizeof(T))
    p[0] = T(42)
    return p[0]


@compile
def make_value(T: param) -> T:
    """Return type is determined by the compile-time type parameter."""
    return T(42)


@compile
def gen_counter(T: param, start: T, count: i32):
    """Parametric yield generator: yields ``start + T(i)`` for i in [0, count)."""
    i: i32 = 0
    while i < count:
        yield start + T(i)
        i = i + 1


@compile
def gen_range(T: param, n: param):
    """Parametric yield generator with a compile-time count parameter."""
    i: T = T(0)
    while i < T(n):
        yield i
        i = i + T(1)


# Helper runners must be defined at module level (before any native execution)
# so that PythoC's build manager does not reject them.

@compile
def make_array_runner() -> i32:
    p: ptr[i32] = make_array(i32, 4)
    p[0] = 10
    p[3] = 20
    result: i32 = p[0] + p[3]
    return result


@compile
def multi_param_runner(value: i32) -> i32:
    return multi_param(i32, 1, i32, ptr(value))


@compile
def use_fixed_array() -> i32:
    p: ptr[i32] = make_array(i32, 5)
    p[0] = 10
    p[4] = 20
    return p[0] + p[4]


@compile
def use_make_value_i32() -> i32:
    return make_value(i32)


@compile
def use_make_value_f64() -> f64:
    return make_value(f64)


@compile
def collect_gen_counter_i32() -> i32:
    total: i32 = 0
    for v in gen_counter(i32, 10, 3):
        total = total + v
    return total


@compile
def collect_gen_counter_i64() -> i64:
    total: i64 = 0
    for v in gen_counter(i64, 100, 3):
        total = total + v
    return total


@compile
def collect_gen_range_u64() -> u64:
    total: u64 = 0
    for v in gen_range(u64, 5):
        total = total + v
    return total


class TestParametricPolymorphism(unittest.TestCase):
    """End-to-end tests for Phase 1 parametric polymorphism."""

    def test_identity_i32(self):
        self.assertEqual(identity(i32, 42), 42)

    def test_identity_f64(self):
        self.assertEqual(identity(f64, 3.5), 3.5)

    def test_cast_to(self):
        self.assertEqual(cast_to(i32, 7), 7)
        self.assertEqual(cast_to(f64, 7), 7.0)

    def test_make_array(self):
        self.assertEqual(make_array_runner(), 30)

    def test_multi_param_out_of_order(self):
        self.assertEqual(multi_param_runner(7), 1)

    def test_all_param(self):
        # sizeof(array[i32, 4]) == 16
        self.assertEqual(all_param(i32, 4), 16)
        # sizeof(array[i64, 2]) == 16
        self.assertEqual(all_param(i64, 2), 16)

    def test_param_at_end(self):
        self.assertEqual(param_at_end(7, f64), 7.0)
        self.assertEqual(param_at_end(7, i32), 7)

    def test_param_in_middle(self):
        self.assertEqual(param_in_middle(1, i32, 2), 3)
        self.assertEqual(param_in_middle(1, i64, 2), 3)

    def test_constant_param_for_array_size(self):
        self.assertEqual(use_fixed_array(), 30)

    def test_specialization_per_type(self):
        self.assertEqual(square(i32, 3), 9)
        self.assertEqual(square(f64, 3.0), 9.0)
        self.assertEqual(square(i64, 7), 49)

    def test_param_type_for_local_and_pointer(self):
        self.assertEqual(make_single(i32), 42)
        self.assertEqual(make_single(f64), 42.0)

    def test_return_type_from_param(self):
        self.assertEqual(use_make_value_i32(), 42)
        self.assertEqual(use_make_value_f64(), 42.0)

    def test_parametric_yield_i32(self):
        # 10 + 11 + 12 == 33
        self.assertEqual(collect_gen_counter_i32(), 33)

    def test_parametric_yield_i64(self):
        # 100 + 101 + 102 == 303
        self.assertEqual(collect_gen_counter_i64(), 303)

    def test_parametric_yield_constant_param(self):
        # 0 + 1 + 2 + 3 + 4 == 10
        self.assertEqual(collect_gen_range_u64(), 10)

    def test_specialization_cache(self):
        # Two calls with the same parametric args should reuse the wrapper.
        spec1 = identity._factory_func(i32)
        spec2 = identity._factory_func(i32)
        self.assertIs(spec1, spec2)

        spec3 = identity._factory_func(f64)
        self.assertIsNot(spec3, spec1)

    def test_partial_application(self):
        # Binding only the compile-time parameter returns a specialized
        # wrapper that can be called with the remaining runtime argument.
        identity_i32 = identity(i32)
        self.assertEqual(identity_i32(42), 42)
        self.assertEqual(identity_i32(-7), -7)

        identity_f64 = identity(f64)
        self.assertEqual(identity_f64(3.5), 3.5)


if __name__ == "__main__":
    unittest.main()
