#!/usr/bin/env python3
"""
Integer division semantics: pythoc follows C (truncating sdiv), not Python
true-division, for typed operands lowered through LLVM. Regression test for
the bug where i64/i32 `/` was lowered to fdiv, corrupting results (f64 bits
read as integers).

Note: pure pyconst expressions (both operands Python values) intentionally
fold with Python semantics -- that is the meta-language layer; see the
comment in ValueRefDispatcher._evaluate_python_binop.
"""

import os

from pythoc import i32, i64, u64, f64, compile


@compile
def div_runtime_i64(x: i64) -> i64:
    return x / 10


@compile
def div_runtime_neg(x: i64) -> i64:
    return x / 10  # C truncates toward zero


@compile
def div_const_i64() -> i64:
    a: i64 = 9223372036854775807
    return a / 10


@compile
def div_const_small() -> i32:
    a: i32 = 7
    b: i32 = 2
    return a / b


@compile
def div_u64(x: u64) -> u64:
    return x / 3


@compile
def mod_runtime(x: i64) -> i64:
    return x % 10


@compile
def div_f64_stays_float(x: f64) -> f64:
    return x / 4


def test_runtime():
    assert div_runtime_i64(9223372036854775807) == 922337203685477580
    assert div_runtime_neg(-7) == 0  # -7/10 truncates to 0 in C
    assert div_runtime_i64(os.getpid() * 1000003 + 12345) == (os.getpid() * 1000003 + 12345) // 10


def test_constants():
    assert div_const_i64() == 922337203685477580
    assert div_const_small() == 3


def test_unsigned_and_mod():
    assert div_u64(18446744073709551615) == 18446744073709551615 // 3
    assert mod_runtime(9223372036854775807) == 7


def test_float_division_unchanged():
    assert div_f64_stays_float(9.0) == 2.25


if __name__ == '__main__':
    test_runtime()
    test_constants()
    test_unsigned_and_mod()
    test_float_division_unchanged()
    print('PASS test_int_division')
