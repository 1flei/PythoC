#!/usr/bin/env python3
"""Integration tests for std option/result do-notation."""

import unittest

from pythoc import compile, enum, i32
from pythoc.std.result import result_wrap, option_wrap


O, O_api = option_wrap(i32, name="O")


@enum(i32)
class Err:
    Code: i32


R, R_api = result_wrap(i32, Err, name="R")


@compile
def maybe_add1(x: i32) -> O:
    if x > 0:
        return O_api.some(x + 1)
    return O_api.none()


@compile
def checked_add2(x: i32) -> R:
    if x > 0:
        return R_api.ok(x + 2)
    return R_api.err(Err(Err.Code, 7))


@compile
def checked_add3(x: i32) -> R:
    if x > 3:
        return R_api.ok(x + 3)
    return R_api.err(Err(Err.Code, 9))


@compile
def test_option_do_ok() -> i32:
    r = O_api.do(v + 10 for v in O_api.bind(maybe_add1(5)))
    match r:
        case (O.Some, value):
            return value
        case (O.NoneVal):
            return -1


@compile
def test_option_do_none() -> i32:
    r = O_api.do(v + 10 for v in O_api.bind(maybe_add1(0)))
    match r:
        case (O.Some, value):
            return value
        case (O.NoneVal):
            return -1


@compile
def test_result_do_ok() -> i32:
    r = R_api.do(v1 + v2 for v1 in R_api.bind(checked_add2(3)) for v2 in R_api.bind(checked_add3(v1)))
    match r:
        case (R.Ok, value):
            return value
        case (R.Err, _):
            return -1


@compile
def test_result_do_first_err() -> i32:
    r = R_api.do(v1 + v2 for v1 in R_api.bind(checked_add2(0)) for v2 in R_api.bind(checked_add3(v1)))
    match r:
        case (R.Ok, value):
            return value
        case (R.Err, e):
            match e:
                case (Err.Code, c):
                    return c
                case _:
                    return -999


@compile
def test_result_do_second_err() -> i32:
    r = R_api.do(v1 + v2 for v1 in R_api.bind(checked_add2(1)) for v2 in R_api.bind(checked_add3(v1)))
    match r:
        case (R.Ok, value):
            return value
        case (R.Err, e):
            match e:
                case (Err.Code, c):
                    return c
                case _:
                    return -999


class TestStdResultOption(unittest.TestCase):
    def test_option_ok(self):
        self.assertEqual(test_option_do_ok(), 16)

    def test_option_none(self):
        self.assertEqual(test_option_do_none(), -1)

    def test_result_ok(self):
        self.assertEqual(test_result_do_ok(), 13)

    def test_result_first_err(self):
        self.assertEqual(test_result_do_first_err(), 7)

    def test_result_second_err(self):
        self.assertEqual(test_result_do_second_err(), 9)


if __name__ == "__main__":
    unittest.main()
