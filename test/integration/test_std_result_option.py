#!/usr/bin/env python3
"""Integration tests for std option/result do-notation."""

import unittest

from pythoc import compile, enum, i32, struct
from pythoc import bool as pc_bool
from pythoc.std.result import result_wrap, option_wrap


# =============================================================================
# Basic Option/Result types
# =============================================================================

O, O_api = option_wrap(i32, name="O")


@enum(i32)
class Err:
    Code: i32


R, R_api = result_wrap(i32, Err, name="R")


# =============================================================================
# Multi-variant error enum
# =============================================================================

@enum(i32)
class MultiErr:
    NotFound: i32
    Overflow: None
    InvalidArg: i32


MR, MR_api = result_wrap(i32, MultiErr, name="MR")


# =============================================================================
# Struct-valued Option and Result
# =============================================================================

Point = struct["x": i32, "y": i32]

OP, OP_api = option_wrap(Point, name="OP")
RP, RP_api = result_wrap(Point, Err, name="RP")


# =============================================================================
# Basic helper functions
# =============================================================================

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


# =============================================================================
# Basic do-notation tests
# =============================================================================

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


# =============================================================================
# Triple bind chain: all ok, fail at each position
# =============================================================================

@compile
def checked_double(x: i32) -> R:
    if x < 100:
        return R_api.ok(x * 2)
    return R_api.err(Err(Err.Code, 11))


@compile
def test_triple_bind_all_ok() -> i32:
    # 3 -> +2=5 -> +3=8 -> *2=16, final = 5+8+16 = 29
    r = R_api.do(
        v1 + v2 + v3
        for v1 in R_api.bind(checked_add2(3))
        for v2 in R_api.bind(checked_add3(v1))
        for v3 in R_api.bind(checked_double(v2))
    )
    match r:
        case (R.Ok, value):
            return value
        case (R.Err, _):
            return -1


@compile
def test_triple_bind_fail_third() -> i32:
    # 3 -> +2=5 -> +3=8, but double(200) -> err(11)
    r = R_api.do(
        v1 + v2 + v3
        for v1 in R_api.bind(checked_add2(3))
        for v2 in R_api.bind(checked_add3(v1))
        for v3 in R_api.bind(checked_double(200))
    )
    match r:
        case (R.Ok, value):
            return value
        case (R.Err, e):
            match e:
                case (Err.Code, c):
                    return c
                case _:
                    return -999


# =============================================================================
# Multi-variant error: dispatch on different error variants
# =============================================================================

@compile
def multi_err_fn(x: i32) -> MR:
    if x > 0:
        return MR_api.ok(x * 10)
    if x == 0:
        return MR_api.err(MultiErr(MultiErr.Overflow))
    return MR_api.err(MultiErr(MultiErr.NotFound, -x))


@compile
def test_multi_err_ok() -> i32:
    r = MR_api.do(v + 1 for v in MR_api.bind(multi_err_fn(5)))
    match r:
        case (MR.Ok, value):
            return value
        case (MR.Err, _):
            return -1


@compile
def test_multi_err_overflow() -> i32:
    r = MR_api.do(v + 1 for v in MR_api.bind(multi_err_fn(0)))
    match r:
        case (MR.Ok, _):
            return -1
        case (MR.Err, e):
            match e:
                case (MultiErr.Overflow):
                    return 888
                case (MultiErr.NotFound, _):
                    return -2
                case (MultiErr.InvalidArg, _):
                    return -3


@compile
def test_multi_err_not_found() -> i32:
    r = MR_api.do(v + 1 for v in MR_api.bind(multi_err_fn(-7)))
    match r:
        case (MR.Ok, _):
            return -1
        case (MR.Err, e):
            match e:
                case (MultiErr.Overflow):
                    return -2
                case (MultiErr.NotFound, code):
                    return code
                case (MultiErr.InvalidArg, _):
                    return -3


# =============================================================================
# Chained multi-err: first err wins
# =============================================================================

@compile
def multi_err_fn2(x: i32) -> MR:
    if x > 10:
        return MR_api.ok(x - 10)
    return MR_api.err(MultiErr(MultiErr.InvalidArg, x))


@compile
def test_multi_err_chain_first_wins() -> i32:
    # multi_err_fn(0) -> Overflow, multi_err_fn2 never runs
    r = MR_api.do(
        v1 + v2
        for v1 in MR_api.bind(multi_err_fn(0))
        for v2 in MR_api.bind(multi_err_fn2(v1))
    )
    match r:
        case (MR.Ok, _):
            return -1
        case (MR.Err, e):
            match e:
                case (MultiErr.Overflow):
                    return 777
                case (MultiErr.NotFound, _):
                    return -2
                case (MultiErr.InvalidArg, _):
                    return -3


@compile
def test_multi_err_chain_second_wins() -> i32:
    # multi_err_fn(1) -> ok(10), multi_err_fn2(10) -> err(InvalidArg, 10)
    r = MR_api.do(
        v1 + v2
        for v1 in MR_api.bind(multi_err_fn(1))
        for v2 in MR_api.bind(multi_err_fn2(v1))
    )
    match r:
        case (MR.Ok, _):
            return -1
        case (MR.Err, e):
            match e:
                case (MultiErr.Overflow):
                    return -2
                case (MultiErr.NotFound, _):
                    return -3
                case (MultiErr.InvalidArg, code):
                    return code


# =============================================================================
# Struct-valued Option: Some(Point) / None
# =============================================================================

@compile
def make_point(x: i32, y: i32) -> OP:
    if x >= 0:
        p: Point = (x, y)
        return OP_api.some(p)
    return OP_api.none()


@compile
def offset_point(p: Point, dx: i32, dy: i32) -> Point:
    r: Point = (p.x + dx, p.y + dy)
    return r


@compile
def test_option_struct_some() -> i32:
    r = OP_api.do(
        offset_point(p, 10, 20)
        for p in OP_api.bind(make_point(3, 4))
    )
    match r:
        case (OP.Some, value):
            return value.x + value.y
        case (OP.NoneVal):
            return -1


@compile
def test_option_struct_none() -> i32:
    r = OP_api.do(
        offset_point(p, 10, 20)
        for p in OP_api.bind(make_point(-1, 4))
    )
    match r:
        case (OP.Some, _):
            return -1
        case (OP.NoneVal):
            return 0


# =============================================================================
# Struct-valued Result: Ok(Point) / Err
# =============================================================================

@compile
def checked_make_point(x: i32, y: i32) -> RP:
    if x >= 0:
        p: Point = (x, y)
        return RP_api.ok(p)
    return RP_api.err(Err(Err.Code, 42))


@compile
def test_result_struct_ok() -> i32:
    r = RP_api.do(
        offset_point(p, 100, 200)
        for p in RP_api.bind(checked_make_point(5, 7))
    )
    match r:
        case (RP.Ok, value):
            return value.x + value.y
        case (RP.Err, _):
            return -1


@compile
def test_result_struct_err() -> i32:
    r = RP_api.do(
        offset_point(p, 100, 200)
        for p in RP_api.bind(checked_make_point(-1, 7))
    )
    match r:
        case (RP.Ok, _):
            return -1
        case (RP.Err, e):
            match e:
                case (Err.Code, c):
                    return c
                case _:
                    return -999


# =============================================================================
# Struct-valued Result with two binds: both points ok -> combine
# =============================================================================

@compile
def test_result_struct_chain_ok() -> i32:
    r = RP_api.do(
        offset_point(p1, p2.x, p2.y)
        for p1 in RP_api.bind(checked_make_point(10, 20))
        for p2 in RP_api.bind(checked_make_point(30, 40))
    )
    match r:
        case (RP.Ok, value):
            return value.x + value.y
        case (RP.Err, _):
            return -1


@compile
def test_result_struct_chain_second_err() -> i32:
    r = RP_api.do(
        offset_point(p1, p2.x, p2.y)
        for p1 in RP_api.bind(checked_make_point(10, 20))
        for p2 in RP_api.bind(checked_make_point(-5, 40))
    )
    match r:
        case (RP.Ok, _):
            return -1
        case (RP.Err, e):
            match e:
                case (Err.Code, c):
                    return c
                case _:
                    return -999


# =============================================================================
# Option chaining: bind result feeds into another option bind
# =============================================================================

@compile
def clamp_positive(x: i32) -> O:
    if x > 0:
        return O_api.some(x)
    return O_api.none()


@compile
def test_option_chain_two_binds_ok() -> i32:
    # 5 -> +1=6 -> clamp(6)=Some(6) -> 6+100=106
    r = O_api.do(
        v2 + 100
        for v1 in O_api.bind(maybe_add1(5))
        for v2 in O_api.bind(clamp_positive(v1))
    )
    match r:
        case (O.Some, value):
            return value
        case (O.NoneVal):
            return -1


@compile
def test_option_chain_first_none() -> i32:
    # 0 -> maybe_add1 -> None, second bind never runs
    r = O_api.do(
        v2 + 100
        for v1 in O_api.bind(maybe_add1(0))
        for v2 in O_api.bind(clamp_positive(v1))
    )
    match r:
        case (O.Some, _):
            return -1
        case (O.NoneVal):
            return 0


@compile
def test_option_chain_second_none() -> i32:
    # maybe_add1(5)=Some(6), clamp_positive(-6) would be None
    # but 6 > 0, so let's use clamp_positive with a negative input
    # We need a function that returns None for positive inputs > threshold
    r = O_api.do(
        v2 + 100
        for v1 in O_api.bind(maybe_add1(5))
        for v2 in O_api.bind(clamp_positive(-v1))
    )
    match r:
        case (O.Some, _):
            return -1
        case (O.NoneVal):
            return 0


# =============================================================================
# Boundary values: i32 edge cases
# =============================================================================

@compile
def test_result_boundary_one() -> i32:
    # x=1 -> checked_add2(1)=ok(3), checked_add3(3)=err(9) (3 <= 3)
    r = R_api.do(
        v1 + v2
        for v1 in R_api.bind(checked_add2(1))
        for v2 in R_api.bind(checked_add3(v1))
    )
    match r:
        case (R.Ok, _):
            return -1
        case (R.Err, e):
            match e:
                case (Err.Code, c):
                    return c
                case _:
                    return -999


@compile
def test_result_boundary_two() -> i32:
    # x=2 -> checked_add2(2)=ok(4), checked_add3(4)=ok(7), final = 4+7 = 11
    r = R_api.do(
        v1 + v2
        for v1 in R_api.bind(checked_add2(2))
        for v2 in R_api.bind(checked_add3(v1))
    )
    match r:
        case (R.Ok, value):
            return value
        case (R.Err, _):
            return -1


# =============================================================================
# is_some / is_none helpers
# =============================================================================

@compile
def test_is_some_true() -> i32:
    o: O = O_api.some(42)
    if O_api.is_some(o):
        return 1
    return 0


@compile
def test_is_some_false() -> i32:
    o: O = O_api.none()
    if O_api.is_some(o):
        return 1
    return 0


@compile
def test_is_none_true() -> i32:
    o: O = O_api.none()
    if O_api.is_none(o):
        return 1
    return 0


@compile
def test_is_ok_true() -> i32:
    r: R = R_api.ok(99)
    if R_api.is_ok(r):
        return 1
    return 0


@compile
def test_is_err_true() -> i32:
    r: R = R_api.err(Err(Err.Code, 5))
    if R_api.is_err(r):
        return 1
    return 0


# =============================================================================
# Direct match without do-notation
# =============================================================================

@compile
def test_option_direct_match_some() -> i32:
    o: O = maybe_add1(10)
    match o:
        case (O.Some, v):
            return v * 2
        case (O.NoneVal):
            return -1


@compile
def test_result_direct_match_ok() -> i32:
    r: R = checked_add2(5)
    match r:
        case (R.Ok, v):
            return v + 1000
        case (R.Err, _):
            return -1


@compile
def test_result_direct_match_err() -> i32:
    r: R = checked_add2(-1)
    match r:
        case (R.Ok, _):
            return -1
        case (R.Err, e):
            match e:
                case (Err.Code, c):
                    return c + 100
                case _:
                    return -999


# =============================================================================
# Test class
# =============================================================================

class TestStdResultOption(unittest.TestCase):
    # -- basic do-notation --
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

    # -- triple bind chain --
    def test_triple_bind_all_ok(self):
        # 3 -> +2=5 -> +3=8 -> *2=16, sum = 5+8+16 = 29
        self.assertEqual(test_triple_bind_all_ok(), 29)

    def test_triple_bind_fail_third(self):
        self.assertEqual(test_triple_bind_fail_third(), 11)

    # -- multi-variant error --
    def test_multi_err_ok(self):
        # 5 -> ok(50) -> 50+1 = 51
        self.assertEqual(test_multi_err_ok(), 51)

    def test_multi_err_overflow(self):
        self.assertEqual(test_multi_err_overflow(), 888)

    def test_multi_err_not_found(self):
        # -7 -> err(NotFound, 7)
        self.assertEqual(test_multi_err_not_found(), 7)

    def test_multi_err_chain_first_wins(self):
        self.assertEqual(test_multi_err_chain_first_wins(), 777)

    def test_multi_err_chain_second_wins(self):
        # multi_err_fn(1)=ok(10), multi_err_fn2(10)=err(InvalidArg, 10)
        self.assertEqual(test_multi_err_chain_second_wins(), 10)

    # -- struct-valued option --
    def test_option_struct_some(self):
        # make_point(3,4)=Some(Point(3,4)), offset(+10,+20)=Point(13,24), 13+24=37
        self.assertEqual(test_option_struct_some(), 37)

    def test_option_struct_none(self):
        self.assertEqual(test_option_struct_none(), 0)

    # -- struct-valued result --
    def test_result_struct_ok(self):
        # checked_make_point(5,7)=Ok(Point(5,7)), offset(+100,+200)=Point(105,207), 105+207=312
        self.assertEqual(test_result_struct_ok(), 312)

    def test_result_struct_err(self):
        self.assertEqual(test_result_struct_err(), 42)

    def test_result_struct_chain_ok(self):
        # p1=Point(10,20), p2=Point(30,40), offset(p1, p2.x, p2.y)=Point(40,60), 40+60=100
        self.assertEqual(test_result_struct_chain_ok(), 100)

    def test_result_struct_chain_second_err(self):
        self.assertEqual(test_result_struct_chain_second_err(), 42)

    # -- option chaining --
    def test_option_chain_two_binds_ok(self):
        # 5->6, clamp(6)=Some(6), 6+100=106
        self.assertEqual(test_option_chain_two_binds_ok(), 106)

    def test_option_chain_first_none(self):
        self.assertEqual(test_option_chain_first_none(), 0)

    def test_option_chain_second_none(self):
        self.assertEqual(test_option_chain_second_none(), 0)

    # -- boundary values --
    def test_result_boundary_one(self):
        # checked_add2(1)=ok(3), checked_add3(3)=err(9)
        self.assertEqual(test_result_boundary_one(), 9)

    def test_result_boundary_two(self):
        # checked_add2(2)=ok(4), checked_add3(4)=ok(7), 4+7=11
        self.assertEqual(test_result_boundary_two(), 11)

    # -- is_some/is_none/is_ok/is_err --
    def test_is_some_true(self):
        self.assertEqual(test_is_some_true(), 1)

    def test_is_some_false(self):
        self.assertEqual(test_is_some_false(), 0)

    def test_is_none_true(self):
        self.assertEqual(test_is_none_true(), 1)

    def test_is_ok_true(self):
        self.assertEqual(test_is_ok_true(), 1)

    def test_is_err_true(self):
        self.assertEqual(test_is_err_true(), 1)

    # -- direct match without do --
    def test_option_direct_match_some(self):
        # maybe_add1(10) = Some(11), 11*2 = 22
        self.assertEqual(test_option_direct_match_some(), 22)

    def test_result_direct_match_ok(self):
        # checked_add2(5) = ok(7), 7+1000 = 1007
        self.assertEqual(test_result_direct_match_ok(), 1007)

    def test_result_direct_match_err(self):
        # checked_add2(-1) = err(Code, 7), 7+100 = 107
        self.assertEqual(test_result_direct_match_err(), 107)


if __name__ == "__main__":
    unittest.main()
