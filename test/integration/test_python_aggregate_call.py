"""Python -> PythoC vectorcall for pointers, aggregates, and i128."""

import ctypes
import os
import struct as pystruct
import subprocess
import unittest

from pythoc import (
    array,
    compile,
    const,
    consume,
    enum,
    f64,
    i8,
    i32,
    i64,
    i128,
    linear,
    nullptr,
    ptr,
    struct,
    union,
    u128,
    void,
)


@enum
class Color:
    Red: None
    Green: None


@enum(i32)
class Result:
    Ok: i32
    Err: i32


@compile
def deref(p: ptr[i32]) -> i32:
    return p[0]


@compile
def store_i32(p: ptr[i32], value: i32) -> void:
    p[0] = value


@compile
def first_byte(p: ptr[i8]) -> i32:
    return p[0]


@compile
def is_null(p: ptr[i32]) -> i32:
    if p == nullptr:
        return 1
    return 0


@compile
def shift(p: struct[i32, i32]) -> struct[i32, i32]:
    result: struct[i32, i32] = (p[0] + 1, p[1] + 2)
    return result


@compile
def shift_sum(p: struct[i32, i32]) -> i32:
    q: struct[i32, i32] = shift(p)
    return q[0] + q[1]


@compile
def add_xy(p: struct[f64, f64]) -> f64:
    return p[0] + p[1]


@compile
def sum4s(p: struct[i64, i64, i64, i64]) -> i64:
    return p[0] + p[1] + p[2] + p[3]


@compile
def make4(x: i64) -> struct[i64, i64, i64, i64]:
    result: struct[i64, i64, i64, i64] = (x, x + 1, x + 2, x + 3)
    return result


@compile
def sum_pair(p: struct[array[i32, 2]]) -> i32:
    return p[0][0] + p[0][1]


@compile
def sum4(xs: array[i32, 4]) -> i32:
    return xs[0] + xs[1] + xs[2] + xs[3]


@compile
def sum23(xs: array[i32, 2, 3]) -> i32:
    return xs[0][0] + xs[0][1] + xs[0][2] + xs[1][0] + xs[1][1] + xs[1][2]


@compile
def union_low(n: union[i32, f64]) -> i32:
    return n[0]


@compile
def add128(a: i128, b: i128) -> i128:
    return a + b


@compile
def add_u128(a: u128, b: u128) -> u128:
    return a + b


@compile
def inc_const(x: const[i32]) -> i32:
    return x + 1


@compile
def after_token(tok: linear, x: i32) -> i32:
    consume(tok)
    return x


@compile
def before_token(x: i32, tok: linear) -> i32:
    consume(tok)
    return x


@compile
class Point:
    x: i32
    y: i32


@compile
def manhattan(p: Point) -> i32:
    return p.x + p.y


@compile
def color_tag(c: Color) -> i32:
    tag: i8 = c[0]
    return tag


@compile
def result_tag(r: Result) -> i32:
    tag: i32 = r[0]
    return tag


class TestPythonAggregateCall(unittest.TestCase):
    def test_pointer_struct_array_union_and_i128(self):
        slot = ctypes.c_int32(41)
        self.assertEqual(deref(ctypes.addressof(slot)), 41)
        self.assertTrue(deref.is_fast_bound())
        store_i32(ctypes.addressof(slot), 9)
        self.assertEqual(slot.value, 9)
        self.assertEqual(first_byte(b'AB'), 65)
        self.assertEqual(is_null(None), 1)
        self.assertEqual(is_null(0), 1)

        self.assertEqual(shift((10, 20)), (11, 22))
        self.assertTrue(shift.is_fast_bound())
        self.assertEqual(shift_sum((10, 20)), 33)
        self.assertAlmostEqual(add_xy((1.5, 2.25)), 3.75)
        self.assertEqual(sum4s((1, 2, 3, 4)), 10)
        self.assertEqual(make4(5), (5, 6, 7, 8))
        self.assertTrue(make4.is_fast_bound())

        self.assertEqual(sum_pair(((10, 32),)), 42)
        self.assertEqual(sum4((1, 2, 3, 4)), 10)
        self.assertTrue(sum4.is_fast_bound())
        self.assertEqual(sum23(((1, 2, 3), (4, 5, 6))), 21)

        self.assertEqual(union_low(pystruct.pack('<q', 42)), 42)
        self.assertTrue(union_low.is_fast_bound())

        self.assertEqual(add128(20, 22), 42)
        self.assertEqual(add128(-2, -3), -5)
        self.assertEqual(add128(1 << 100, 3), (1 << 100) + 3)
        self.assertTrue(add128.is_fast_bound())
        self.assertEqual(add_u128(2 ** 100, 5), 2 ** 100 + 5)
        with self.assertRaises(OverflowError):
            add128(1 << 127, 0)

        self.assertEqual(inc_const(41), 42)
        self.assertTrue(inc_const.is_fast_bound())
        self.assertEqual(after_token(None, 7), 7)
        self.assertEqual(before_token(7, None), 7)
        self.assertTrue(before_token.is_fast_bound())
        self.assertEqual(manhattan((3, 4)), 7)
        self.assertEqual(manhattan({'x': 3, 'y': 4}), 7)
        self.assertEqual(color_tag((0,)), 0)
        self.assertEqual(color_tag((1,)), 1)
        self.assertTrue(color_tag.is_fast_bound())
        self.assertEqual(result_tag((0, pystruct.pack('<i', 42))), 0)
        self.assertEqual(result_tag((1, pystruct.pack('<i', 7))), 1)
        self.assertTrue(result_tag.is_fast_bound())

        obj = os.path.join(
            'build', 'test', 'integration', 'test_python_aggregate_call.o'
        )
        relocations = subprocess.check_output(['objdump', '-r', obj], text=True)
        symbols = subprocess.check_output(['nm', obj], text=True)
        self.assertIn('shift', relocations)
        self.assertNotIn('PyObject', symbols)
        self.assertNotIn('pythoc_pyadapter_', symbols)


if __name__ == '__main__':
    unittest.main()
