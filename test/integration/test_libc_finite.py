"""Native finite classification handles IEEE boundary bit patterns."""

import math
import random
import struct
import unittest

from pythoc import compile, f32, f64, i32, u32, u64, union
from pythoc.libc.math import (
    __inline_isfinitef as finite_float,
    __inline_isfinited as finite_double,
    __inline_isfinitel as finite_long_compat,
)


@compile
def classify_float(bits: u32) -> i32:
    value: union[u32, f32]
    value[0] = bits
    return finite_float(value[1])


@compile
def classify_double(bits: u64) -> i32:
    value: union[u64, f64]
    value[0] = bits
    return finite_double(value[1])


@compile
def classify_long_compat(bits: u64) -> i32:
    value: union[u64, f64]
    value[0] = bits
    return finite_long_compat(value[1])


class LibcFiniteRuntimeTest(unittest.TestCase):
    def test_float_boundaries_and_bit_patterns(self):
        cases = [0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000,
                 0x3f800000, 0xbf800000, 0x7f7fffff, 0xff7fffff,
                 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00000,
                 0x7f800001, 0xff800001, 0x7fffffff, 0xffffffff]
        random_values = random.Random(1501)
        cases += [random_values.getrandbits(32) for _ in range(512)]
        for bits in cases:
            with self.subTest(bits=hex(bits)):
                value = struct.unpack('=f', struct.pack('=I', bits))[0]
                self.assertEqual(classify_float(bits), math.isfinite(value))

    def test_double_boundaries_and_bit_patterns(self):
        cases = [0, 0x8000000000000000, 1, 0x8000000000000001,
                 0x000fffffffffffff, 0x0010000000000000,
                 0x3ff0000000000000, 0xbff0000000000000,
                 0x7fefffffffffffff, 0xffefffffffffffff,
                 0x7ff0000000000000, 0xfff0000000000000,
                 0x7ff8000000000000, 0xfff8000000000000,
                 0x7ff0000000000001, 0xfff0000000000001,
                 0x7fffffffffffffff, 0xffffffffffffffff]
        random_values = random.Random(1502)
        cases += [random_values.getrandbits(64) for _ in range(512)]
        for bits in cases:
            with self.subTest(bits=hex(bits)):
                value = struct.unpack('=d', struct.pack('=Q', bits))[0]
                expected = math.isfinite(value)
                self.assertEqual(classify_double(bits), expected)
                self.assertEqual(classify_long_compat(bits), expected)


if __name__ == '__main__':
    unittest.main()
