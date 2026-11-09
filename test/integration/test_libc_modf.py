"""The modf binding returns both parts without overwriting adjacent fields."""

import math
import struct
import sys
import unittest

from pythoc import compile, f64, ptr, u64
from pythoc.libc.math import modf


@compile
class SplitResult:
    before: u64
    integral: f64
    after: u64
    fraction: f64


@compile
def split_value(value: f64) -> SplitResult:
    result: SplitResult = (0xfedcba9876543210, 1234567.25,
                           0x0123456789abcdef, 0.0)
    result.fraction = modf(value, ptr(result.integral))
    return result


class ModfRuntimeTest(unittest.TestCase):
    def test_fraction_and_integral_match_math_modf(self):
        cases = [0.0, -0.0, 0.25, -0.25, 1.75, -1.75,
                 2.0 ** 51 + 0.5, -(2.0 ** 51 + 0.5),
                 float.fromhex('0x0.0000000000001p-1022'),
                 -float.fromhex('0x0.0000000000001p-1022'),
                 sys.float_info.max, -sys.float_info.max,
                 math.inf, -math.inf, math.nan]
        for value in cases:
            with self.subTest(value=value):
                result = split_value(value)
                fraction, integral = math.modf(value)
                for actual, expected in ((result.fraction, fraction),
                                         (result.integral, integral)):
                    if math.isnan(expected):
                        self.assertTrue(math.isnan(actual))
                    else:
                        self.assertEqual(struct.pack('=d', actual),
                                         struct.pack('=d', expected))
                self.assertEqual(result.before, 0xfedcba9876543210)
                self.assertEqual(result.after, 0x0123456789abcdef)


if __name__ == '__main__':
    unittest.main()
