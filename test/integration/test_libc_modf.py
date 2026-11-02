"""The modf binding preserves both results and the C output-pointer ABI."""

import ctypes
import math
from pathlib import Path
import struct
import sys
import unittest

from pythoc import compile, f64, ptr
from test.native_reference import c_reference
from pythoc.libc.math import modf


@compile
def split_value(x: f64, integral: ptr[f64]) -> f64:
    return modf(x, integral)


class GuardedOutput(ctypes.Structure):
    _fields_ = [('before', ctypes.c_uint64), ('integral', ctypes.c_double),
                ('after', ctypes.c_uint64)]


class ModfRuntimeTest(unittest.TestCase):
    def test_fraction_and_integral_match_an_independent_c_call(self):
        source = (Path(__file__).resolve().parent / 'fixtures'
                  / 'libc_modf_reference.c')
        with c_reference(source, ['m']) as library:
            reference = library.split_reference
            reference.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
            reference.restype = ctypes.c_double
            cases = [0.0, -0.0, 0.25, -0.25, 1.75, -1.75,
                     2.0 ** 51 + 0.5, -(2.0 ** 51 + 0.5),
                     float.fromhex('0x0.0000000000001p-1022'),
                     -float.fromhex('0x0.0000000000001p-1022'),
                     sys.float_info.max, -sys.float_info.max,
                     math.inf, -math.inf, math.nan]
            before = 0xfedcba9876543210
            after = 0x0123456789abcdef
            for value in cases:
                with self.subTest(value=value):
                    actual = GuardedOutput(before, 1234567.25, after)
                    expected = GuardedOutput(before, 1234567.25, after)
                    actual_pointer = (ctypes.addressof(actual)
                                      + GuardedOutput.integral.offset)
                    expected_pointer = ctypes.cast(
                        ctypes.byref(expected, GuardedOutput.integral.offset),
                        ctypes.POINTER(ctypes.c_double),
                    )
                    actual_fraction = split_value(value, actual_pointer)
                    expected_fraction = reference(value, expected_pointer)
                    for result, target in [
                        (actual_fraction, expected_fraction),
                        (actual.integral, expected.integral),
                    ]:
                        if math.isnan(target):
                            self.assertTrue(math.isnan(result))
                        else:
                            self.assertEqual(struct.pack('=d', result),
                                             struct.pack('=d', target))
                    self.assertEqual((actual.before, actual.after),
                                     (before, after))


if __name__ == '__main__':
    unittest.main()
