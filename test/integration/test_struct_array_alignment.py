#!/usr/bin/env python3
"""Array field alignment: offsetof/sizeof must follow C rules.

C aligns an array to its element type, so a char array can sit at an odd
offset.  Compiled field access uses LLVM's layout; the compile-time
offsetof()/sizeof() builtins must report the same layout.
"""

import unittest
from pythoc import compile, u8, i8, i32, array, ptr, sizeof, offsetof


@compile
class OddArray:
    a: u8
    b: array[i8, 4]
    # C layout: a at 0, b at 1, sizeof 5, alignment 1.


@compile
def report_sizeof() -> i32:
    return sizeof(OddArray)


@compile
def report_offsetof() -> i32:
    return offsetof("OddArray", "b")


@compile
def write_b_then_read_raw() -> i32:
    """Write through the field, read back through a byte pointer at the
    C offset (1).  Guards against offsetof() and codegen disagreeing."""
    s: OddArray
    s.b[0] = 42
    p: ptr[i8] = ptr[i8](ptr(s))
    return p[1]


class TestStructArrayAlignment(unittest.TestCase):
    def test_offsetof_array_field(self):
        self.assertEqual(report_offsetof(), 1)

    def test_sizeof_array_field(self):
        self.assertEqual(report_sizeof(), 5)

    def test_codegen_matches_c_layout(self):
        self.assertEqual(write_b_then_read_raw(), 42)


if __name__ == '__main__':
    unittest.main()
