"""modf uses the C fractional-return and integral-output prototype."""

import unittest

from pythoc import f64, ptr
from pythoc.libc.math import modf


class ModfBindingTest(unittest.TestCase):
    def test_modf_has_a_typed_integral_output_pointer(self):
        self.assertEqual(len(modf.param_types), 2)
        self.assertIs(modf.param_types[0][1], f64)
        self.assertEqual(modf.param_types[1][1].get_llvm_type(),
                         ptr[f64].get_llvm_type())
        self.assertIs(modf.return_type, f64)


if __name__ == '__main__':
    unittest.main()
