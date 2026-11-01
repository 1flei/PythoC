"""Finite classifiers keep typed signatures and star-import visibility."""

import unittest

from pythoc import f32, f64, i32
from pythoc.libc import math


class LibcFiniteBindingsTest(unittest.TestCase):
    def test_classifiers_are_exported_native_functions(self):
        for name, value_type in [
            ('__inline_isfinitef', f32),
            ('__inline_isfinited', f64),
            ('__inline_isfinitel', f64),
        ]:
            with self.subTest(name=name):
                self.assertIn(name, math.__all__)
                function = getattr(math, name)
                self.assertTrue(function._is_compiled)
                self.assertIs(function._func_info.param_type_hints['x'], value_type)
                self.assertIs(function._func_info.return_type_hint, i32)
        self.assertEqual(len(math.__all__), len(set(math.__all__)))


if __name__ == '__main__':
    unittest.main()
