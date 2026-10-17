"""Public bindings use data globals, typed pointers, and explicit exports."""

import locale
import unittest

from pythoc import i32, ptr
from pythoc.decorators.extern import ExternGlobal
from pythoc.libc import errno, math, stdio, wchar
from pythoc.libc import locale as pc_locale
from pythoc.libc._platform import IS_MACOS
from pythoc.libc.stddef import wchar_t


class TestLibcLinkSupport(unittest.TestCase):
    def test_standard_stream_names_are_noncallable_data_aliases(self):
        for name, alias in (("stdin", "__stdinp"), ("stdout", "__stdoutp"),
                            ("stderr", "__stderrp")):
            binding = getattr(stdio, name)
            self.assertIsInstance(binding, ExternGlobal)
            self.assertIs(binding, getattr(stdio, alias))
            self.assertFalse(callable(binding))
            self.assertEqual(binding.c_name, alias if IS_MACOS else name)

    def test_errno_names_have_one_native_binding_and_one_shim(self):
        native = "__error" if IS_MACOS else "__errno_location"
        shim = "__errno_location" if IS_MACOS else "__error"
        self.assertTrue(getattr(errno, native)._is_extern)
        self.assertTrue(getattr(errno, shim)._is_compiled)

    def test_locale_categories_match_the_host(self):
        for name in pc_locale.__all__:
            if name.startswith("LC_"):
                self.assertEqual(getattr(pc_locale, name), getattr(locale, name))

    def test_underscored_classifiers_are_star_exports(self):
        names = ("__inline_isinff", "__inline_isinfd", "__inline_isinfl",
                 "__inline_isnanf", "__inline_isnand", "__inline_isnanl")
        for name in names:
            self.assertIn(name, math.__all__)
            self.assertTrue(callable(getattr(math, name)))
        self.assertEqual(len(math.__all__), len(set(math.__all__)))
        for name in math.__all__:
            self.assertTrue(hasattr(math, name))

    def test_wide_character_signatures_keep_typed_output_pointers(self):
        self.assertIs(dict(wchar.wcschr.param_types)["c"], wchar_t)
        expected = ptr[ptr[wchar_t]].get_llvm_type()
        self.assertEqual(
            dict(wchar.wcstol.param_types)["endptr"].get_llvm_type(), expected,
        )
        self.assertEqual(
            dict(wchar.wcstok.param_types)["ptr_"].get_llvm_type(), expected,
        )
        self.assertIs(stdio.getc.return_type, i32)


if __name__ == "__main__":
    unittest.main()
