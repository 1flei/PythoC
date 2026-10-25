"""Check Windows binding declarations using their real module sources."""

from contextlib import ExitStack
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from pythoc import i32, u16
from pythoc import libc
from pythoc.libc import _platform
from pythoc.session import CompileSession


class TestWindowsLibcBindings(unittest.TestCase):
    def test_windows_modules_import_with_native_widths_and_real_stream_api(self):
        modules = {}
        with CompileSession(), ExitStack() as stack:
            stack.enter_context(patch.multiple(
                _platform, IS_MACOS=False, IS_LINUX=False, IS_WINDOWS=True,
            ))
            for name in ('stddef', 'locale', 'stdio', 'wchar'):
                qualified = 'pythoc.libc.' + name
                source = Path(libc.__file__).with_name(name + '.py')
                spec = importlib.util.spec_from_file_location(qualified, source)
                module = importlib.util.module_from_spec(spec)
                stack.enter_context(patch.dict(sys.modules, {qualified: module}))
                stack.enter_context(patch.object(libc, name, module,
                                                create=True))
                spec.loader.exec_module(module)
                modules[name] = module
            self.assertIs(modules['stddef'].wchar_t, u16)
            self.assertIs(modules['wchar'].wcstol.return_type, i32)
            self.assertIs(modules['wchar'].wchar_t, u16)
            self.assertEqual(modules['locale'].LC_CTYPE, 2)
            self.assertNotIn('LC_MESSAGES', modules['locale'].__all__)
            self.assertNotIn('stdin', modules['stdio'].__all__)
            accessor = getattr(modules['stdio'], '__acrt_iob_func')
            self.assertEqual(accessor.c_name, '__acrt_iob_func')
            self.assertTrue(accessor._is_extern)
            self.assertTrue(modules['stdio'].standard_stream._is_compiled)


if __name__ == '__main__':
    unittest.main()
