"""The errno guard and extern calls resolve the same host C runtime."""

import ctypes
import unittest
from unittest.mock import Mock, patch

from pythoc.decorators.extern import _load_lib_handle
from pythoc.utils.c_runtime import load_c_runtime
from pythoc.utils.link_utils import _errno_slot_accessor


class TestCRuntime(unittest.TestCase):
    def tearDown(self):
        load_c_runtime.cache_clear()
        _errno_slot_accessor.cache_clear()

    def test_windows_uses_ucrt_for_errno_and_extern_functions(self):
        load_c_runtime.cache_clear()
        _errno_slot_accessor.cache_clear()
        library = Mock()
        with patch('sys.platform', 'win32'), patch(
            'platform.system', return_value='Windows',
        ), patch('ctypes.CDLL', return_value=library) as load:
            self.assertIs(_errno_slot_accessor(), library._errno)
            self.assertIs(_load_lib_handle('c'), library)
            self.assertIs(_load_lib_handle('m'), library)
        load.assert_called_once_with('ucrtbase.dll')
        self.assertEqual(library._errno.restype,
                         ctypes.POINTER(ctypes.c_int))

    def test_posix_uses_the_process_native_errno_accessor(self):
        for platform, name in (('darwin', '__error'),
                               ('linux', '__errno_location')):
            with self.subTest(platform=platform):
                load_c_runtime.cache_clear()
                _errno_slot_accessor.cache_clear()
                library = Mock()
                with patch('sys.platform', platform), patch(
                    'ctypes.CDLL', return_value=library,
                ) as load:
                    self.assertIs(_errno_slot_accessor(),
                                  getattr(library, name))
                load.assert_called_once_with(None)


if __name__ == '__main__':
    unittest.main()
