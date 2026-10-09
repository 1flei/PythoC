"""Unit tests for Windows link inputs of the Python call/adapter path.

Windows DLLs resolve every symbol at link time, so the runtime and the
adapters must link the pythonXY import library, and kernel references
must go through the group DLL's .lib import library.  Posix platforms
must NOT link libpython at all (a statically linked interpreter plus a
linked libpython loads a second, uninitialized runtime into the process).
"""

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pythoc.python_adapter import _kernel_link_input, _python_libraries

_WIN32_SYSCONFIG = SimpleNamespace(
    get_config_var=lambda key: {
        'LIBDIR': None,
        'LDVERSION': None,
        'VERSION': '3.12',
        'py_version_nodot': '312',
    }.get(key),
)


class TestPythonLibraries(unittest.TestCase):
    def test_posix_never_links_libpython(self):
        for platform in ('darwin', 'linux'):
            with patch('pythoc.python_adapter.sys.platform', platform):
                self.assertEqual(_python_libraries(), [])

    def test_windows_links_python_import_library(self):
        with patch('pythoc.python_adapter.sys.platform', 'win32'), \
                patch('pythoc.python_adapter.sysconfig', _WIN32_SYSCONFIG):
            flags = _python_libraries()
        self.assertEqual(len(flags), 2)
        expected_dir = os.path.join(sys.base_prefix, 'libs')
        self.assertEqual(flags[0], '-L{}'.format(expected_dir))
        self.assertEqual(flags[1], '-lpython312')

    def test_windows_flags_have_no_none(self):
        # Regression guard for '-LNone' / '-lpythonNone'.
        with patch('pythoc.python_adapter.sys.platform', 'win32'), \
                patch('pythoc.python_adapter.sysconfig', _WIN32_SYSCONFIG):
            flags = _python_libraries()
        for flag in flags:
            self.assertNotIn('None', flag)


class TestKernelLinkInput(unittest.TestCase):
    def test_posix_keeps_dll_path(self):
        with patch('pythoc.python_adapter.sys.platform', 'darwin'):
            self.assertEqual(
                _kernel_link_input('/x/mod_1.so'), '/x/mod_1.so')

    def test_windows_prefers_import_library(self):
        with tempfile.TemporaryDirectory() as d:
            dll = os.path.join(d, 'mod_1.dll')
            implib = os.path.join(d, 'mod_1.lib')
            open(dll, 'w').close()
            with patch('pythoc.python_adapter.sys.platform', 'win32'):
                self.assertEqual(_kernel_link_input(dll), dll)
                open(implib, 'w').close()
                self.assertEqual(_kernel_link_input(dll), implib)


if __name__ == '__main__':
    unittest.main()
