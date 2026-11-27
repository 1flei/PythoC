"""Unit tests for Windows link inputs of the Python call/adapter path.

Windows links the runtime and adapters against the pythonXY import
library. On POSIX they resolve CPython symbols from the running process.
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pythoc.python_adapter import _python_libraries

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


if __name__ == '__main__':
    unittest.main()
