"""Unit tests for cimport/extern platform helpers that integration tests
cannot reach on the local platform (Windows-specific branches)."""

import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pythoc.cimport_clang import (
    ClangCImportError, is_clang_backend_available,
)


class TestClangBackendAvailability(unittest.TestCase):
    def test_missing_bindings_are_unavailable(self):
        with patch('pythoc.cimport_clang._load_cindex',
                   side_effect=ClangCImportError('bindings missing')):
            self.assertFalse(is_clang_backend_available())

    def test_missing_native_library_is_unavailable(self):
        class LibclangError(Exception):
            pass

        cindex = SimpleNamespace(
            LibclangError=LibclangError,
            Index=SimpleNamespace(create=Mock(side_effect=LibclangError('missing'))),
        )
        with patch('pythoc.cimport_clang._load_cindex', return_value=cindex):
            self.assertFalse(is_clang_backend_available())

    def test_unexpected_loading_error_propagates(self):
        with patch('pythoc.cimport_clang._load_cindex',
                   side_effect=RuntimeError('unexpected loading error')):
            with self.assertRaisesRegex(RuntimeError, 'unexpected loading error'):
                is_clang_backend_available()

    def test_unexpected_index_error_propagates(self):
        class LibclangError(Exception):
            pass

        cindex = SimpleNamespace(
            LibclangError=LibclangError,
            Index=SimpleNamespace(create=Mock(side_effect=ValueError('bad index'))),
        )
        with patch('pythoc.cimport_clang._load_cindex', return_value=cindex):
            with self.assertRaisesRegex(ValueError, 'bad index'):
                is_clang_backend_available()

    def test_usable_index_is_available(self):
        cindex = SimpleNamespace(Index=SimpleNamespace(create=Mock()))
        with patch('pythoc.cimport_clang._load_cindex', return_value=cindex):
            self.assertTrue(is_clang_backend_available())


class TestNormalizeLibForGeneratedSource(unittest.TestCase):
    """On Windows, bare library names must never be abspath'ed into the
    generated bindings (that turned 'c' into '<cwd>/c' and broke every
    zig link line); only real paths get the backslash normalization."""

    def test_bare_names_pass_through_on_windows(self):
        from pythoc.cimport import _normalize_lib_for_generated_source
        # Patch the real os module: 'pythoc.cimport.os' is not importable
        # (pythoc.cimport is a module, not a package), so mock cannot
        # resolve it as a patch target.
        with patch("os.name", "nt"):
            self.assertEqual(_normalize_lib_for_generated_source('c'), 'c')
            self.assertEqual(_normalize_lib_for_generated_source('m'), 'm')
            self.assertEqual(_normalize_lib_for_generated_source('mylib'),
                             'mylib')
            self.assertEqual(_normalize_lib_for_generated_source(''), '')

    def test_paths_normalized_on_windows(self):
        from pythoc.cimport import _normalize_lib_for_generated_source
        with patch("os.name", "nt"):
            out = _normalize_lib_for_generated_source('subdir\\mylib')
            self.assertNotIn('\\', out)
            self.assertIn('/', out)

    def test_posix_passthrough(self):
        from pythoc.cimport import _normalize_lib_for_generated_source
        if os.name == 'nt':
            # A relative path with a backslash separator IS path-like on
            # Windows and gets normalized; passthrough is POSIX-only.
            self.skipTest('POSIX behavior')
        self.assertEqual(
            _normalize_lib_for_generated_source('some\\name'), 'some\\name')


class TestExternLibEmptyString(unittest.TestCase):
    """lib='' (process-global symbols from registered object files) is a
    distinct spec and must not be collapsed into 'c'."""

    def test_extern_preserves_empty_lib(self):
        from pythoc.decorators.extern import extern

        def f(a):
            return a

        self.assertEqual(extern(f, lib='').lib, '')
        self.assertEqual(extern(f).lib, 'c')
        self.assertEqual(extern(f, lib='m').lib, 'm')

    def test_empty_lib_python_side_raises_clear_error_on_windows(self):
        from pythoc.decorators.extern import _load_lib_handle

        with patch.object(sys, 'platform', 'win32'):
            with self.assertRaises(RuntimeError) as ctx:
                _load_lib_handle('')
        self.assertIn('not supported on Windows', str(ctx.exception))

    def test_empty_lib_python_side_loads_process_on_posix(self):
        from pythoc.decorators.extern import _load_lib_handle

        if sys.platform == 'win32':
            self.skipTest('POSIX behavior')
        # No registered link objects: resolves to the current process.
        handle = _load_lib_handle('')
        import ctypes
        self.assertIsInstance(handle, ctypes.CDLL)


if __name__ == '__main__':
    unittest.main()
