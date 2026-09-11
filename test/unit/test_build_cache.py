"""Unit tests for BuildCache compiler-stamp invalidation.

Regression: cached .o files were keyed on user-source mtimes only, so
editing pythoc's own compiler code left stale artifacts behind that were
silently served as cache hits, masking (or faking) compiler regressions.
An .o must now be newer than both its source and the compiler stamp.
"""

import os
import tempfile
import time
import unittest

from pythoc.build.cache import BuildCache
from pythoc.utils.compiler_stamp import get_compiler_mtime


class TestCheckObjUptodateCompilerStamp(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pythoc_build_cache_')
        self.source_file = os.path.join(self.tmpdir, 'mod.py')
        with open(self.source_file, 'w') as f:
            f.write('# fake source\n')
        self.obj_file = os.path.join(self.tmpdir, 'mod.o')
        with open(self.obj_file, 'wb') as f:
            f.write(b'fake object')

    def test_compiler_stamp_is_present_and_finite(self):
        self.assertGreater(get_compiler_mtime(), 0.0)

    def test_obj_older_than_compiler_is_stale(self):
        # Newer than nothing relevant: backdated well before any pythoc edit.
        os.utime(self.source_file, (1000000, 1000000))
        os.utime(self.obj_file, (1000000, 1000000))
        self.assertFalse(
            BuildCache.check_obj_uptodate(self.obj_file, self.source_file)
        )

    def test_obj_newer_than_source_and_compiler_is_fresh(self):
        future = time.time() + 10
        os.utime(self.source_file, (1000000, 1000000))
        os.utime(self.obj_file, (future, future))
        self.assertTrue(
            BuildCache.check_obj_uptodate(self.obj_file, self.source_file)
        )

    def test_missing_obj_is_stale(self):
        os.remove(self.obj_file)
        self.assertFalse(
            BuildCache.check_obj_uptodate(self.obj_file, self.source_file)
        )


class TestPinWin32DllImplib(unittest.TestCase):
    """The win32 DLL link line must pin -implib to <output>.lib.

    Regression: zig derives the import library name from an input file
    when source files (e.g. .S) appear on the link line, so <output>.lib
    was never written and check_so_needs_relink kept reporting the DLL as
    stale.  A second execute_function call for the same group then tried
    to relink a DLL already loaded in the process, and lld-link failed
    with Permission denied (test_runtime_mem_pool on Windows CI).
    """

    def _expected_flag(self, output_file):
        implib = os.path.splitext(os.path.abspath(output_file))[0] + '.lib'
        return f'-Wl,-implib,{implib}'

    def test_flag_inserted_before_output(self):
        from pythoc.utils.link_utils import _pin_win32_dll_implib
        cmd = ['zig', 'cc', '-shared', 'a.o', 'a.exports.def', '-o',
               'a.dll', 'dep.lib', 'ctx.S']
        out = _pin_win32_dll_implib(cmd, 'a.dll')
        flag = self._expected_flag('a.dll')
        self.assertIn(flag, out)
        self.assertLess(out.index(flag), out.index('-o'))
        # Original arguments are preserved.
        for arg in cmd:
            self.assertIn(arg, out)

    def test_idempotent(self):
        from pythoc.utils.link_utils import _pin_win32_dll_implib
        cmd = ['zig', 'cc', '-shared', 'a.o', '-o', 'a.dll']
        once = _pin_win32_dll_implib(cmd, 'a.dll')
        twice = _pin_win32_dll_implib(once, 'a.dll')
        self.assertEqual(once, twice)

    def test_appended_when_no_output_flag(self):
        from pythoc.utils.link_utils import _pin_win32_dll_implib
        cmd = ['zig', 'cc', '-shared', 'a.o']
        out = _pin_win32_dll_implib(cmd, 'a.dll')
        self.assertEqual(out[-1], self._expected_flag('a.dll'))


if __name__ == '__main__':
    unittest.main()
