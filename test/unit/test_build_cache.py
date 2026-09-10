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


if __name__ == '__main__':
    unittest.main()
