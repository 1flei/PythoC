"""An unchanged import table must not force replacement of a mapped DLL."""

import os
from pathlib import Path
import tempfile
import unittest

from pythoc.build.cache import BuildCache
from pythoc.utils.link_utils import _win32_dll_uptodate


class WindowsDllFreshnessTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix='pc-dll-freshness-')
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.obj = self.file('kernel.o', 200)
        self.dll = self.file('kernel.dll', 300)
        self.definition = self.file('kernel.exports.def', 100)
        self.implib = self.file('kernel.lib', 100)

    def file(self, name, timestamp):
        path = self.root / name
        path.write_bytes(b'test artifact')
        os.utime(path, (timestamp, timestamp))
        return str(path)

    def test_unchanged_import_table_does_not_force_relink(self):
        self.assertFalse(BuildCache.check_so_needs_relink(self.dll, [self.obj]))
        self.assertTrue(_win32_dll_uptodate(self.dll, [self.obj]))

    def test_new_object_still_requires_relink(self):
        os.utime(self.obj, (400, 400))
        self.assertTrue(BuildCache.check_so_needs_relink(self.dll, [self.obj]))
        self.assertFalse(_win32_dll_uptodate(self.dll, [self.obj]))

    def test_missing_import_library_requires_repair(self):
        os.remove(self.implib)
        self.assertTrue(BuildCache.check_so_needs_relink(self.dll, [self.obj]))
        self.assertFalse(_win32_dll_uptodate(self.dll, [self.obj]))

    def test_missing_export_definition_requires_repair(self):
        os.remove(self.definition)
        self.assertTrue(BuildCache.check_so_needs_relink(self.dll, [self.obj]))
        self.assertFalse(_win32_dll_uptodate(self.dll, [self.obj]))


if __name__ == '__main__':
    unittest.main()
