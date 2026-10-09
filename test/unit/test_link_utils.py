"""Unit tests for link command construction in utils.link_utils."""

import os
import sys
import tempfile
import threading
import time
import unittest

from pythoc.utils.link_utils import (
    _errno_slot_accessor,
    _preserve_c_errno,
    build_link_command,
    file_lock,
)


class TestBuildLinkCommandExtraFlags(unittest.TestCase):
    def test_extra_flags_appended_last(self):
        cmd = build_link_command(
            ['a.o'], 'out', link_libraries=[],
            extra_flags=['-Wl,-dead_strip', '-v'],
        )
        self.assertEqual(cmd[-2:], ['-Wl,-dead_strip', '-v'])

    def test_extra_flags_default_off(self):
        with_flags = build_link_command(
            ['a.o'], 'out', link_libraries=[], extra_flags=['-v'],
        )
        without_flags = build_link_command(['a.o'], 'out', link_libraries=[])
        self.assertEqual(with_flags, without_flags + ['-v'])

    def test_extra_flags_shared(self):
        cmd = build_link_command(
            ['a.o'], 'out.so', shared=True, link_libraries=[],
            extra_flags=['-undefined', 'dynamic_lookup'],
        )
        self.assertEqual(cmd[-2:], ['-undefined', 'dynamic_lookup'])


@unittest.skipIf(sys.platform == 'win32', 'errno slot checked on POSIX only')
class TestFileLockPreservesErrno(unittest.TestCase):
    def setUp(self):
        self.accessor = _errno_slot_accessor()
        if self.accessor is None:
            self.skipTest('platform errno slot not reachable')

    def _errno(self):
        return self.accessor().contents.value

    def _hold_lock(self, path, hold_seconds):
        if sys.platform == 'win32':
            import msvcrt

            def lock(fd):
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

            def unlock(fd):
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            def lock(fd):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

            def unlock(fd):
                fcntl.flock(fd, fcntl.LOCK_UN)

        other = open(path, 'a')
        lock(other.fileno())

        def release_soon():
            time.sleep(hold_seconds)
            unlock(other.fileno())
            other.close()

        threading.Thread(target=release_soon).start()

    def test_preserve_c_errno_restores(self):
        self.accessor().contents.value = 4
        with _preserve_c_errno():
            self.accessor().contents.value = 11
        self.assertEqual(self._errno(), 4)

    def test_contended_file_lock_leaves_errno_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            lockpath = os.path.join(tmp, 'x.lock')
            self._hold_lock(lockpath, hold_seconds=0.3)
            self.accessor().contents.value = 0
            with file_lock(lockpath, timeout=30):
                pass
            self.assertEqual(self._errno(), 0)


if __name__ == '__main__':
    unittest.main()
