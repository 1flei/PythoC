#!/usr/bin/env python3
"""Platform-specific libc binding tests.

These tests verify that libc bindings compile and link correctly on the
current platform.  They are expected to run on macOS, Linux and Windows CI
runners; expected values differ by platform where noted.
"""

import importlib
import platform
import sys
import unittest

from pythoc import compile, i32, sizeof, offsetof
from pythoc.libc.signal import raise_, siginfo_t, sigset_t, sigaction
from pythoc.libc.time import timeval, tm, time_t
from pythoc.libc.ucontext import ucontext_t
from pythoc.libc.sys_stat import stat, stat_
from pythoc.libc.sys_resource import rusage
from pythoc.libc.pthread import pthread_mutex_t, PTHREAD_MUTEX_RECURSIVE
from pythoc.libc.fcntl import flock


IS_ARM64 = platform.machine().lower() in ('arm64', 'aarch64')
IS_POSIX = sys.platform != 'win32'


if IS_POSIX:
    # dirent / cassert refuse to import on Windows; setenv / gethostname
    # have no UCRT symbol.  Guard the imports so collection works everywhere.
    from pythoc import array, i8, nullptr, ptr
    from pythoc.libc.dirent import DIR, dirent, opendir, readdir, closedir
    from pythoc.libc.stdlib import getenv, setenv, unsetenv
    from pythoc.libc.unistd import gethostname


    @compile
    def test_dirent_size() -> i32:
        """Return sizeof(dirent) to verify platform layout."""
        return sizeof(dirent)


    @compile
    def test_dirent_d_name_offset() -> i32:
        """Return offsetof(dirent, d_name) to verify field layout."""
        return offsetof("dirent", "d_name")


    @compile
    def test_opendir_readdir() -> i32:
        """Open '.', read one entry, close; 0 on full success."""
        d: ptr[DIR] = opendir(".")
        if d == nullptr:
            return 1
        e: ptr[dirent] = readdir(d)
        if e == nullptr:
            return 2
        return closedir(d)


    @compile
    def test_setenv_roundtrip() -> i32:
        """setenv/getenv/unsetenv roundtrip; 0 on success."""
        if setenv("PYTHOC_TEST_ENV_VAR", "42", 1) != 0:
            return 1
        v: ptr[i8] = getenv("PYTHOC_TEST_ENV_VAR")
        if v == nullptr:
            return 2
        if v[0] != 52 or v[1] != 50:  # '4', '2'
            return 3
        return unsetenv("PYTHOC_TEST_ENV_VAR")


    @compile
    def test_gethostname_call() -> i32:
        """gethostname into a stack buffer; 0 on success."""
        buf: array[i8, 256] = ""
        return gethostname(buf, 256)


@compile
def test_raise_null_signal() -> i32:
    """Verify raise_ maps to the C raise symbol.

    raise(0) is a null signal: it performs error checking but does not
    actually send a signal.  Returns 0 on success.
    """
    return raise_(0)


@compile
def test_timeval_size() -> i32:
    """Return sizeof(timeval) to verify platform layout."""
    return sizeof(timeval)


@compile
def test_tm_size() -> i32:
    """Return sizeof(tm) to verify platform layout."""
    return sizeof(tm)


@compile
def test_ucontext_size() -> i32:
    """Return sizeof(ucontext_t) to verify platform layout."""
    return sizeof(ucontext_t)


@compile
def test_stat_size() -> i32:
    """Return sizeof(stat) to verify platform layout."""
    return sizeof(stat)


@compile
def test_stat_st_size_offset() -> i32:
    """Return offsetof(stat, st_size) to verify field layout."""
    return offsetof("stat", "st_size")


@compile
def test_rusage_size() -> i32:
    """Return sizeof(rusage); getrusage writes the full 144-byte struct."""
    return sizeof(rusage)


@compile
def test_pthread_mutex_size() -> i32:
    """Return sizeof(pthread_mutex_t) to verify platform layout."""
    return sizeof(pthread_mutex_t)


@compile
def test_flock_size() -> i32:
    """Return sizeof(flock) to verify platform layout."""
    return sizeof(flock)


class TestLibcPlatform(unittest.TestCase):
    @unittest.skipIf(sys.platform == 'win32',
                     'raise(0) is a POSIX null-signal probe; the UCRT treats '
                     '0 as an invalid parameter and terminates the process')
    def test_raise_null_signal(self):
        """raise_(0) should succeed on POSIX platforms."""
        self.assertEqual(test_raise_null_signal(), 0)

    def test_timeval_layout(self):
        size = test_timeval_size()
        if sys.platform == 'darwin':
            # macOS: time_t tv_sec (i64) + suseconds_t tv_usec (i32) + 4 pad
            self.assertEqual(size, 16)
        elif sys.platform.startswith('linux'):
            # glibc: time_t tv_sec (i64) + suseconds_t tv_usec (i64)
            self.assertEqual(size, 16)
        elif sys.platform == 'win32':
            # MSVCRT: time_t tv_sec (i64) + long tv_usec (i32)
            self.assertEqual(size, 12)

    def test_tm_layout(self):
        size = test_tm_size()
        if sys.platform == 'win32':
            # MSVCRT: 9 * i32
            self.assertEqual(size, 36)
        else:
            # BSD/glibc: 9*i32 + tm_gmtoff (i64) + tm_zone (ptr)
            self.assertEqual(size, 56)

    def test_ucontext_layout(self):
        size = test_ucontext_size()
        if sys.platform == 'darwin':
            # macOS ucontext_t is 56 bytes on both x86_64 and ARM64
            self.assertEqual(size, 56)
        elif sys.platform.startswith('linux'):
            # Linux layouts are much larger due to inline fpstate / regspace
            self.assertGreater(size, 500)
        else:
            # Windows opaque fallback
            self.assertEqual(size, 1)

    def test_signal_types_defined(self):
        """Signal types should always be defined (possibly opaque)."""
        self.assertIsNotNone(siginfo_t)
        self.assertIsNotNone(sigset_t)
        self.assertIsNotNone(sigaction)

    def test_stat_layout(self):
        size = test_stat_size()
        st_size_off = test_stat_st_size_offset()
        if sys.platform == 'darwin':
            # macOS 64-bit inode layout: 144 bytes, st_size after 4 timespecs
            self.assertEqual(size, 144)
            self.assertEqual(st_size_off, 96)
        elif sys.platform.startswith('linux'):
            # glibc: 144 bytes on x86_64, 128 on aarch64; st_size at 48
            self.assertEqual(size, 128 if IS_ARM64 else 144)
            self.assertEqual(st_size_off, 48)
        # Windows uses an opaque placeholder; no fixed expectation.

    def test_stat_function_renamed(self):
        """'stat' is the struct; the C function is exported as 'stat_'."""
        self.assertTrue(getattr(stat, '_is_struct', False))
        self.assertEqual(stat_.c_name, 'stat')

    def test_rusage_layout(self):
        # Full 144-byte struct on macOS/glibc; opaque placeholder matches.
        self.assertEqual(test_rusage_size(), 144)

    def test_pthread_mutex_layout(self):
        size = test_pthread_mutex_size()
        if sys.platform == 'darwin':
            self.assertEqual(size, 64)
            self.assertEqual(PTHREAD_MUTEX_RECURSIVE, 2)
        elif sys.platform.startswith('linux'):
            self.assertEqual(size, 40)
            self.assertEqual(PTHREAD_MUTEX_RECURSIVE, 1)

    def test_flock_layout(self):
        size = test_flock_size()
        if sys.platform == 'darwin':
            # BSD order: l_start, l_len, l_pid, l_type, l_whence
            self.assertEqual(size, 24)
        elif sys.platform.startswith('linux'):
            # glibc order: l_type, l_whence first, tail padding to 8
            self.assertEqual(size, 32)

    @unittest.skipIf(not IS_POSIX, '<dirent.h> is POSIX-only')
    def test_dirent_layout(self):
        size = test_dirent_size()
        d_name_off = test_dirent_d_name_offset()
        if sys.platform == 'darwin':
            # ino + seekoff + reclen + namlen + type + d_name[1024], 8-aligned
            self.assertEqual(size, 1048)
            self.assertEqual(d_name_off, 21)
        elif sys.platform.startswith('linux'):
            # glibc: ino + off + reclen + type + d_name[256], 8-aligned
            self.assertEqual(size, 280)
            self.assertEqual(d_name_off, 19)

    @unittest.skipIf(not IS_POSIX, '<dirent.h> is POSIX-only')
    def test_opendir_readdir(self):
        self.assertEqual(test_opendir_readdir(), 0)

    @unittest.skipIf(not IS_POSIX, 'setenv/gethostname have no UCRT symbol')
    def test_setenv_roundtrip(self):
        self.assertEqual(test_setenv_roundtrip(), 0)

    @unittest.skipIf(not IS_POSIX, 'setenv/gethostname have no UCRT symbol')
    def test_gethostname_call(self):
        self.assertEqual(test_gethostname_call(), 0)

    def test_cassert_symbol(self):
        """cassert exposes the platform assertion handler, or refuses to import."""
        if sys.platform == 'darwin':
            m = importlib.import_module('pythoc.libc.cassert')
            self.assertEqual(getattr(m, '__assert_rtn').c_name, '__assert_rtn')
        elif sys.platform.startswith('linux'):
            m = importlib.import_module('pythoc.libc.cassert')
            self.assertEqual(getattr(m, '__assert_fail').c_name, '__assert_fail')
        else:
            with self.assertRaises(NotImplementedError):
                importlib.import_module('pythoc.libc.cassert')


if __name__ == '__main__':
    unittest.main()
