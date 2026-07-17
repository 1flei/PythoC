#!/usr/bin/env python3
"""Platform-specific libc binding tests.

These tests verify that libc bindings compile and link correctly on the
current platform.  They are expected to run on macOS, Linux and Windows CI
runners; expected values differ by platform where noted.
"""

import sys
import unittest

from pythoc import compile, i32, sizeof
from pythoc.libc.signal import raise_, siginfo_t, sigset_t, sigaction
from pythoc.libc.time import timeval, tm, time_t
from pythoc.libc.ucontext import ucontext_t


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


class TestLibcPlatform(unittest.TestCase):
    def test_raise_null_signal(self):
        """raise_(0) should succeed on all POSIX platforms and Windows."""
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


if __name__ == '__main__':
    unittest.main()
