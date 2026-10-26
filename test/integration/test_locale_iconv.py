"""Locale constants and native iconv success, error and reset behavior."""

import ctypes
import errno
import locale as host_locale
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

from pythoc import compile, i8, i32, nullptr, ptr, u64
from pythoc.libc.errno import errno_slot
from pythoc.libc.locale import localeconv
from pythoc.utils.cc_utils import find_available_cc


IS_POSIX = sys.platform == 'darwin' or sys.platform.startswith('linux')
SIZE_ERROR = (1 << 64) - 1


@compile
def lconv_address() -> u64:
    return u64(localeconv())


if IS_POSIX:
    from pythoc.libc import locale
    from pythoc.libc.iconv import iconv, iconv_close, iconv_open

    @compile
    def langinfo_address(item: i32) -> u64:
        return u64(locale.nl_langinfo(item))

    @compile
    def convert_bytes(source: ptr[i8], length: u64, output: ptr[i8],
                      capacity: u64, state: ptr[u64]) -> u64:
        descriptor = iconv_open('UTF-16LE', 'UTF-8')
        if u64(descriptor) == u64(0xffffffffffffffff):
            return u64(0xfffffffffffffffe)
        input_cursor: ptr[i8] = source
        output_cursor: ptr[i8] = output
        input_left: u64 = length
        output_left: u64 = capacity
        result = iconv(descriptor, ptr(input_cursor), ptr(input_left),
                       ptr(output_cursor), ptr(output_left))
        state[0] = u64(errno_slot()[0])
        state[1] = input_left
        state[2] = output_left
        state[3] = u64(input_cursor) - u64(source)
        state[4] = u64(output_cursor) - u64(output)
        state[5] = u64(iconv_close(descriptor))
        return result

    @compile
    def invalid_encoding() -> i32:
        descriptor = iconv_open('PYTHOC-NONEXISTENT-ENCODING', 'UTF-8')
        if u64(descriptor) != u64(0xffffffffffffffff):
            iconv_close(descriptor)
            return -1
        return errno_slot()[0]

    @compile
    def reset_conversion() -> i32:
        descriptor = iconv_open('UTF-16LE', 'UTF-8')
        if u64(descriptor) == u64(0xffffffffffffffff):
            return -1
        # NULL input resets conversion state; NULL output discards shift bytes.
        flushed = iconv(descriptor, nullptr, nullptr, nullptr, nullptr)
        input_cursor: ptr[i8] = nullptr
        reset = iconv(descriptor, ptr(input_cursor), nullptr, nullptr, nullptr)
        closed = iconv_close(descriptor)
        return i32(flushed == 0) + i32(reset == 0) + i32(closed == 0)


class TestLocaleNative(unittest.TestCase):
    def test_localeconv_returns_real_structure(self):
        self.assertNotEqual(lconv_address(), 0)

    @unittest.skipUnless(IS_POSIX, 'langinfo.h requires macOS or Linux')
    def test_nl_langinfo_matches_host(self):
        for name in ('CODESET', 'RADIXCHAR', 'THOUSEP', 'DAY_1', 'MON_12'):
            item = getattr(locale, name)
            actual = ctypes.string_at(int(langinfo_address(item)))
            expected = host_locale.nl_langinfo(getattr(host_locale, name))
            self.assertEqual(actual.decode(), expected, name)

    @unittest.skipUnless(IS_POSIX, 'POSIX locale header oracle')
    def test_constants_match_independent_c_headers(self):
        source = Path(__file__).parents[1] / 'c_oracles' / 'locale_items.c'
        with tempfile.TemporaryDirectory(prefix='pc-locale-items-') as temp:
            executable = str(Path(temp) / 'oracle.exe')
            subprocess.run(shlex.split(find_available_cc()) + [
                '-std=c11', str(source), '-o', executable,
            ], check=True, capture_output=True, text=True)
            actual = subprocess.check_output([executable], text=True)
        for line in actual.splitlines():
            name, value = line.split()
            self.assertEqual(getattr(locale, name), int(value), name)
            self.assertIn(name, locale.__all__)


@unittest.skipUnless(IS_POSIX, 'iconv requires macOS or Linux')
class TestIconvNative(unittest.TestCase):
    def convert(self, source, capacity):
        input_buffer = ctypes.create_string_buffer(source)
        output = ctypes.create_string_buffer(capacity)
        state = (ctypes.c_uint64 * 6)()
        result = int(convert_bytes(
            ctypes.cast(input_buffer, ctypes.c_void_p), len(source),
            ctypes.cast(output, ctypes.c_void_p), capacity,
            ctypes.cast(state, ctypes.c_void_p),
        ))
        self.assertEqual(state[5], 0)
        return result, output.raw, tuple(state)

    def test_non_ascii_cross_encoding_and_cursor_updates(self):
        result, output, state = self.convert(b'\xc3\xa9\xe4\xb8\xad', 8)
        self.assertEqual(result, 0)
        self.assertEqual(output[:4], b'\xe9\x00\x2d\x4e')
        self.assertEqual(state[1:5], (0, 4, 5, 4))

    def test_output_too_small(self):
        result, output, state = self.convert(b'\xc3\xa9', 1)
        self.assertEqual(result, SIZE_ERROR)
        self.assertEqual(state[:5], (errno.E2BIG, 2, 1, 0, 0))
        self.assertEqual(output, b'\x00')

    def test_invalid_and_incomplete_input(self):
        for source, error in ((b'\xff', errno.EILSEQ), (b'\xc3', errno.EINVAL)):
            with self.subTest(source=source):
                result, _, state = self.convert(source, 8)
                self.assertEqual(result, SIZE_ERROR)
                self.assertEqual(state[:5], (error, len(source), 8, 0, 0))

    def test_invalid_encoding_name(self):
        self.assertEqual(invalid_encoding(), errno.EINVAL)

    def test_flush_and_reset(self):
        self.assertEqual(reset_conversion(), 3)


if __name__ == '__main__':
    unittest.main()
