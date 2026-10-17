"""Runtime behavior of the libc additions used by standalone programs."""

import ctypes
import locale
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import unittest

from pythoc import (
    array, compile, extern, f32, f64, i8, i32, nullptr, ptr, u64,
)
from pythoc.libc.errno import (
    __error as native_error, __errno_location as native_errno_location,
)
from pythoc.libc.locale import LC_CTYPE, setlocale
from pythoc.libc.math import (
    __inline_isinfd as isinf_double, __inline_isinff as isinf_float,
    __inline_isinfl as isinf_long, __inline_isnand as isnan_double,
    __inline_isnanf as isnan_float, __inline_isnanl as isnan_long, frexp,
)
from pythoc.libc.signal import kill
from pythoc.libc.stdio import (
    fclose, ferror, fputc, getc, perror, rewind, stderr, stdin, stdout,
)
from pythoc.libc.string import strerror
from pythoc.libc.wchar import wcschr, wcsncmp, wcstok, wcstol


@extern(lib="c")
def tmpfile() -> ptr[i8]:
    pass


@compile
def stream_address(index: i32) -> u64:
    if index == 0:
        return u64(stdin)
    if index == 1:
        return u64(stdout)
    return u64(stderr)


@compile
def stream_roundtrip() -> i32:
    stream = tmpfile()
    if stream == nullptr:
        return -1
    written = fputc(65, stream)
    rewind(stream)
    value = getc(stream)
    end = getc(stream)
    error = ferror(stream)
    closed = fclose(stream)
    return (i32(written == 65) + i32(value == 65) + i32(end == -1)
            + i32(error == 0) + i32(closed == 0))


@compile
def errno_roundtrip() -> i32:
    pointer = native_error()
    previous = pointer[0]
    pointer[0] = 123
    result = i32(native_errno_location() == pointer)
    result = result + native_errno_location()[0]
    pointer[0] = previous
    return result


@compile
def errno_address() -> u64:
    return u64(native_error())


@compile
def print_error() -> i32:
    pointer = native_error()
    previous = pointer[0]
    pointer[0] = 22
    perror("pc-link-support")
    pointer[0] = previous
    return 0


@compile
def signal_probe(pid: i32) -> i32:
    return kill(pid, 0)


@compile
def error_message(code: i32) -> u64:
    return u64(strerror(code))


@compile
def locale_query() -> u64:
    return u64(setlocale(LC_CTYPE, nullptr))


@compile
def frexp_roundtrip(value: f64) -> f64:
    exponent: i32 = 0
    fraction = frexp(value, ptr(exponent))
    return fraction + f64(exponent)


@compile
def classify_double(value: f64) -> i32:
    return i32(isinf_double(value) != 0) + 2 * i32(isnan_double(value) != 0)


@compile
def classify_float(value: f32) -> i32:
    return i32(isinf_float(value) != 0) + 2 * i32(isnan_float(value) != 0)


@compile
def classify_long(value: f64) -> i32:
    return i32(isinf_long(value) != 0) + 2 * i32(isnan_long(value) != 0)


@compile
def wide_parse() -> i32:
    text: array[i32, 5] = (49, 50, 51, 120, 0)
    end: ptr[i32] = nullptr
    value = wcstol(text, ptr(end), 10)
    found = wcschr(text, 120)
    return (i32(value) + i32(end[0] == 120)
            + i32(found == end) + i32(wcschr(text, 122) == nullptr))


@compile
def wide_compare() -> i32:
    left: array[i32, 4] = (97, 98, 120, 0)
    right: array[i32, 4] = (97, 98, 121, 0)
    return i32(wcsncmp(left, right, 2) == 0) + i32(wcsncmp(left, right, 3) < 0)


@compile
def wide_tokens() -> i32:
    text: array[i32, 6] = (97, 44, 98, 44, 99, 0)
    delimiters: array[i32, 2] = (44, 0)
    state: ptr[i32] = nullptr
    first = wcstok(text, delimiters, ptr(state))
    second = wcstok(nullptr, delimiters, ptr(state))
    third = wcstok(nullptr, delimiters, ptr(state))
    end = wcstok(nullptr, delimiters, ptr(state))
    return (first[0] + second[0] + third[0] + i32(end == nullptr)
            + i32(text[1] == 0) + i32(text[3] == 0))


class TestLibcLinkSupport(unittest.TestCase):
    def test_standard_streams_are_actual_host_pointers(self):
        names = ("__stdinp", "__stdoutp", "__stderrp") if (
            sys.platform == "darwin"
        ) else ("stdin", "stdout", "stderr")
        host = ctypes.CDLL(None)
        for index, name in enumerate(names):
            self.assertEqual(stream_address(index),
                             ctypes.c_void_p.in_dll(host, name).value)

    def test_getc_eof_and_resource_close(self):
        self.assertEqual(stream_roundtrip(), 5)

    def test_errno_aliases_and_thread_local_slots(self):
        self.assertEqual(errno_roundtrip(), 124)
        errno_address()
        addresses = []
        errors = []
        barrier = threading.Barrier(2)

        def worker():
            try:
                address = int(errno_address())
                barrier.wait(timeout=10)
                addresses.append(address)
                self.assertEqual(errno_roundtrip(), 124)
            except Exception as error:
                errors.append(error)

        workers = [threading.Thread(target=worker) for _ in range(2)]
        for worker_thread in workers:
            worker_thread.start()
        for worker_thread in workers:
            worker_thread.join(15)
            self.assertFalse(worker_thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(set(addresses)), 2)

    def test_perror_has_a_real_stderr_side_effect(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--perror-child"],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("pc-link-support:", result.stderr)
        self.assertIn(os.strerror(22), result.stderr)

    def test_kill_zero_only_probes_this_process(self):
        self.assertEqual(signal_probe(os.getpid()), 0)

    def test_strerror_matches_host(self):
        address = int(error_message(22))
        self.assertNotEqual(address, 0)
        self.assertEqual(ctypes.string_at(address).decode(), os.strerror(22))

    def test_locale_query_does_not_change_the_locale(self):
        self.assertEqual(ctypes.string_at(int(locale_query())).decode(),
                         locale.setlocale(locale.LC_CTYPE))

    def test_frexp_writes_the_exponent(self):
        for value in (12.0, -12.0, 0.0, 0.125):
            fraction, exponent = math.frexp(value)
            self.assertEqual(frexp_roundtrip(value), fraction + exponent)

    def test_float_classifiers(self):
        for function in (classify_double, classify_float, classify_long):
            for value in (0.0, -0.0, 1.0, math.inf, -math.inf, math.nan):
                expected = int(math.isinf(value)) + 2 * int(math.isnan(value))
                self.assertEqual(function(value), expected)

    def test_wide_string_search_and_endptr(self):
        self.assertEqual(wide_parse(), 126)

    def test_wide_string_bounded_compare(self):
        self.assertEqual(wide_compare(), 2)

    def test_wcstok_progress_and_input_mutation(self):
        self.assertEqual(wide_tokens(), 297)


if __name__ == "__main__":
    if "--perror-child" in sys.argv:
        raise SystemExit(int(print_error()))
    unittest.main()
