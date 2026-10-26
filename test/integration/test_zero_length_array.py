"""Zero-length arrays (``array[T, 0]``), the C zero-length array extension.

C (GCC extension) allows a struct field declared as ``T name[0]``. Such a
field contributes its element alignment to the struct but no storage:

    struct A { char a; int b[0]; };   /* sizeof 4, align 4 */
    struct D { char pad[0]; };        /* sizeof 0, align 1 */

CPython's ``pycore_qsbr.h`` uses this to pad a struct to a fixed size::

    struct _qsbr_pad {
        struct _qsbr_thread_state ts;
        char __padding[64 - sizeof(struct _qsbr_thread_state)];
    };

When the expression folds to zero, the emitted PythoC annotation is
``'array[i8, (64 - sizeof(...))]'``; before this fix the type-subscript
guard rejected ``dim == 0`` outright, blocking every TU that includes the
header. The guard now only rejects negative dimensions, and struct layout
counts zero-sized fields for alignment but not for storage.
"""

import ctypes
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from pythoc import (
    array, compile, i8, i32, ptr, sizeof, struct, u8, u32, u64,
)
from pythoc.logger import logger
from pythoc.utils.cc_utils import find_available_cc


@compile
class PadOnly:
    """Struct consisting solely of a zero-length array: size 0, align 1."""
    pad: array[i8, 0]


@compile
class CharThenZeroIntArray:
    """char a; int b[0];  -> size 4, align 4 (alignment survives, size not)."""
    a: i8
    b: array[u32, 0]


@compile
class ZeroArrayThenLong:
    """int a[0]; long b;  -> size 8, align 8."""
    a: array[u32, 0]
    b: u64


@compile
class LongThenZeroArray:
    """long b; int a[0];  -> size 8, align 8."""
    b: u64
    a: array[u32, 0]


@compile
class ThreadState64:
    """Mirror of the qsbr scenario: exactly 64 bytes."""
    head: u64
    local: u32
    shared: u32
    mode: u32
    flags: u32
    seq: u64
    idx: u32
    reserved: array[u8, 28]


@compile
class QsbrPad:
    """The distillation: padding dimension folds to zero via sizeof."""
    ts: ThreadState64
    __padding: 'array[i8, (64 - sizeof(ThreadState64))]'


@compile
def sizeof_thread_state() -> u64:
    return sizeof(ThreadState64)


@compile
def sizeof_qsbr_pad() -> u64:
    return sizeof(QsbrPad)


@compile
def access_after_zero_field(p: ptr[CharThenZeroIntArray]) -> i8:
    """Access an ordinary field adjacent to a zero-length field."""
    return p.a


@compile
def zero_field_offset(p: ptr[CharThenZeroIntArray]) -> u64:
    return u64(ptr(p.b)) - u64(p)


@compile
def construct_zero_array() -> u64:
    value: array[i32, 0] = array[i32, 0]()
    return sizeof(array[i32, 0])


class TestZeroLengthArray(unittest.TestCase):

    def test_pad_only_size_and_align(self):
        self.assertEqual(PadOnly.get_size_bytes(), 0)
        self.assertEqual(PadOnly.get_alignment(), 1)

    def test_char_then_zero_int_array(self):
        self.assertEqual(CharThenZeroIntArray.get_size_bytes(), 4)
        self.assertEqual(CharThenZeroIntArray.get_alignment(), 4)

    def test_zero_array_then_long(self):
        self.assertEqual(ZeroArrayThenLong.get_size_bytes(), 8)
        self.assertEqual(ZeroArrayThenLong.get_alignment(), 8)

    def test_long_then_zero_array(self):
        self.assertEqual(LongThenZeroArray.get_size_bytes(), 8)
        self.assertEqual(LongThenZeroArray.get_alignment(), 8)

    def test_negative_dimension_rejected(self):
        with patch.object(logger, 'raise_on_error', True):
            with self.assertRaises(TypeError):
                array[i8, -1]

    def test_qsbr_shape_ground_truth(self):
        """The cpython pycore_qsbr.h distillation: 64-byte struct padded
        with a zero-length array stays 64 bytes, identical to C."""
        self.assertEqual(sizeof_thread_state(), 64)
        self.assertEqual(sizeof_qsbr_pad(), 64)

    def test_field_access_through_pointer(self):
        buf = (ctypes.c_char * 8)()
        address = ctypes.cast(buf, ctypes.c_void_p)
        self.assertEqual(access_after_zero_field(address), 0)
        self.assertEqual(zero_field_offset(address), 4)

    def test_local_zero_array_construction(self):
        self.assertEqual(construct_zero_array(), 0)

    def test_layout_matches_independent_c_oracle(self):
        source = Path(__file__).parents[1] / 'c_oracles' / 'zero_length_array.c'
        with tempfile.TemporaryDirectory(prefix='pc-zero-layout-') as temp:
            executable = str(Path(temp) / 'oracle.exe')
            subprocess.run(shlex.split(find_available_cc()) + [
                '-std=c11', str(source), '-o', executable,
            ], check=True, capture_output=True, text=True)
            actual = subprocess.check_output(
                [executable], text=True).splitlines()
        middle = struct[i8, array[i32, 0], i8]
        types = (PadOnly, CharThenZeroIntArray, ZeroArrayThenLong,
                 LongThenZeroArray, middle, array[i32, 3, 0])
        expected = []
        for value in types:
            size, alignment = value.get_size_bytes(), value.get_alignment()
            expected.append(f'{size} {alignment}')
            ctype = value.get_ctypes_type()
            self.assertEqual((ctypes.sizeof(ctype), ctypes.alignment(ctype)),
                             (size, alignment))
        expected.append('{} {}'.format(*middle.get_layout().field_offsets[1:]))
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
