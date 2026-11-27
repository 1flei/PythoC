"""Named alignment affects record layout, native fields, and host FFI."""

import ctypes
import unittest

from pythoc import compile, ptr, struct, u8, u16, union, void


@struct(member_alignments={'word': 64})
class Record:
    leading: u8
    word: u16
    trailing: u8


@union(member_alignments={'word': 64})
class Alternative:
    byte: u8
    word: u16


@compile
def fill(record: ptr[Record]) -> void:
    record.leading = 0x51
    record.word = 0x1234
    record.trailing = 0x32


class MemberAlignmentTest(unittest.TestCase):
    def test_struct_member_alignment(self):
        self.assertEqual(Record.get_layout().field_offsets, (0, 8, 10))
        record_type = Record.get_ctypes_type()
        self.assertEqual((ctypes.sizeof(record_type), ctypes.alignment(record_type)), (16, 8))
        value = record_type()
        fill(ctypes.byref(value))
        self.assertEqual((value.leading, value.word, value.trailing), (0x51, 0x1234, 0x32))

    def test_union_member_alignment(self):
        self.assertEqual(Alternative.get_layout().field_offsets, (0, 0))
        alternative_type = Alternative.get_ctypes_type()
        self.assertEqual((ctypes.sizeof(alternative_type), ctypes.alignment(alternative_type)), (8, 8))


if __name__ == '__main__':
    unittest.main()
