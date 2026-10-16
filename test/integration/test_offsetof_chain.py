"""Nested offsets must agree with the addresses used by compiled code."""

import ctypes
import unittest

from pythoc import compile, i8, i32, i64, offsetof, ptr, static, u64, union


@compile
class OffsetInner:
    tag: i8
    value: i64


@union
class OffsetChoice:
    integer: i32
    record: OffsetInner


@compile
class OffsetOuter:
    prefix: i32
    choice: OffsetChoice


class CInner(ctypes.Structure):
    _fields_ = [("tag", ctypes.c_int8), ("value", ctypes.c_int64)]


class CChoice(ctypes.Union):
    _fields_ = [("integer", ctypes.c_int32), ("record", CInner)]


class COuter(ctypes.Structure):
    _fields_ = [("prefix", ctypes.c_int32), ("choice", CChoice)]


@compile
def nested_offset() -> u64:
    return offsetof(OffsetOuter, "choice.record.value")


@compile
def union_member_offset() -> u64:
    return offsetof(OffsetChoice, "record")


@compile
def nested_offset_address_agrees() -> i32:
    value: OffsetOuter
    value.prefix = 17
    value.choice.record.tag = i8(3)
    value.choice.record.value = i64(123)
    base: ptr[i8] = ptr[i8](ptr(value))
    member: ptr[i8] = base + offsetof(OffsetOuter, "choice.record.value")
    if ptr[i64](member)[0] == 123 and value.prefix == 17:
        return 1
    return 0


@compile
def static_nested_offset() -> u64:
    offsets: static[u64] = offsetof(OffsetOuter, "choice.record.value")
    return offsets


class TestOffsetofChain(unittest.TestCase):
    def test_nested_offset_matches_ctypes(self):
        self.assertEqual(
            nested_offset(), COuter.choice.offset + CInner.value.offset
        )

    def test_union_member_has_zero_offset(self):
        self.assertEqual(union_member_offset(), 0)

    def test_nested_offset_matches_real_member_address(self):
        self.assertEqual(nested_offset_address_agrees(), 1)

    def test_nested_offset_remains_a_static_constant(self):
        self.assertEqual(
            static_nested_offset(), COuter.choice.offset + CInner.value.offset
        )


if __name__ == "__main__":
    unittest.main()
