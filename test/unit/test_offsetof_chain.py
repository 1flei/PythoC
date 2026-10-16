"""Nested member offsets and invalid paths use the same field lookup."""

import ctypes
import unittest
from unittest.mock import patch

from pythoc import compile, i8, i32, i64, offsetof, ptr, struct, union
from pythoc.logger import logger


Inner = struct["tag": i8, "value": i64]
Choice = union["integer": i32, "record": Inner]
Middle = struct["prefix": i32, "choice": Choice]
Outer = struct["marker": i8, "middle": Middle, "tail": i32]


class CInner(ctypes.Structure):
    _fields_ = [("tag", ctypes.c_int8), ("value", ctypes.c_int64)]


class CChoice(ctypes.Union):
    _fields_ = [("integer", ctypes.c_int32), ("record", CInner)]


class CMiddle(ctypes.Structure):
    _fields_ = [("prefix", ctypes.c_int32), ("choice", CChoice)]


class COuter(ctypes.Structure):
    _fields_ = [
        ("marker", ctypes.c_int8),
        ("middle", CMiddle),
        ("tail", ctypes.c_int32),
    ]


@union
class DecoratedChoice:
    integer: i32
    record: Inner


class TestOffsetofChain(unittest.TestCase):
    def test_single_member_offset_is_unchanged(self):
        self.assertEqual(
            offsetof._get_field_offset(Outer, "tail"), COuter.tail.offset
        )

    def test_nested_struct_offsets_include_padding(self):
        self.assertEqual(
            offsetof._get_field_offset(Outer, "middle.choice"),
            COuter.middle.offset + CMiddle.choice.offset,
        )

    def test_union_member_at_the_end_has_zero_local_offset(self):
        self.assertEqual(
            offsetof._get_field_offset(Outer, "middle.choice.record"),
            COuter.middle.offset + CMiddle.choice.offset,
        )

    def test_chain_continues_through_union_into_struct(self):
        self.assertEqual(
            offsetof._get_field_offset(Outer, "middle.choice.record.value"),
            COuter.middle.offset + CMiddle.choice.offset + CInner.value.offset,
        )

    def test_decorated_union_uses_the_same_offsets(self):
        self.assertEqual(
            offsetof._get_field_offset(DecoratedChoice, "record"), 0
        )
        self.assertEqual(
            offsetof._get_field_offset(DecoratedChoice, "record.value"),
            CInner.value.offset,
        )

    def test_missing_members_are_rejected_even_in_unions(self):
        for path in (
            "missing",
            "middle.missing",
            "middle.choice.missing",
            "middle.choice.missing.value",
            "middle.choice.record.missing",
        ):
            with self.subTest(path=path):
                with patch.object(logger, "raise_on_error", True):
                    with self.assertRaises(AttributeError):
                        offsetof._get_field_offset(Outer, path)

    def test_empty_path_components_are_rejected(self):
        for path in ("", ".middle", "middle.", "middle..choice"):
            with self.subTest(path=path):
                with patch.object(logger, "raise_on_error", True):
                    with self.assertRaises(AttributeError):
                        offsetof._get_field_offset(Outer, path)

    def test_chain_cannot_traverse_a_scalar_or_pointer(self):
        holder = struct["scalar": i32, "pointer": ptr[Inner]]
        for path in ("scalar.value", "pointer.value"):
            with self.subTest(path=path):
                with patch.object(logger, "raise_on_error", True):
                    with self.assertRaises(TypeError):
                        offsetof._get_field_offset(holder, path)

    def test_nested_forward_field_is_resolved_before_lookup(self):
        @compile
        class ForwardHolder:
            prefix: i32
            child: "OffsetofLaterChild"

        @compile
        class OffsetofLaterChild:
            tag: i8
            value: i64

        expected = COuter.middle.offset + CInner.value.offset
        self.assertEqual(
            offsetof._get_field_offset(ForwardHolder, "child.value"), expected
        )


if __name__ == "__main__":
    unittest.main()
