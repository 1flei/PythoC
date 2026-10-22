"""End-to-end target layout regressions."""

import unittest
from pathlib import Path
import shlex
import subprocess
import tempfile

from pythoc import (
    array, compile, f64, i8, i24, i32, i64, i128, offsetof, ptr, simd,
    sizeof, struct, union,
)
from pythoc.utils.cc_utils import find_available_cc


Payload = struct[
    ("payload", union[array[i24, 3]]),
    ("canary", i64),
]

VectorPayload = struct[
    ('tag', i8), ('payload', union[simd[i32, 8]]), ('canary', i64),
]
WidePayload = struct[
    ('tag', i8), ('payload', union[i128]), ('canary', i64),
]


@compile
def layout_size() -> i64:
    return sizeof(Payload)


@compile
def unusual_integer_size() -> i64:
    return sizeof(i24)


@compile
def trailing_field_offset() -> i64:
    return offsetof(Payload, "canary")


@compile
def union_write_preserves_trailing_field() -> i64:
    value: Payload
    value[1] = i64(0x123456789ABCDEF0)
    value[0][0][2] = i24(1)
    return value[1]


@compile
def vector_union_write_preserves_neighbors() -> i64:
    value: VectorPayload
    value.tag = i8(23)
    value.canary = i64(0x123456789ABCDEF0)
    value.payload[0] = simd[i32, 8](7)
    if value.tag != 23 or value.payload[0][7] != 7:
        return -1
    return value.canary


@compile
def wide_union_write_preserves_neighbors() -> i64:
    value: WidePayload
    value.tag = i8(23)
    value.canary = i64(0x123456789ABCDEF0)
    value.payload[0] = i128(1) << 100
    if value.tag != 23 or value.payload[0] != i128(1) << 100:
        return -1
    return value.canary


@compile
def unusual_integer_stride() -> i64:
    values: array[i24, 2]
    return i64(ptr(values[1])) - i64(ptr(values[0]))


class TestTargetLayoutIntegration(unittest.TestCase):
    def test_sizeof_uses_target_allocation_size(self):
        self.assertEqual(unusual_integer_size(), 4)

    def test_sizeof_uses_semantic_allocation_size(self):
        self.assertEqual(layout_size(), 24)

    def test_offsetof_uses_semantic_field_offsets(self):
        self.assertEqual(trailing_field_offset(), 16)

    def test_union_write_does_not_overwrite_adjacent_field(self):
        self.assertEqual(
            union_write_preserves_trailing_field(),
            0x123456789ABCDEF0,
        )

    def test_wide_union_preserves_neighbors(self):
        self.assertEqual(wide_union_write_preserves_neighbors(),
                         0x123456789ABCDEF0)

    def test_vector_union_preserves_neighbors(self):
        self.assertEqual(vector_union_write_preserves_neighbors(),
                         0x123456789ABCDEF0)

    def test_pointer_stride_matches_sizeof(self):
        self.assertEqual(unusual_integer_stride(), unusual_integer_size())

    def test_natural_layout_matches_independent_c_compiler(self):
        source = Path(__file__).parents[1] / 'c_oracles' / 'natural_layout.c'
        with tempfile.TemporaryDirectory(prefix='pc-natural-layout-') as temp:
            executable = str(Path(temp) / 'oracle.exe')
            command = shlex.split(find_available_cc())
            subprocess.run(command + ['-std=c11', str(source), '-o', executable],
                           check=True, capture_output=True, text=True)
            actual = tuple(map(int, subprocess.check_output(
                [executable], text=True).split()))
        scalar = union[i32, f64]
        natural = struct[i8, scalar, array[i32, 2, 3], i8].get_layout()
        vector = union[simd[i32, 4], i8]
        vector_record = struct[i8, vector, i64].get_layout()
        expected = (
            scalar.get_size_bytes(), scalar.get_alignment(),
            natural.allocation_size, natural.alignment,
            *natural.field_offsets[1:],
            vector.get_size_bytes(), vector.get_alignment(),
            vector_record.allocation_size, vector_record.alignment,
            vector_record.field_offsets[2],
        )
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
