"""Explicit C ABI configurations preserve zero-width field alignment."""

import unittest

from pythoc import array, i8, struct, u3, u5, u6
from test.utils.c_bitfield_profiles import c_bitfield_options


class TestCBitfieldProfiles(unittest.TestCase):
    def test_target_profiles_cover_alignment_and_microsoft_unit_changes(self):
        cases = (
            ('arm64-apple-darwin', 6, 2, (0, 3, 8), 4),
            ('x86_64-unknown-linux-gnu', 6, 2, (0, 3, 8), 4),
            ('aarch64-unknown-linux-gnu', 8, 4, (0, 3, 8), 4),
            ('x86_64-pc-windows-gnu', 8, 4, (0, 32, 64), 12),
        )
        for triple, size, alignment, mixed_offsets, mixed_size in cases:
            with self.subTest(triple=triple):
                options = c_bitfield_options(triple)

                @struct(**options['boundary'])
                class Boundary:
                    a: u3
                    b: u3

                @struct(**options['mixed'])
                class Mixed:
                    a: u3
                    b: u5
                    c: u6

                layout = Boundary._struct_type.get_layout()
                self.assertEqual(layout.field_offsets_bits, (0, 32))
                self.assertEqual((layout.allocation_size, layout.alignment,
                                  layout.stride), (size, alignment, size))
                record = struct[i8, array[Boundary, 2], i8].get_layout()
                self.assertEqual(record.field_offsets[2], alignment + 2 * size)
                mixed = Mixed._struct_type.get_layout()
                self.assertEqual(mixed.field_offsets_bits, mixed_offsets)
                self.assertEqual(mixed.allocation_size, mixed_size)


if __name__ == '__main__':
    unittest.main()
