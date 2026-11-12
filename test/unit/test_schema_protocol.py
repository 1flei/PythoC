import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pythoc.builtin_entities import i32, linear
from pythoc.builtin_entities.struct import struct
from pythoc.schema_protocol import (
    get_linear_schema_paths,
    get_schema_field_names,
    get_schema_field_types,
    is_linear_schema_type,
    is_schema_type,
)


class TestSchemaProtocol(unittest.TestCase):
    def test_schema_field_helpers_preserve_named_layout(self):
        packet = struct["size": i32, "token": linear]

        self.assertTrue(is_schema_type(packet))
        self.assertEqual(get_schema_field_names(packet), ["size", "token"])
        self.assertEqual(get_schema_field_types(packet), [i32, linear])

    def test_linear_schema_helpers_recurse_nested_composites(self):
        inner = struct[linear, i32]
        outer = struct[i32, inner]

        self.assertTrue(is_linear_schema_type(outer))
        self.assertEqual(get_linear_schema_paths(outer), [(1, 0)])

    def test_indexed_field_queries_do_linear_total_work(self):
        from pythoc.schema_protocol import (get_field_layout_type,
                                            get_schema_field_type)

        class CountedFields(list):
            accesses = 0

            def __iter__(self):
                self.accesses += len(self)
                return super().__iter__()

            def __getitem__(self, index):
                self.accesses += 1
                return super().__getitem__(index)

        class Record:
            _field_types = CountedFields([i32] * 128)

        for lookup in (get_field_layout_type, get_schema_field_type):
            Record._field_types.accesses = 0
            for index in range(128):
                self.assertIs(lookup(Record, index), i32)
            self.assertLessEqual(Record._field_types.accesses, 256)


if __name__ == "__main__":
    unittest.main()
