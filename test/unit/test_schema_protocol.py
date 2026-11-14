import os
import sys
import unittest
from unittest.mock import patch

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

    def test_shared_nonlinear_schema_work_is_bounded_by_unique_nodes(self):
        for depth in (16, 64, 256):
            with self.subTest(depth=depth):
                schema = i32
                for _ in range(depth):
                    schema = type('SharedSchema', (), {
                        '_field_types': [schema, schema],
                    })
                with patch('pythoc.schema_protocol.get_schema_field_types',
                           wraps=get_schema_field_types) as fields:
                    self.assertFalse(is_linear_schema_type(schema))
                self.assertEqual(fields.call_count, depth + 1)

    def test_linear_sibling_after_shared_nonlinear_schema_is_found(self):
        shared = struct['value': i32]
        pair = struct['left': shared, 'right': shared]
        schema = struct['data': pair, 'token': struct['proof': linear]]
        self.assertTrue(is_linear_schema_type(schema))

    def test_linearity_queries_observe_current_schema_fields(self):
        schema = struct['value': i32]
        self.assertFalse(is_linear_schema_type(schema))
        with patch.object(schema, '_field_types', [linear]):
            self.assertTrue(is_linear_schema_type(schema))
        self.assertFalse(is_linear_schema_type(schema))

    def test_cyclic_schema_queries_terminate_and_find_linear_fields(self):
        schema = type('RecursiveSchema', (), {})
        schema._field_types = [schema, i32]
        self.assertFalse(is_linear_schema_type(schema))
        schema._field_types.append(linear)
        self.assertTrue(is_linear_schema_type(schema))


if __name__ == "__main__":
    unittest.main()
