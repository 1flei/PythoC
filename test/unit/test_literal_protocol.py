import ast
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pythoc.builtin_entities import i32, linear
from pythoc.builtin_entities.consume import consume
from pythoc.builtin_entities.struct import struct
from pythoc.literal_protocol import (
    extract_subscript_items,
    get_linear_schema_paths,
    get_mapping_entries,
    get_multidim_subscript_indices,
    get_sequence_elements,
    get_unpack_values,
    is_linear_schema_type,
    is_pc_dict_type,
    is_pc_tuple_type,
    iter_literal_value_refs,
    iter_value_ref_leaves,
    project_tracking_metadata,
    rebuild_mapping_carrier,
    rebuild_sequence_carrier,
    wrap_literal_result,
)
from pythoc.valueref import wrap_value


class _MockVisitor:
    @staticmethod
    def _is_linear_type(type_hint):
        return type_hint is linear or getattr(type_hint, "_is_linear", False)


class TestLiteralProtocol(unittest.TestCase):
    def test_iter_literal_value_refs_recurses_mapping_and_sequence(self):
        left = wrap_value(object(), kind="value", type_hint=linear, var_name="left", linear_path=())
        right = wrap_value(object(), kind="value", type_hint=linear, var_name="right", linear_path=(1,))

        nested = wrap_literal_result({"token": (left, right)})
        refs = list(iter_literal_value_refs(nested.get_python_value()))

        self.assertEqual([ref.var_name for ref in refs], ["left", "right"])
        self.assertEqual([ref.linear_path for ref in refs], [(), (1,)])

    def test_iter_value_ref_leaves_keeps_non_python_and_recurses_python(self):
        tracked = wrap_value(object(), kind="value", type_hint=linear, var_name="token", linear_path=())
        nested = wrap_literal_result((tracked,))

        direct_refs = list(iter_value_ref_leaves(tracked))
        nested_refs = list(iter_value_ref_leaves(nested))

        self.assertEqual(direct_refs, [tracked])
        self.assertEqual(nested_refs, [tracked])

    def test_get_unpack_values_uses_python_sequence_carrier(self):
        first = wrap_value(object(), kind="value", type_hint=linear, var_name="first", linear_path=())
        second = wrap_value(object(), kind="value", type_hint=linear, var_name="second", linear_path=(1,))

        tuple_value = wrap_literal_result((first, second))
        unpacked = get_unpack_values(None, tuple_value, None)

        self.assertEqual(unpacked[0].var_name, "first")
        self.assertEqual(unpacked[1].var_name, "second")

    def test_get_multidim_subscript_indices_normalizes_python_sequence(self):
        index = wrap_literal_result((1, 2, 3))

        indices = get_multidim_subscript_indices(None, index, None)

        self.assertEqual([item.get_python_value() for item in indices], [1, 2, 3])

    def test_extract_subscript_items_uses_sequence_carrier_path(self):
        tuple_items = wrap_literal_result((i32, linear)).get_python_value()

        self.assertEqual(extract_subscript_items(tuple_items), (i32, linear))

    def test_struct_normalize_subscript_items_accepts_slice_pair_carrier(self):
        named_item = wrap_literal_result(("field", i32)).get_python_value()

        self.assertEqual(struct.normalize_subscript_items(named_item), (("field", i32),))

    def test_struct_normalize_subscript_items_accepts_tuple_of_slice_pair_carriers(self):
        named_items = wrap_literal_result((("field", i32), ("flag", linear))).get_python_value()

        self.assertEqual(
            struct.normalize_subscript_items(named_items),
            (("field", i32), ("flag", linear)),
        )

    def test_linear_schema_helpers_recurse_nested_composites(self):
        inner = struct[linear, i32]
        outer = struct[i32, inner]

        self.assertTrue(is_linear_schema_type(outer))
        self.assertEqual(get_linear_schema_paths(outer), [(1, 0)])

    def test_project_tracking_metadata_extends_child_path(self):
        base = wrap_value(
            object(),
            kind="address",
            type_hint=struct[linear],
            address=object(),
            var_name="resource",
            linear_path=(2,),
        )

        self.assertEqual(project_tracking_metadata(base, 1), ("resource", (2, 1)))

    def test_rebuild_helpers_preserve_carrier_flavor(self):
        tracked = wrap_value(object(), kind="value", type_hint=linear, var_name="token", linear_path=())
        replacement = wrap_value(object(), kind="value", type_hint=linear, var_name="fresh", linear_path=())

        tuple_template = wrap_literal_result((tracked,)).get_python_value()
        rebuilt_tuple = rebuild_sequence_carrier(tuple_template, [replacement])
        self.assertTrue(is_pc_tuple_type(rebuilt_tuple))
        self.assertEqual(get_sequence_elements(rebuilt_tuple), [replacement])

        dict_template = wrap_literal_result({"token": tracked}).get_python_value()
        rebuilt_dict = rebuild_mapping_carrier(
            dict_template,
            [(wrap_literal_result("token"), replacement)],
        )
        self.assertTrue(is_pc_dict_type(rebuilt_dict))
        self.assertEqual(get_mapping_entries(rebuilt_dict)[0][1], replacement)

    def test_consume_accepts_nested_linear_mapping_carrier(self):
        tracked = wrap_value(object(), kind="value", type_hint=linear, var_name="proof", linear_path=())
        dict_value = wrap_literal_result({"token": tracked})
        call_node = ast.parse("consume(x)").body[0].value

        result = consume.handle_type_call(_MockVisitor(), None, [dict_value], call_node)

        self.assertTrue(result.is_python_value())
        self.assertEqual(result.type_hint.get_name(), "void")


if __name__ == "__main__":
    unittest.main()
