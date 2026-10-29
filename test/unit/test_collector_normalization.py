"""Raw calls and ABI-shaped calls converge without repacking their carriers."""

import unittest

from llvmlite import ir
from pythoc import enum, i32, struct
from pythoc.call_normalization import (
    NormalizedCallArgs, build_kwargs_carrier, build_varargs_carrier,
    normalize_typed_collectors,
)
from pythoc.literal_protocol import get_sequence_elements
from pythoc.valueref import wrap_value


@enum(i32)
class Input:
    Value: i32


class CollectorNormalizationTest(unittest.TestCase):
    def test_raw_single_enum_is_packed(self):
        context = ir.Context()
        value = wrap_value(ir.Constant(Input.get_llvm_type(context), None),
                           kind='value', type_hint=Input)
        pack = struct[Input]
        normalized = normalize_typed_collectors(
            [value], (pack,), has_varargs=True, has_kwargs=False)
        self.assertIs(get_sequence_elements(normalized[0].get_python_value())[0], value)
        repeated = normalize_typed_collectors(
            normalized, (pack,), has_varargs=True, has_kwargs=False)
        self.assertIs(repeated[0], normalized[0])

    def test_native_abi_carrier_is_preserved(self):
        pack = struct[i32]
        value = wrap_value(ir.Constant(pack.get_llvm_type(ir.Context()), None),
                           kind='value', type_hint=pack)
        normalized = normalize_typed_collectors(
            NormalizedCallArgs([value]), (pack,),
            has_varargs=True, has_kwargs=False)
        self.assertIs(normalized[0], value)

    def test_native_kwargs_carrier_is_preserved(self):
        pack = struct['answer': i32]
        value = wrap_value(ir.Constant(pack.get_llvm_type(ir.Context()), None),
                           kind='value', type_hint=pack)
        normalized = normalize_typed_collectors(
            NormalizedCallArgs([value]), (pack,),
            has_varargs=False, has_kwargs=True)
        self.assertIs(normalized[0], value)

    def test_raw_struct_is_not_mistaken_for_its_collector(self):
        payload = struct['answer': i32]
        value = wrap_value(ir.Constant(payload.get_llvm_type(ir.Context()), None),
                           kind='value', type_hint=payload)
        normalized = normalize_typed_collectors(
            [value], (struct[payload],), has_varargs=True, has_kwargs=False)
        elements = get_sequence_elements(normalized[0].get_python_value())
        self.assertEqual(len(elements), 1)
        self.assertIs(elements[0], value)

    def test_nonempty_kwargs_and_positional_carriers_are_idempotent(self):
        value = wrap_value(ir.Constant(ir.IntType(32), 7),
                           kind='value', type_hint=i32)
        positional = build_varargs_carrier([value])
        keywords = build_kwargs_carrier({'answer': value})
        args = NormalizedCallArgs([positional, keywords])
        for _ in range(2):
            args = normalize_typed_collectors(
                args, (struct[i32], struct['answer': i32]),
                has_varargs=True, has_kwargs=True)
            self.assertIs(args[0], positional)
            self.assertIs(args[1], keywords)

    def test_both_literal_collectors_and_fixed_prefix_are_preserved(self):
        value = wrap_value(ir.Constant(ir.IntType(32), 7), kind='value', type_hint=i32)
        args = NormalizedCallArgs([
            value, build_varargs_carrier([value]), build_kwargs_carrier({})])
        normalized = normalize_typed_collectors(
            args, (i32, struct[i32], struct[tuple()]),
            has_varargs=True, has_kwargs=True)
        self.assertEqual([id(value) for value in normalized], [id(value) for value in args])

    def test_raw_tuple_payload_is_packed_even_when_it_looks_like_a_carrier(self):
        value = wrap_value(ir.Constant(ir.IntType(32), 7),
                           kind='value', type_hint=i32)
        payload = build_varargs_carrier([value, value])
        normalized = normalize_typed_collectors(
            [payload], (struct[struct[i32, i32]],),
            has_varargs=True, has_kwargs=False)
        elements = get_sequence_elements(normalized[0].get_python_value())
        self.assertEqual(len(elements), 1)
        self.assertIs(elements[0], payload)
        self.assertIs(normalize_typed_collectors(
            normalized, (struct[struct[i32, i32]],),
            has_varargs=True, has_kwargs=False), normalized)


if __name__ == '__main__':
    unittest.main()
