"""Poly rejects incomplete dispatch plans before generating native code."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pythoc import enum, f64, i32, func, u32
from pythoc.logger import logger
from pythoc.std.poly import Poly
from pythoc.type_converter import TypeConverter


@enum(i32)
class Number:
    Integer: i32
    Float: f64


@enum(i32)
class OptionalNumber:
    Integer: i32
    Absent: None


def overload(argument, result):
    return SimpleNamespace(_is_compiled=True, _func_info=SimpleNamespace(
        callable_pc_type=func[argument, result],
        param_names=['value'], param_type_hints={'value': argument},
        return_type_hint=result,
    ))


def call_with_enum(dispatch, enum_type):
    return dispatch.resolve(enum_type)


class PolyValidationTest(unittest.TestCase):
    def setUp(self):
        self.errors = patch.object(logger, 'raise_on_error', True)
        self.errors.start()
        self.addCleanup(self.errors.stop)

    def test_missing_payload_overload_is_rejected(self):
        dispatch = Poly(overload(i32, i32))
        with self.assertRaisesRegex(TypeError, 'missing function overloads'):
            call_with_enum(dispatch, Number)

    def test_incompatible_returns_are_rejected(self):
        dispatch = Poly(overload(i32, i32), overload(f64, f64))
        with self.assertRaisesRegex(TypeError, 'inconsistent return types'):
            call_with_enum(dispatch, Number)

    def test_payloadless_variant_cannot_fall_through_dispatch(self):
        dispatch = Poly(overload(i32, i32))
        with self.assertRaisesRegex(TypeError, 'requires a payload'):
            call_with_enum(dispatch, OptionalNumber)

    def test_forward_signature_refreshes_after_completion(self):
        from pythoc import ptr, mark_type_defined
        from pythoc.session import CompileSession
        with CompileSession():
            function = overload(ptr['PolyLateType'], i32)
            dispatch = Poly(function)
            mark_type_defined('PolyLateType', i32)
            self.assertIs(dispatch.resolve(ptr[i32]), function)

    def test_invalid_append_does_not_change_dispatcher(self):
        function = overload(i32, i32)
        dispatch = Poly(function)
        with self.assertRaises(TypeError):
            dispatch.append(object())
        self.assertEqual(dispatch._funcs, [function])

    def test_call_protocol_is_the_inline_entry_protocol(self):
        dispatch = Poly(overload(i32, i32))
        self.assertIs(dispatch.handle_call, dispatch._call.handle_call)
        self.assertEqual(dispatch.defer_linear_transfer,
                         dispatch._call.defer_linear_transfer)

    def test_selection_reads_the_argument_carrier_directly(self):
        from llvmlite import ir
        from pythoc.builtin_entities.pc_tuple import create_pc_tuple_type
        from pythoc.valueref import wrap_value

        function = overload(i32, i32)
        dispatch = Poly(function)
        value = wrap_value(ir.Constant(ir.IntType(32), 7), kind='value', type_hint=i32)
        arguments = create_pc_tuple_type([value])
        self.assertIs(dispatch._select(arguments), function)

    def test_literal_selection_uses_the_shared_default_type_rule(self):
        from pythoc.builtin_entities.pc_tuple import create_pc_tuple_type
        from pythoc.valueref import wrap_python_constant

        function = overload(u32, i32)
        dispatch = Poly(function)
        arguments = create_pc_tuple_type([wrap_python_constant(7)])
        with patch.object(TypeConverter, 'infer_default_pc_type_from_python',
                          return_value=u32) as infer_type:
            self.assertIs(dispatch._select(arguments), function)
        infer_type.assert_called_once_with(7)


if __name__ == '__main__':
    unittest.main()
