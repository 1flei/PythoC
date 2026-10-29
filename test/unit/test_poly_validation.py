"""Poly rejects incomplete dispatch plans before generating native code."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from llvmlite import ir
from pythoc import enum, f64, i32
from pythoc.logger import logger
from pythoc.std.poly import Poly
from pythoc.valueref import wrap_value


@enum(i32)
class Number:
    Integer: i32
    Float: f64


@enum(i32)
class OptionalNumber:
    Integer: i32
    Absent: None


def overload(argument, result):
    return SimpleNamespace(_func_info=SimpleNamespace(
        param_names=['value'], param_type_hints={'value': argument},
        return_type_hint=result,
    ))


def call_with_enum(dispatch, enum_type):
    value = wrap_value(ir.Constant(enum_type.get_llvm_type(ir.Context()), None),
                       kind='value', type_hint=enum_type)
    return dispatch.handle_call(None, None, [value], None)


class PolyValidationTest(unittest.TestCase):
    def setUp(self):
        self.errors = patch.object(logger, 'raise_on_error', True)
        self.errors.start()
        self.addCleanup(self.errors.stop)

    def test_missing_payload_overload_is_rejected(self):
        dispatch = Poly(overload(i32, i32))
        with self.assertRaisesRegex(RuntimeError, 'missing function overloads'):
            call_with_enum(dispatch, Number)

    def test_incompatible_returns_are_rejected(self):
        dispatch = Poly(overload(i32, i32), overload(f64, f64))
        with self.assertRaisesRegex(RuntimeError, 'inconsistent return types'):
            call_with_enum(dispatch, Number)

    def test_payloadless_variant_cannot_fall_through_dispatch(self):
        dispatch = Poly(overload(i32, i32))
        with self.assertRaisesRegex(TypeError, 'requires a payload'):
            call_with_enum(dispatch, OptionalNumber)


if __name__ == '__main__':
    unittest.main()
