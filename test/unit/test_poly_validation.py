"""Poly rejects incomplete dispatch plans before generating native code."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pythoc import enum, f64, i32
from pythoc.logger import logger
from pythoc.std.poly import Poly


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
    return dispatch.resolve(enum_type)


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
        with self.assertRaises(NameError):
            dispatch.append(object())
        self.assertEqual(dispatch._funcs, [function])


if __name__ == '__main__':
    unittest.main()
