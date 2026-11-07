"""Ordinary handlers use argument values independently of source spelling."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from pythoc import (
    assume, compile, func, i32, llvm_asm, offsetof, param, sizeof, struct,
    typeof, u64,
)
from pythoc import bool as pc_bool
from pythoc.build.output_manager import flush_all_pending_outputs
from pythoc.session import CompileSession
from test.utils.test_utils import DeferredTestCase


Record = struct['first': i32, 'second': i32]
FIELD = 'second'


@compile
def assume_direct() -> i32:
    return i32(assume(3, 'checked'))


@compile
def assume_starred() -> i32:
    return i32(assume(*(3, 'checked')))


@compile
def literal_field_name() -> u64:
    return offsetof(Record, 'second')


@compile
def aliased_field_name() -> u64:
    return offsetof(Record, FIELD)


@compile
def values(T: param, value: T):
    yield value


@compile
def yield_direct() -> i32:
    for value in values(i32, i32(7)):
        return value
    return 0


@compile
def yield_starred() -> i32:
    for value in values(*(i32, i32(7))):
        return value
    return 0


@compile
def literal_asm_bool() -> i32:
    llvm_asm('', '', side_effect=True)
    return 7


@compile
def computed_asm_bool() -> i32:
    llvm_asm('', '', side_effect=(1 == 1))
    return 7


@compile
def literal_asm_inputs() -> i32:
    llvm_asm('', 'r', inputs=[i32(4)])
    return 7


@compile
def aliased_asm_inputs() -> i32:
    operands = [i32(4)]
    llvm_asm('', 'r', inputs=operands)
    return 7


@compile
def positive(value: i32) -> pc_bool:
    return value > 0


@compile
def negative(value: i32) -> pc_bool:
    return value < 0


predicates = SimpleNamespace(negative=negative)


def predicate_id(type_object):
    name = type_object._predicates[0].__name__
    return {'positive': 1, 'negative': 2}[name]


@compile
def direct_predicate() -> i32:
    value = assume(i32(3), positive)
    return predicate_id(typeof(value)) * 10 + i32(positive(i32(3)))


@compile
def attribute_predicate() -> i32:
    value = assume(i32(3), predicates.negative)
    return predicate_id(typeof(value)) * 10 + i32(predicates.negative(i32(3)))


class CallValueBoundaryTest(DeferredTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        flush_all_pending_outputs()

    def test_assume_accepts_expanded_arguments(self):
        self.assertEqual(assume_direct(), 3)
        self.assertEqual(assume_starred(), 3)

    def test_offsetof_accepts_a_constant_field_name_alias(self):
        self.assertEqual(literal_field_name(), 4)
        self.assertEqual(aliased_field_name(), 4)

    def test_parametric_yield_accepts_expanded_arguments(self):
        self.assertEqual(yield_direct(), 7)
        self.assertEqual(yield_starred(), 7)

    def test_asm_accepts_a_computed_constant_bool(self):
        self.assertEqual(literal_asm_bool(), 7)
        self.assertEqual(computed_asm_bool(), 7)

    def test_asm_accepts_an_operand_carrier_alias(self):
        self.assertEqual(literal_asm_inputs(), 7)
        self.assertEqual(aliased_asm_inputs(), 7)

    def test_assume_retains_direct_predicate_identity(self):
        self.assertEqual(direct_predicate(), 11)

    def test_assume_retains_attribute_predicate_identity(self):
        self.assertEqual(attribute_predicate(), 20)

    def test_cast_rejects_unrecognized_keywords(self):
        with CompileSession(), self.assertRaisesRegex(TypeError, 'unexpected keyword'):
            @compile(suffix='cast_unknown_keyword')
            def invalid_cast() -> i32:
                return i32(7, mystery=9)

            flush_all_pending_outputs()

    def test_sizeof_rejects_unrecognized_keywords(self):
        with CompileSession(), self.assertRaisesRegex(TypeError, 'unexpected keyword'):
            @compile(suffix='sizeof_unknown_keyword')
            def invalid_sizeof() -> u64:
                return sizeof(i32, mystery=9)

            flush_all_pending_outputs()

    def test_runtime_pointer_cannot_supply_predicate_identity(self):
        with CompileSession(), self.assertRaisesRegex(TypeError, 'no static identity'):
            @compile(suffix='runtime_predicate_pointer')
            def invalid_predicate() -> i32:
                predicate: func[i32, pc_bool] = positive
                return i32(assume(i32(3), predicate))

            flush_all_pending_outputs()

    def test_shadowed_pointer_cannot_recover_a_global_predicate(self):
        with CompileSession(), self.assertRaisesRegex(TypeError, 'no static identity'):
            @compile(suffix='shadowed_predicate_pointer')
            def invalid_predicate() -> i32:
                positive: func[i32, pc_bool] = negative
                return i32(assume(i32(3), positive))

            flush_all_pending_outputs()


if __name__ == '__main__':
    unittest.main()
