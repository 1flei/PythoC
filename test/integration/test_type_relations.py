"""Compiler boundaries use PythoC type relations before LLVM lowering."""

from contextlib import chdir
import os
from pathlib import Path
import tempfile
import unittest

import pythoc

from pythoc import array, compile, const, func, i32, i64, ptr, refined, static, void
from pythoc.builtin_entities.enum import _create_enum_type
from pythoc.builtin_entities.python_type import PythonType
from pythoc.builtin_entities.struct import create_struct_type
from pythoc.build.output_manager import clear_failed_group, flush_all_pending_outputs
from pythoc.logger import set_raise_on_error

from fixtures.type_relation_matrix import (
    CASES, ENTRIES, build_conversion, build_probe,
)


set_raise_on_error(True)
_TypeRelationTag = PythonType.wrap(7, is_constant=True)
FIXTURES = Path(__file__).resolve().parent / 'fixtures'


def _subprocess_environment():
    environment = os.environ.copy()
    environment['PYTHONPATH'] = os.pathsep.join([
        str(Path(pythoc.__file__).resolve().parent.parent),
        environment.get('PYTHONPATH', ''),
    ])
    environment['PC_RAISE_ON_ERROR'] = '1'
    return environment


class TypeRelationsTest(unittest.TestCase):
    def setUp(self):
        self.tmp_path = Path(self.enterContext(tempfile.TemporaryDirectory(
            prefix='type-relations-')))
        self.addCleanup(os.chdir, Path.cwd())

    def _reject(self, wrapper, message, error_type=TypeError):
        try:
            with self.assertRaisesRegex(error_type, message):
                flush_all_pending_outputs()
        finally:
            clear_failed_group(wrapper._binding.group_key)

    def test_same_display_name_different_record_pointer_rejected(self):
        small = create_struct_type([i32], ['value'],
                                   python_class=type('SameRecord', (), {}))
        large = create_struct_type([i64], ['value'],
                                   python_class=type('SameRecord', (), {}))

        @compile(suffix='relation_same_name_pointer')
        def should_fail(value: ptr[small]) -> ptr[large]:
            return value

        self._reject(should_fail, 'Cannot implicitly convert')

    def test_refinement_wrapped_in_const_requires_proof(self):
        @compile(suffix='relation_wrapped_refinement')
        def should_fail(value: i32) -> i32:
            checked: const[refined[i32, 'checked']] = value
            return value

        self._reject(should_fail, 'refined type')

    def test_implicit_pointer_conversion_cannot_discard_const(self):
        @compile(suffix='relation_pointee_const')
        def should_fail(value: ptr[const[i32]]) -> ptr[i32]:
            return value

        self._reject(should_fail, 'Cannot implicitly convert')

    def test_implicit_pointer_conversion_can_add_const(self):
        @compile(suffix='relation_add_pointee_const')
        def add_const(value: ptr[i32]) -> ptr[const[i32]]:
            return value

        @compile(suffix='relation_add_const_runtime')
        def run() -> i32:
            value: i32 = 37
            return add_const(ptr(value))[0]

        flush_all_pending_outputs()
        self.assertEqual(run(), 37)

    def test_static_aggregate_cannot_discard_pointee_const(self):
        @compile(suffix='relation_static_nested_const')
        def should_fail() -> i32:
            values: static[array[ptr[i32], 1]] = (ptr[const[i32]](0),)
            return i32(values[0] == ptr[i32](0))

        self._reject(should_fail, 'static aggregate initializer')

    def test_callable_wrapper_rejects_wrong_signature_at_admission(self):
        @compile(suffix='relation_callable_source')
        def source(value: i32) -> i32:
            return value

        @compile(suffix='relation_callable_signature')
        def should_fail() -> i32:
            pointer: func[i64, i64] = source
            return 0

        self._reject(should_fail, 'has signature')

    def test_ternary_pointer_common_type_retains_const(self):
        @compile(suffix='relation_ternary_pointer_const')
        def choose(flag: i32, left: ptr[const[i32]],
                   right: ptr[i32]) -> ptr[const[i32]]:
            return left if flag else right

        @compile(suffix='relation_ternary_const_runtime')
        def run(flag: i32) -> i32:
            left: i32 = 37
            right: i32 = 11
            return choose(flag, ptr(left), ptr(right))[0]

        flush_all_pending_outputs()
        self.assertEqual(run(1), 37)
        self.assertEqual(run(0), 11)

    def test_aggregate_conversion_checks_nested_pointer_direction(self):
        source = create_struct_type([ptr[const[i32]]], ['item'])
        target = create_struct_type([ptr[i32]], ['item'])

        @compile(suffix='relation_nested_const_pointer')
        def should_fail(value: source) -> target:
            return value

        self._reject(should_fail, 'incompatible fields')

    def test_array_decay_uses_semantic_element_type(self):
        @compile(suffix='relation_array_decay')
        def read_second() -> i32:
            values: array[i32, 3] = [1, 2, 3]
            pointer: ptr[i32] = values
            return pointer[1]

        flush_all_pending_outputs()
        self.assertEqual(read_second(), 2)

    def test_literal_array_rejects_bad_element(self):
        @compile(suffix='relation_array_bad_element')
        def should_fail() -> i32:
            values: array[i32, 2] = [1, 'bad']
            return values[0]

        self._reject(should_fail, 'Cannot implicitly convert')

    def test_string_literal_flows_to_void_pointer(self):
        @compile(suffix='relation_string_void_pointer')
        def make_string() -> ptr[void]:
            return 'hello'

        from pythoc import i8

        @compile(suffix='relation_string_void_runtime')
        def read() -> i32:
            return ptr[i8](make_string())[0]

        flush_all_pending_outputs()
        self.assertEqual(read(), ord('h'))

    def test_struct_to_enum_checks_tag_and_payload_before_lowering(self):
        target = _create_enum_type([('Ok', i32, 0), ('Empty', None, 1)], i32)
        invalid = create_struct_type([PythonType.wrap(9), i32],
                                     ['tag', 'payload'])

        @compile(suffix='relation_enum_bad_tag')
        def should_fail(value: invalid) -> target:
            return value

        self._reject(should_fail, 'Cannot implicitly convert')

    def test_identical_enum_layout_does_not_authorize_conversion(self):
        source = _create_enum_type([('Ok', i32, 0)], i32)
        target = _create_enum_type([('Ok', i32, 1)], i32)

        @compile(suffix='relation_enum_same_layout')
        def should_fail(value: source) -> target:
            return value

        self._reject(should_fail, 'Cannot implicitly convert')

    def test_mixed_integer_operations_use_one_symmetric_type(self):
        from pythoc import u32, u64

        @compile(suffix='relation_mixed_numeric')
        def compare(left: i32, right: u32) -> i32:
            return i32(left < right) + 2 * i32(right > left)

        @compile(suffix='relation_mixed_ternary')
        def select(flag: i32, left: i32, right: u32) -> u64:
            return left if flag else right

        @compile(suffix='relation_mixed_division')
        def divide(left: i32, right: u32) -> u32:
            return left / right

        flush_all_pending_outputs()
        self.assertEqual(compare(-1, 1), 0)
        self.assertEqual(compare(0, 1), 3)
        self.assertEqual(select(1, -1, 1), 4294967295)
        self.assertEqual(select(0, -1, 1), 1)
        self.assertEqual(divide(-1, 2), 2147483647)

    def test_enum_constructor_rejects_invalid_variant_shapes(self):
        result = _create_enum_type([('Ok', i32, 0), ('Empty', None, 1)], i32)

        @compile(suffix='relation_enum_constructor_missing')
        def missing() -> result:
            return result(result.Ok)

        self._reject(missing, 'known valid variant tag')

        @compile(suffix='relation_enum_constructor_extra')
        def extra() -> result:
            return result(result.Empty, 4)

        self._reject(extra, 'known valid variant tag')

        @compile(suffix='relation_enum_constructor_invalid')
        def invalid() -> result:
            return result(99, 4)

        self._reject(invalid, 'known valid variant tag')

        @compile(suffix='relation_enum_constructor_dynamic')
        def dynamic(tag: i32) -> i32:
            item: result = result(tag, 4)
            return i32(item[0])

        flush_all_pending_outputs()
        self.assertEqual(dynamic(0), 0)

    def test_enum_constructor_cannot_discard_payload_const(self):
        result = _create_enum_type([('Ok', ptr[i32], 0)], i32)

        @compile(suffix='relation_enum_constructor_const')
        def invalid(value: ptr[const[i32]]) -> result:
            return result(result.Ok, value)

        self._reject(invalid, 'known valid variant tag')

    def test_runtime_enum_constructor_traps_ineligible_tag(self):
        import os
        import signal
        import subprocess
        import sys
        if os.name != 'posix':
            self.skipTest('signal result requires POSIX')
        script = FIXTURES / 'type_relation_enum_trap.py'
        environment = _subprocess_environment()
        valid = subprocess.run(
            [sys.executable, str(script), '0'], cwd=self.tmp_path,
            capture_output=True, env=environment, timeout=120)
        self.assertEqual(valid.returncode, 0, valid.stderr.decode())
        for tag in ('1', '99'):
            with self.subTest(tag=tag):
                invalid = subprocess.run(
                    [sys.executable, str(script), tag], cwd=self.tmp_path,
                    capture_output=True, env=environment, timeout=120)
                self.assertIn(invalid.returncode,
                              (-signal.SIGILL, -signal.SIGTRAP),
                              invalid.stderr.decode())

    def test_pointer_comparison_checks_semantics_before_same_ir_type(self):
        checked = refined[i32, 'checked']

        @compile(suffix='relation_compare_refined_pointer')
        def invalid(left: ptr[checked], right: ptr[i32]) -> i32:
            return i32(left == right)

        self._reject(invalid, 'Cannot compare incompatible pointer types')

    def test_pointer_comparison_allows_independent_storage_qualifiers(self):
        from pythoc import volatile

        @compile(suffix='relation_compare_const_volatile')
        def compare(left: ptr[const[i32]], right: ptr[volatile[i32]]) -> i32:
            return i32(left == right)

        @compile(suffix='relation_compare_qualified_runtime')
        def run() -> i32:
            left: i32 = 37
            right: i32 = 11
            return compare(ptr(left), ptr(left)) * 10 + compare(ptr(left), ptr(right))

        flush_all_pending_outputs()
        self.assertEqual(run(), 10)

    def test_shift_count_does_not_change_left_signedness(self):
        from pythoc import u32, u64

        @compile(suffix='relation_shift_signed')
        def signed_shift(value: i32, count: u64) -> i64:
            return value >> count

        @compile(suffix='relation_shift_unsigned')
        def unsigned_shift(value: u32, count: i64) -> u64:
            return value >> count

        flush_all_pending_outputs()
        self.assertEqual(signed_shift(-8, 1), -4)
        self.assertEqual(unsigned_shift(4294967288, 1), 2147483644)

    def test_shift_rejects_float_literal_count(self):
        @compile(suffix='relation_shift_float_count')
        def invalid(value: i32) -> i32:
            return value << 1.5

        self._reject(invalid, 'Shift operands must have integer types')

    def test_negative_constant_cast_matches_runtime_cast(self):
        from pythoc import i8, f64

        @compile(suffix='relation_negative_constant')
        def constant() -> i64:
            return i64(i8(-1))

        @compile(suffix='relation_negative_runtime')
        def runtime(value: i8) -> i64:
            return i64(value)

        @compile(suffix='relation_negative_float')
        def floating() -> f64:
            return f64(i8(-1))

        flush_all_pending_outputs()
        self.assertEqual(constant(), -1)
        self.assertEqual(runtime(-1), -1)
        self.assertEqual(floating(), -1.0)

    def test_arithmetic_erases_refinement_before_widening(self):
        checked = refined[i32, 'checked']

        @compile(suffix='relation_refined_arithmetic')
        def add(value: checked, other: i64) -> i64:
            return value + other

        @compile(suffix='relation_refined_arithmetic_caller')
        def call() -> i64:
            return add(checked(3), 4)

        flush_all_pending_outputs()
        self.assertEqual(call(), 7)

    def test_aggregate_default_cannot_forge_refinement(self):
        checked = refined[i32, 'checked']

        @compile(suffix='relation_default_refinement')
        def invalid() -> i32:
            values: array[checked, 1] = []
            return i32(values[0])

        self._reject(invalid, 'Cannot implicitly convert')

    def test_static_aggregate_default_cannot_forge_refinement(self):
        checked = refined[i32, 'checked']

        @compile(suffix='relation_static_default_refinement')
        def invalid() -> i32:
            values: static[array[checked, 1]] = ()
            return i32(values[0])

        self._reject(invalid, 'static aggregate initializer')

    def test_pointer_arithmetic_rejects_float_offsets(self):
        @compile(suffix='relation_pointer_float_offset')
        def invalid(value: ptr[i32]) -> ptr[i32]:
            return value + 1.5

        self._reject(invalid, 'must have an integer type')

    def test_pointer_index_zero_extends_unsigned_one_bit(self):
        from pythoc import u1

        @compile(suffix='relation_pointer_unsigned_index')
        def read(index: u1) -> i32:
            values: array[i32, 2] = [3, 7]
            return values[index]

        flush_all_pending_outputs()
        self.assertEqual(read(1), 7)

    def test_varargs_promote_unsigned_boolean_and_qualified_integers(self):
        from pythoc import i8, u8, u16, bool as pc_bool
        from pythoc.libc.stdio import snprintf
        from pythoc.libc.string import strcmp

        @compile(suffix='relation_varargs_promotions')
        def call() -> i32:
            buffer: array[i8, 32]
            value: const[u16] = 65535
            length = snprintf(buffer, 32, '%d:%d:%d:%d',
                              u8(255), pc_bool(True), i8(-1), value)
            return i32(length == 14 and strcmp(buffer, '255:1:-1:65535') == 0)

        flush_all_pending_outputs()
        self.assertEqual(call(), 1)

    def test_variadic_function_pointer_requires_fixed_arguments(self):
        signature = func[i32, i32]
        signature.has_llvm_varargs = True

        @compile(suffix='relation_varargs_fixed_arity')
        def invalid(pointer: signature) -> i32:
            return pointer()

        self._reject(invalid, 'expects at least 1 arguments')

    def test_const_pointer_keeps_mutable_pointee(self):
        @compile(suffix='relation_const_pointer_mutable_data')
        def write() -> i32:
            value: i32 = 1
            address: const[ptr[i32]] = ptr(value)
            address[0] = 7
            return value

        flush_all_pending_outputs()
        self.assertEqual(write(), 7)

    def test_const_array_decay_cannot_escape_to_mutable_pointer(self):
        @compile(suffix='relation_const_array_decay')
        def invalid() -> i32:
            values: const[array[i32, 2]] = [1, 2]
            address: ptr[i32] = values
            return address[0]

        self._reject(invalid, 'Cannot implicitly convert')

    def test_volatile_pointer_accesses_survive_pcir_without_extra_reads(self):
        os.chdir(self.tmp_path)
        from pythoc import volatile

        @compile(suffix='relation_volatile_pcir')
        def access(address: ptr[volatile[i32]]) -> i32:
            address[0] = 1
            return address[0] + address[0]

        flush_all_pending_outputs()
        emitted = str(access._func_info.llvm_function)
        self.assertEqual(emitted.count('store volatile i32'), 1)
        self.assertEqual(emitted.count('load volatile i32,'), 2)
        self.assertNotIn('load i32,', emitted)

    def test_simd_store_rejects_const_and_refined_pointees(self):
        from pythoc import simd, simd_store

        @compile(suffix='relation_simd_const_store')
        def invalid(address: ptr[const[i32]]) -> i32:
            simd_store(address, simd[i32, 4](1))
            return 0

        self._reject(invalid, 'Cannot store through a pointer to const')
        checked = refined[i32, 'checked']

        @compile(suffix='relation_simd_refined_store')
        def proof(address: ptr[checked]) -> i32:
            simd_store(address, simd[i32, 4](1))
            return 0

        self._reject(proof, 'pointer must address its lane type')

    def test_simd_volatile_accesses_survive_pcir(self):
        os.chdir(self.tmp_path)
        from pythoc import simd, simd_load, simd_store, simd_reduce_add, volatile

        @compile(suffix='relation_simd_volatile')
        def access(address: ptr[volatile[i32]]) -> i32:
            value = simd_load(address, simd[i32, 4])
            simd_store(address, value)
            return simd_reduce_add(value)

        flush_all_pending_outputs()
        emitted = str(access._func_info.llvm_function)
        self.assertIn('load volatile <4 x i32>', emitted)
        self.assertIn('store volatile <4 x i32>', emitted)

    def test_atomic_store_rejects_const_target(self):
        from pythoc import atomic_store

        @compile(suffix='relation_atomic_const_store')
        def invalid(address: ptr[const[i32]]) -> i32:
            atomic_store(address, 1)
            return 0

        self._reject(invalid, 'unsupported atomic store')

    def test_fixed_atomic_rejects_wrong_pointer_type(self):
        from pythoc.builtin_entities import atomic_load_i32

        @compile(suffix='relation_atomic_fixed_pointer')
        def invalid(address: ptr[i64]) -> i32:
            return atomic_load_i32(address)

        self._reject(invalid, 'expected a pointer to i32')

    def test_atomic_cas_checks_expected_pointer_type(self):
        from pythoc import atomic_cas, u32

        @compile(suffix='relation_atomic_expected_pointer')
        def invalid(address: ptr[i32], expected: ptr[u32]) -> i32:
            return atomic_cas(address, expected, 1)

        self._reject(invalid, 'expected pointer must have the same value type')

    def test_atomic_rmw_cannot_invalidate_refinement(self):
        from pythoc import atomic_fetch_add
        checked = refined[i64, 'checked']

        @compile(suffix='relation_atomic_refined_rmw')
        def invalid(address: ptr[checked], value: checked) -> i64:
            return atomic_fetch_add(address, value)

        self._reject(invalid, 'unsupported atomic add')

    def test_atomic_rejects_unsupported_integer_width(self):
        from pythoc import atomic_load, i24

        @compile(suffix='relation_atomic_width')
        def invalid(address: ptr[i24]) -> i32:
            return i32(atomic_load(address))

        self._reject(invalid, 'unsupported atomic load')

    def test_volatile_atomics_and_cas_failure_writeback_survive_pcir(self):
        os.chdir(self.tmp_path)
        from pythoc import atomic_load, atomic_store, atomic_fetch_add, atomic_cas, volatile

        @compile(suffix='relation_atomic_volatile_pcir')
        def access() -> i32:
            value: volatile[i64] = 1
            expected: volatile[i64] = 2
            atomic_store(ptr(value), 1)
            old = atomic_fetch_add(ptr(value), 1)
            success = atomic_cas(ptr(value), ptr(expected), 5)
            return i32(old) + 10 * success + 100 * i32(atomic_load(ptr(value)))

        flush_all_pending_outputs()
        emitted = str(access._func_info.llvm_function)
        self.assertIn('load atomic volatile i64', emitted)
        self.assertIn('store atomic volatile i64', emitted)
        self.assertIn('atomicrmw volatile add', emitted)
        self.assertIn('cmpxchg volatile', emitted)
        branches = [instruction for block in access._func_info.llvm_function.blocks
                    for instruction in block.instructions
                    if instruction.opname == 'br' and len(instruction.operands) == 3]
        failure = branches[-1].operands[2]
        self.assertTrue(any('store atomic volatile' in str(instruction)
                   for instruction in failure.instructions))

    def test_volatile_record_field_accesses_and_address_taking(self):
        os.chdir(self.tmp_path)
        from pythoc import struct, volatile
        record = struct['value': i32]

        @compile(suffix='relation_volatile_record_fields')
        def access(address: ptr[volatile[record]]) -> i32:
            address.value = 1
            return address.value + address.value

        @compile(suffix='relation_volatile_record_address')
        def take(address: ptr[volatile[record]]) -> ptr[volatile[i32]]:
            return ptr(address.value)

        flush_all_pending_outputs()
        emitted = str(access._func_info.llvm_function)
        self.assertEqual(emitted.count('store volatile i32'), 1)
        self.assertEqual(emitted.count('load volatile i32,'), 2)
        self.assertNotIn('load i32,', emitted)
        self.assertNotIn('load volatile', str(take._func_info.llvm_function))

    def test_void_cast_cannot_discard_moved_linear_token(self):
        from pythoc import linear, move

        @compile(suffix='relation_linear_discard')
        def invalid() -> i32:
            proof = linear()
            void(move(proof))
            return 0

        self._reject(invalid, 'Cannot discard a linear value')

    def test_void_cast_cannot_discard_linear_literal_carrier(self):
        from pythoc import linear, move

        @compile(suffix='relation_linear_carrier_discard')
        def invalid() -> i32:
            proof = linear()
            void((move(proof),))
            return 0

        self._reject(invalid, 'Cannot discard a linear value')

    def test_array_constructor_cannot_forge_refinement(self):
        checked = refined[i32, 'checked']

        @compile(suffix='relation_array_constructor_proof')
        def invalid() -> i32:
            values = array[checked, 1]()
            return i32(values[0])

        self._reject(invalid, 'Cannot zero-initialize an array')

    def test_nonempty_linear_arrays_are_explicitly_rejected(self):
        from pythoc import linear

        @compile(suffix='relation_linear_array')
        def invalid() -> i32:
            values: array[linear, 1]
            return 0

        self._reject(invalid, 'Arrays of linear values are unsupported')

    def test_conditional_cannot_copy_linear_branch_references(self):
        from pythoc import linear, consume

        @compile(suffix='relation_linear_conditional_copy')
        def invalid(flag: i32) -> i32:
            first = linear()
            second = linear()
            selected = first if flag else second
            consume(first)
            consume(second)
            consume(selected)
            return 0

        self._reject(invalid, 'Cannot assign linear token', (TypeError, RuntimeError))

    def test_conditional_can_move_the_same_resource_on_both_paths(self):
        from pythoc import linear, consume, move

        @compile(suffix='relation_linear_conditional_move')
        def select(flag: i32) -> i32:
            proof = linear()
            selected = move(proof) if flag else move(proof)
            consume(selected)
            return 7

        flush_all_pending_outputs()
        self.assertEqual(select(0), 7)
        self.assertEqual(select(1), 7)

    def test_annotated_struct_conversion_consumes_original_linear_source(self):
        from pythoc import linear, struct, consume
        first_type = struct['first': linear]
        second_type = struct['second': linear]

        @compile(suffix='relation_linear_annotated_conversion')
        def invalid() -> i32:
            first: first_type = (linear(),)
            second: second_type = first
            consume(first)
            consume(second)
            return 0

        self._reject(invalid, 'already consumed', (TypeError, RuntimeError))

    def test_conditional_preserves_shared_refinement(self):
        checked = refined[i32, 'checked']

        @compile(suffix='relation_conditional_shared_refinement')
        def select(flag: i32, left: checked, right: checked) -> checked:
            return left if flag else right

        @compile(suffix='relation_conditional_shared_refinement_caller')
        def call() -> i32:
            return i32(select(1, checked(3), checked(7)))

        flush_all_pending_outputs()
        self.assertEqual(call(), 3)

    def test_static_default_cannot_forge_refinement(self):
        checked = refined[i32, 'checked']

        @compile(suffix='relation_static_uninitialized_refinement')
        def invalid() -> i32:
            value: static[checked]
            return i32(value)

        self._reject(invalid, 'Cannot implicitly zero-initialize proof-bearing')

    def test_taking_address_borrows_linear_owner(self):
        from pythoc import linear, consume

        @compile(suffix='relation_linear_address_borrow')
        def borrow() -> i32:
            proof = linear()
            address = ptr(proof)
            consume(proof)
            return 7

        flush_all_pending_outputs()
        self.assertEqual(borrow(), 7)

    def test_poly_replacement_invalidates_dynamic_dispatch(self):
        from pythoc.std.poly import Poly
        number = _create_enum_type([('Value', i32, 0)], i32)

        @compile(suffix='relation_poly_original')
        def original(value: i32) -> i32:
            return value + 1

        @compile(suffix='relation_poly_replacement')
        def replacement(value: i32) -> i32:
            return value + 2

        dispatch = Poly(original)
        dispatch.prepare_dynamic_dispatch(number)
        dispatch.append(replacement)

        @compile(suffix='relation_poly_replaced_call')
        def call() -> i32:
            return dispatch(number(number.Value, 40))

        flush_all_pending_outputs()
        self.assertEqual(call(), 42)

    def test_quoted_alias_accepts_pyconst_type_instance(self):
        @compile
        class Holder:
            tag: '_TypeRelationTag'
            value: i32

        @compile(suffix='relation_quoted_type_instance')
        def read() -> i32:
            value: Holder = (7, 42)
            return value.value

        flush_all_pending_outputs()
        self.assertEqual(read(), 42)

    def _check_cimport_enum_representation_matches_c_consumer(self, wide, nested):
        import shutil
        from pythoc.cimport import cimport
        from pythoc.cimport_clang import is_clang_backend_available
        if not is_clang_backend_available() or shutil.which('cc') is None:
            self.skipTest('C compiler and clang bindings are required')
        source = self.tmp_path / 'enum_representation.c'
        if wide:
            source.write_text(
                'enum WideMode { BIG = 1ULL << 40 };\n'
                'struct WideHolder { char prefix; enum WideMode value; char tail; };\n'
                'int verify_wide_enum(struct WideHolder *value) {\n'
                '    return sizeof(*value) * 100 + (value->value == BIG) * 10 + value->tail;\n'
                '}\n'
            )
        else:
            source.write_text(
                'enum SmallMode { SMALL = 1 };\n'
                'struct SmallHolder { char prefix; enum SmallMode value; char tail; };\n'
                'int verify_small_enum(struct SmallHolder *value) {\n'
                '    return sizeof(*value) * 100 + value->value * 10 + value->tail;\n'
                '}\n'
            )
        if nested:
            text = source.read_text()
            declaration = ('enum WideMode { BIG = 1ULL << 40 }' if wide else
                           'enum SmallMode { SMALL = 1 }')
            name = 'WideMode' if wide else 'SmallMode'
            source.write_text(text.replace(declaration + ';\n', '')
                              .replace('enum ' + name + ' value;', declaration + ' value;'))
        bindings = cimport(str(source), compile_sources=True,
                           cflags=[] if wide else ['-fshort-enums'])
        holder_type = bindings.WideHolder if wide else bindings.SmallHolder
        mode_type = bindings.WideMode if wide else bindings.SmallMode
        tag = bindings.BIG if wide else bindings.SMALL
        consumer = bindings.verify_wide_enum if wide else bindings.verify_small_enum

        @compile(suffix=('relation_c_enum_layout', wide, nested))
        def call() -> i32:
            value: holder_type
            value.prefix = 0
            value.value = mode_type(tag)
            value.tail = 7
            return consumer(ptr(value))

        flush_all_pending_outputs()
        self.assertEqual(mode_type.get_size_bytes(), 8 if wide else 1)
        self.assertEqual(call(), 2417 if wide else 317)

    def _check_class_static_cannot_mint_proof_or_own_linear_storage(self, kind):
        from pythoc import linear
        field_type = linear if kind == 'linear' else refined[i32, 'class_checked']
        fields = {'__annotations__': {'value': static[field_type]}}
        if kind == 'refinement_seed':
            fields['value'] = 7
        owner = compile(type('ProofStatic_' + kind, (), fields))

        @compile(suffix='relation_class_static_' + kind)
        def should_fail() -> i32:
            owner.value
            return 0

        self._reject(should_fail, 'linear|zero-initialize|refined type')

    def _check_python_entry_cannot_implicitly_assume_refinement(self, nested):
        from pythoc.python_adapter import function_spec
        checked = refined[i32, 'python_checked']
        parameter = create_struct_type([checked], ['value']) if nested else checked

        @compile(suffix='relation_python_refinement_' + str(nested))
        def requires_proof(value: parameter) -> i32:
            return 42

        flush_all_pending_outputs()
        self.assertIs(function_spec(requires_proof), None)
        with self.assertRaisesRegex(TypeError, 'requires refinement proof'):
            requires_proof({'value': -7} if nested else -7)

    def test_python_entry_can_call_explicit_proof_entry(self):
        checked = refined[i32, 'python_explicit_checked']

        @compile(suffix='relation_python_explicit_proof_kernel')
        def requires_proof(value: checked) -> i32:
            return value

        @compile(suffix='relation_python_explicit_proof_entry')
        def entry(value: i32) -> i32:
            return requires_proof(checked(value))

        flush_all_pending_outputs()
        self.assertEqual(entry(42), 42)

    def _check_pointer_difference_rejects_unrelated_pointees(self, right_type):
        @compile(suffix='relation_pointer_difference_' + right_type.get_type_id())
        def invalid(left: ptr[i32], right: right_type) -> i64:
            return left - right

        self._reject(invalid, 'matching pointee types')

    def test_qualified_pointer_selection_and_difference_execute(self):
        from pythoc import volatile

        @compile(suffix='relation_qualified_pointer_selection')
        def distance(flag: i32) -> i64:
            values: array[i32, 4] = [1, 2, 3, 4]
            left: ptr[const[i32]] = values
            right: ptr[volatile[i32]] = values + 3
            chosen: ptr[volatile[const[i32]]] = left if flag else right
            erased: ptr[const[void]] = left
            restored: ptr[const[i32]] = erased
            return chosen - restored

        flush_all_pending_outputs()
        self.assertEqual(distance(0), 3)
        self.assertEqual(distance(1), 0)

    def _check_indirect_call_preserves_semantic_calling_convention(self, result):
        from pythoc import extern
        from pythoc.callable_lowering import extern_callable_type

        @extern(calling_convention='stdcall', name='relation_stdcall_target')
        def native(value: i32) -> result:
            pass

        signature, _, _ = extern_callable_type(native)

        @compile(suffix=('relation_indirect_stdcall', result))
        def invoke(callback: signature, value: i32) -> result:
            return callback(value)

        flush_all_pending_outputs()
        generated = str(invoke._func_info.llvm_function)
        self.assertIn('call x86_stdcallcc ', generated)
        self.assertIn('load ' + str(signature.get_llvm_type()), generated)
        from llvmlite import binding
        binding.parse_assembly(str(invoke._func_info.llvm_function.module)).verify()

    def test_extern_convention_cannot_disappear_in_ordinary_callback(self):
        from pythoc import extern

        @extern(calling_convention='stdcall', name='relation_stdcall_mismatch')
        def native(value: i32) -> i32:
            pass

        @compile(suffix='relation_extern_stdcall_mismatch')
        def invalid() -> i32:
            callback: func[i32, i32] = native
            return 0

        self._reject(invalid, 'signature|convert')

    def _check_wrapper_reuse_requires_exact_registration_inputs(self, flushed):
        from pythoc import sizeof

        def make(layout, suffix):
            @compile(suffix=suffix)
            def read_size(value: i32) -> i32:
                return value + sizeof(layout)
            return read_size

        suffix = ('relation_registration_type_input', flushed)
        small = create_struct_type([i32])
        first = make(small, suffix)
        identical = make(create_struct_type([i32]), suffix)
        self.assertIs(first._func_info, identical._func_info)
        if flushed:
            flush_all_pending_outputs()
        with self.assertRaisesRegex(TypeError, 'Function specialization collision'):
            make(create_struct_type([i64]), suffix)
        flush_all_pending_outputs()
        self.assertEqual(first(40), 44)

    def test_wrapper_reuse_cannot_replace_new_signature_with_old_one(self):
        def make(parameter):
            @compile(suffix='relation_registration_signature')
            def identity(value: parameter) -> parameter:
                return value
            return identity

        first = make(i32)
        with self.assertRaisesRegex(TypeError, 'Function specialization collision'):
            make(i64)
        flush_all_pending_outputs()
        self.assertEqual(first(42), 42)

    def test_type_inputs_invalidate_cached_object_across_processes(self):
        import os
        import subprocess
        import sys
        from pathlib import Path
        script = FIXTURES / 'type_relation_cache_input.py'
        environment = _subprocess_environment()
        for width, expected in [('32', 4), ('64', 8), ('32', 4)]:
            completed = subprocess.run([sys.executable, str(script), width], cwd=self.tmp_path,
                                       env=environment, capture_output=True, text=True,
                                       check=True, timeout=120)
            self.assertIn('SIZE_RESULT ' + str(expected), completed.stdout)

    def test_linear_value_cannot_be_erased_into_pyconst_storage(self):
        from pythoc import linear
        empty = PythonType.wrap([], is_constant=True)

        @compile(suffix='relation_linear_pyconst_erasure')
        def invalid() -> i32:
            discarded: empty = linear()
            return 0

        self._reject(invalid, 'linear value')

    def test_equivalent_standard_predicates_share_refinement_at_calls(self):
        from pythoc.std.refinement import positive
        first = positive(i32)[1]
        second = positive(i32)[1]

        @compile(suffix='relation_standard_predicate_parameter')
        def accept(value: second) -> i32:
            return value

        @compile(suffix='relation_standard_predicate_call')
        def call() -> i32:
            proof: first = first(42)
            return accept(proof)

        flush_all_pending_outputs()
        self.assertEqual(call(), 42)

    def test_standard_predicate_families_keep_independent_runtime_behavior(self):
        from pythoc.std.refinement import positive, nonnegative, nonzero, in_range
        is_positive = positive(i32)[0]
        is_nonnegative = nonnegative(i32)[0]
        is_nonzero = nonzero(i32)[0]
        is_bounded = in_range(i32, 0, 10)[0]

        @compile(suffix='relation_predicate_family_behavior')
        def check(value: i32) -> i32:
            return (i32(is_positive(value)) * 1000 + i32(is_nonnegative(value)) * 100
                    + i32(is_nonzero(value)) * 10 + i32(is_bounded(value)))

        flush_all_pending_outputs()
        self.assertEqual(check(0), 101)
        self.assertEqual(check(-1), 10)
        self.assertEqual(check(2), 1111)

    def test_const_refinement_base_cannot_be_reassigned(self):
        checked = refined[const[i32], 'const_base']

        @compile(suffix='relation_const_refinement_base')
        def invalid() -> i32:
            value: checked = checked(1)
            value = checked(2)
            return value

        self._reject(invalid, 'Cannot reassign to const', RuntimeError)

    def test_volatile_refinement_base_replays_volatile_load(self):
        from pythoc import volatile
        os.chdir(self.tmp_path)
        checked = refined[volatile[i32], 'volatile_base']

        @compile(suffix='relation_volatile_refinement_base')
        def access(pointer: ptr[checked]) -> i32:
            return pointer[0]

        flush_all_pending_outputs()
        self.assertIn('load volatile i32', str(access._func_info.llvm_function))

    def test_atomic_store_cannot_hide_const_behind_refinement(self):
        from pythoc import atomic_store
        checked = refined[const[i32], 'atomic_const_base']

        @compile(suffix='relation_atomic_refinement_const_base')
        def invalid(pointer: ptr[checked], value: checked) -> i32:
            atomic_store(pointer, value)
            return 0

        self._reject(invalid, 'const', TypeError)

    def test_whole_record_assignment_cannot_overwrite_const_subobject(self):
        import shutil
        import subprocess
        if shutil.which('cc') is not None:
            source = self.tmp_path / 'const_subobject.c'
            source.write_text(
                'struct Record { const int fixed; int mutable; };\n'
                'int check(void) { struct Record value = {1, 2};\n'
                '#if VIOLATE\n value = (struct Record){3, 4};\n'
                '#else\n value.mutable = 41;\n#endif\n'
                'return value.fixed + value.mutable; }\n', encoding='ascii',
            )
            command = ['cc', '-std=c11', '-c', str(source), '-o', str(self.tmp_path / 'record.o')]
            valid = subprocess.run(
                command + ['-DVIOLATE=0'], capture_output=True, timeout=120)
            invalid = subprocess.run(
                command + ['-DVIOLATE=1'], capture_output=True, timeout=120)
            self.assertEqual(valid.returncode, 0, valid.stderr.decode())
            self.assertNotEqual(invalid.returncode, 0)
        record = create_struct_type([const[i32], i32], ['fixed', 'mutable'])

        @compile(suffix='relation_record_const_subobject')
        def invalid() -> i32:
            value: record = (1, 2)
            value = (3, 4)
            return value.mutable

        self._reject(invalid, 'Cannot reassign to const', RuntimeError)

    def test_const_subobject_allows_initialization_and_mutable_sibling_updates(self):
        record = create_struct_type([const[i32], i32], ['fixed', 'mutable'])

        @compile(suffix='relation_record_mutable_sibling')
        def run() -> i32:
            value: record = (1, 2)
            value.mutable = 41
            return value.fixed + value.mutable

        flush_all_pending_outputs()
        self.assertEqual(run(), 42)

    def test_conversion_entry_matrix(self):
        """Check policy at real compiler boundaries, then execute every positive cell."""
        for case in CASES:
            for entry in ENTRIES:
                with self.subTest(relation=case.name, entry=entry):
                    conversion = build_conversion(case, entry)
                    if not case.allowed(entry):
                        self._reject(conversion, case.rejection(entry))
                        continue
                    probe = build_probe(case, entry, conversion)
                    try:
                        flush_all_pending_outputs()
                    except Exception:
                        clear_failed_group(probe._binding.group_key)
                        clear_failed_group(conversion._binding.group_key)
                        raise
                    for flag in (0, 1):
                        expected = case.expected
                        if ((entry == 'conditional' and not flag)
                                or (entry == 'conditional_reversed' and flag)):
                            expected = case.peer_expected
                        with self.subTest(flag=flag):
                            self.assertEqual(probe(flag), expected)

    def test_cimport_enum_representation_matches_c_consumer(self):
        for wide in [False, True]:
            for nested in [False, True]:
                with self.subTest(wide=wide, nested=nested):
                    with tempfile.TemporaryDirectory(prefix='type-matrix-case-') as temp:
                        self.tmp_path = Path(temp)
                        self._check_cimport_enum_representation_matches_c_consumer(wide, nested)

    def test_class_static_cannot_mint_proof_or_own_linear_storage(self):
        for kind in ['refinement_default', 'refinement_seed', 'linear']:
            with self.subTest(kind=kind):
                self._check_class_static_cannot_mint_proof_or_own_linear_storage(kind)

    def test_python_entry_cannot_implicitly_assume_refinement(self):
        for nested in [False, True]:
            with self.subTest(nested=nested):
                self._check_python_entry_cannot_implicitly_assume_refinement(nested)

    def test_pointer_difference_rejects_unrelated_pointees(self):
        for right_type in [ptr[i64], ptr[void], ptr[refined[i32, 'difference_checked']]]:
            with self.subTest(right_type=right_type):
                self._check_pointer_difference_rejects_unrelated_pointees(right_type)

    def test_indirect_call_preserves_semantic_calling_convention(self):
        for result in [i32, create_struct_type([i32, i32]),
                                  create_struct_type([i64, i64, i64])]:
            with self.subTest(result=result):
                with tempfile.TemporaryDirectory(prefix='type-matrix-case-') as temp:
                    with chdir(temp):
                        self._check_indirect_call_preserves_semantic_calling_convention(result)

    def test_wrapper_reuse_requires_exact_registration_inputs(self):
        for flushed in [False, True]:
            with self.subTest(flushed=flushed):
                self._check_wrapper_reuse_requires_exact_registration_inputs(flushed)


if __name__ == '__main__':
    unittest.main()
