"""Full conversion entry coverage with in-memory checks and one native batch."""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import inspect
import textwrap
import unittest

from llvmlite import binding

from pythoc import (
    array, compile, const, f32, f64, func, i8, i32, i64, meta, ptr, refined,
    simd, simd_reduce_add, struct, u1, u32, union, void, volatile,
)
from pythoc import bool as pc_bool
from pythoc.builtin_entities.enum import _create_enum_type
from pythoc.builtin_entities.struct import create_struct_type
from pythoc.compiler import LLVMCompiler
from pythoc.session import CompileSession
from test.utils.test_utils import DeferredTestCase


ENTRIES = (
    'declaration', 'assignment', 'return', 'argument', 'indirect_argument',
    'conditional', 'conditional_reversed', 'sequence', 'mapping', 'constructor',
)


@dataclass(frozen=True)
class RelationCase:
    name: str
    source: object
    target: object
    family: str
    implicit: bool
    constructor: bool
    source_value: object = 37
    peer_value: object = 11
    expected: int = 37
    peer_expected: int = 11

    def allowed(self, entry):
        return self.constructor if entry == 'constructor' else self.implicit

    def rejection(self, entry):
        if entry == 'constructor':
            if self.family in ('record', 'pointer_record', 'union'):
                return 'takes no arguments'
            if self.family == 'enum':
                return 'requires a known valid variant tag'
        return 'Cannot (?:implicitly )?(?:convert|cast)|No common type'


Checked = refined[i32, 'matrix_checked']
OtherProof = refined[i32, 'matrix_other']
Stronger = refined[i32, 'matrix_checked', 'matrix_other']
Record = struct['value': i32]
PointerRecord = struct['value': ptr[i32]]
ConstPointerRecord = struct['value': ptr[const[i32]]]
CheckedRecord = struct['value': Checked]
Enum = _create_enum_type([('Value', i32, 0), ('Empty', None, 1)], i32)
AlternateConvention = func[i32, i32]
AlternateConvention.calling_convention = 'stdcall'
VariadicCallback = func[i32, i32]
VariadicCallback.has_llvm_varargs = True

# implicit and constructor are deliberately separate. A refined scalar
# constructor is a trusted proof entry. Record, union and array constructors
# do not accept a value argument; SIMD constructors splat a scalar, and enum
# constructors require a tag and its declared payload.
CASES = (
    RelationCase('integer_identity', i32, i32, 'scalar', True, True),
    RelationCase('boolean_integer', pc_bool, i32, 'scalar', True, True, 1, 0, 1, 0),
    RelationCase('unsigned_one_bit_widen', u1, i64, 'scalar', True, True, 1, 0, 1, 0),
    RelationCase('integer_widen', i8, i64, 'scalar', True, True, -37, 11, -37),
    RelationCase('integer_narrow', i64, i8, 'scalar', True, True, 293, 11),
    RelationCase('integer_signed_to_unsigned', i32, u32, 'scalar', True, True,
                 -1, 11, 4294967295),
    RelationCase('integer_unsigned_to_signed', u32, i32, 'scalar', True, True,
                 4294967295, 11, -1),
    RelationCase('integer_float', i32, f64, 'scalar', True, True),
    RelationCase('float_integer', f64, i32, 'scalar', True, True, 37.75),
    RelationCase('float_widen', f32, f64, 'scalar', True, True),
    RelationCase('value_const_erasure', const[i32], i32, 'scalar', True, True),
    RelationCase('value_volatile_erasure', volatile[i32], i32, 'scalar', True, True),
    RelationCase('proof_identity', Checked, Checked, 'scalar', True, True),
    RelationCase('erase_proof', Checked, i32, 'scalar', True, True),
    RelationCase('erase_proof_and_widen', Checked, i64, 'scalar', True, True),
    RelationCase('invent_proof', i32, Checked, 'scalar', False, True),
    RelationCase('replace_proof', Checked, OtherProof, 'scalar', False, True),
    RelationCase('weaken_proof', Stronger, Checked, 'scalar', True, True),
    RelationCase('strengthen_proof', Checked, Stronger, 'scalar', False, True),
    RelationCase('qualified_proof', const[Checked], Checked, 'scalar', True, True),
    RelationCase('pointer_identity', ptr[i32], ptr[i32], 'pointer', True, True),
    RelationCase('add_const', ptr[i32], ptr[const[i32]], 'pointer', True, True),
    RelationCase('drop_const', ptr[const[i32]], ptr[i32], 'pointer', False, True),
    RelationCase('add_volatile', ptr[i32], ptr[volatile[i32]], 'pointer', True, True),
    RelationCase('drop_volatile', ptr[volatile[i32]], ptr[i32], 'pointer', False, True),
    RelationCase('add_both_qualifiers', ptr[i32], ptr[const[volatile[i32]]],
                 'pointer', True, True),
    RelationCase('exchange_qualifiers', ptr[const[i32]], ptr[volatile[i32]],
                 'pointer', False, True),
    RelationCase('pointer_storage_const', const[ptr[i32]], ptr[i32],
                 'pointer', True, True),
    RelationCase('erase_to_void_pointer', ptr[i32], ptr[void], 'pointer', True, True),
    RelationCase('restore_void_pointer', ptr[void], ptr[i32], 'pointer', True, True),
    RelationCase('void_pointer_keeps_const', ptr[const[i32]], ptr[const[void]],
                 'pointer', True, True),
    RelationCase('void_pointer_cannot_erase_const', ptr[const[i32]], ptr[void],
                 'pointer', False, True),
    RelationCase('void_pointer_cannot_restore_mutable', ptr[const[void]], ptr[i32],
                 'pointer', False, True),
    RelationCase('unrelated_pointees', ptr[i32], ptr[i64], 'pointer', False, True),
    RelationCase('pointee_proof_is_not_value_proof', ptr[Checked], ptr[i32],
                 'pointer', False, True),
    RelationCase('indirection_identity', ptr[ptr[i32]], ptr[ptr[i32]],
                 'indirection', True, True),
    RelationCase('inner_const_is_not_covariant', ptr[ptr[i32]], ptr[ptr[const[i32]]],
                 'indirection', False, True),
    RelationCase('inner_const_reverse', ptr[ptr[const[i32]]], ptr[ptr[i32]],
                 'indirection', False, True),
    RelationCase('record_identity', Record, Record, 'record', True, False),
    RelationCase('record_field_labels', Record, struct['renamed': i32],
                 'record', True, False),
    RelationCase('record_field_widen', Record, struct['value': i64],
                 'record', True, False),
    RelationCase('record_erase_proof', CheckedRecord, Record, 'record', True, False),
    RelationCase('record_invent_proof', Record, CheckedRecord, 'record', False, False),
    RelationCase('record_add_const', PointerRecord, ConstPointerRecord,
                 'pointer_record', True, False),
    RelationCase('record_drop_const', ConstPointerRecord, PointerRecord,
                 'pointer_record', False, False),
    RelationCase('record_field_count', Record, struct[i32, i32],
                 'record', False, False),
    RelationCase('same_display_name_record_pointer',
                 ptr[create_struct_type([i32], ['value'],
                                        python_class=type('MatrixRecord', (), {}))],
                 ptr[create_struct_type([i64], ['value'],
                                        python_class=type('MatrixRecord', (), {}))],
                 'pointer', False, True),
    RelationCase('array_pointer_identity', ptr[array[i32, 2]], ptr[array[i32, 2]],
                 'array_pointer', True, True),
    RelationCase('array_pointer_extent', ptr[array[i32, 2]], ptr[array[i32, 3]],
                 'array_pointer', False, True),
    RelationCase('array_pointer_element', ptr[array[i32, 2]], ptr[array[u32, 2]],
                 'array_pointer', False, True),
    RelationCase('array_decay', array[i32, 2], ptr[i32], 'decay', True, True),
    RelationCase('const_array_decay', const[array[i32, 2]], ptr[const[i32]],
                 'decay', True, True),
    RelationCase('const_array_decay_to_mutable', const[array[i32, 2]], ptr[i32],
                 'decay', False, True),
    RelationCase('volatile_array_decay', volatile[array[i32, 2]], ptr[volatile[i32]],
                 'decay', True, True),
    RelationCase('volatile_array_decay_to_plain', volatile[array[i32, 2]], ptr[i32],
                 'decay', False, True),
    RelationCase('multidimensional_decay', array[i32, 2, 2], ptr[array[i32, 2]],
                 'decay_2d', True, True),
    RelationCase('multidimensional_wrong_stride', array[i32, 2, 2], ptr[array[i32, 3]],
                 'decay_2d', False, True),
    RelationCase('callback_identity', func[i32, i32], func[i32, i32],
                 'callback', True, True),
    RelationCase('callback_labels', func['left': i32, i32], func['right': i32, i32],
                 'callback', True, True),
    RelationCase('callback_parameter', func[i32, i32], func[i64, i32],
                 'callback', False, False),
    RelationCase('callback_result', func[i32, i32], func[i32, i64],
                 'callback', False, False),
    RelationCase('callback_arity', func[i32, i32], func[i32, i32, i32],
                 'callback', False, False),
    RelationCase('callback_convention', AlternateConvention, func[i32, i32],
                 'callback', False, False),
    RelationCase('callback_convention_reverse', func[i32, i32], AlternateConvention,
                 'callback', False, False),
    RelationCase('callback_varargs', VariadicCallback, func[i32, i32],
                 'callback', False, False),
    RelationCase('callback_varargs_reverse', func[i32, i32], VariadicCallback,
                 'callback', False, False),
    RelationCase('enum_identity', Enum, Enum, 'enum', True, False),
    RelationCase('enum_variant_tags', Enum,
                 _create_enum_type([('Value', i32, 2), ('Empty', None, 1)], i32),
                 'enum', False, False),
    RelationCase('enum_variant_names', Enum,
                 _create_enum_type([('Other', i32, 0), ('Empty', None, 1)], i32),
                 'enum', False, False),
    RelationCase('enum_payload', Enum,
                 _create_enum_type([('Value', u32, 0), ('Empty', None, 1)], i32),
                 'enum', False, False),
    RelationCase('union_identity', union['value': i32], union['value': i32],
                 'union', True, False),
    RelationCase('union_member_type', union['value': i32], union['value': u32],
                 'union', False, False),
    RelationCase('simd_identity', simd[i32, 4], simd[i32, 4],
                 'simd', True, False, expected=148, peer_expected=44),
    RelationCase('simd_lane_count', simd[i32, 4], simd[i32, 8], 'simd', False, False),
    RelationCase('simd_lane_type', simd[i32, 4], simd[u32, 4], 'simd', False, False),
)


def increment(value: i32) -> i32:
    return value + 1


def peer_callback(value: i32) -> i32:
    return value - 25


def accept(value: Target) -> Target:
    return value


def entry_declaration(value: Source, peer: Target, flag: i32) -> Target:
    result: Target = value
    return result


def entry_assignment(value: Source, peer: Target, flag: i32) -> Target:
    result: Target
    result = value
    return result


def entry_return(value: Source, peer: Target, flag: i32) -> Target:
    return value


def entry_argument(value: Source, peer: Target, flag: i32) -> Target:
    return accept(value)


def entry_indirect_argument(value: Source, peer: Target, flag: i32) -> Target:
    callback: func[Target, Target] = accept
    return callback(value)


def entry_conditional(value: Source, peer: Target, flag: i32) -> Target:
    return value if flag else peer


def entry_conditional_reversed(value: Source, peer: Target, flag: i32) -> Target:
    return peer if flag else value


def entry_sequence(value: Source, peer: Target, flag: i32) -> Target:
    holder: Carrier = (value,)
    return holder[0]


def entry_mapping(value: Source, peer: Target, flag: i32) -> Target:
    holder: NamedCarrier = {'value': value}
    return holder.value


def entry_constructor(value: Source, peer: Target, flag: i32) -> Target:
    return Target(value)


def array_declaration(value: Source, peer: Target, flag: i32) -> Target:
    result: Target = value[0]
    return result


def array_assignment(value: Source, peer: Target, flag: i32) -> Target:
    result: Target
    result = value[0]
    return result


def array_return(value: Source, peer: Target, flag: i32) -> Target:
    return value[0]


def array_argument(value: Source, peer: Target, flag: i32) -> Target:
    return accept(value[0])


def array_indirect_argument(value: Source, peer: Target, flag: i32) -> Target:
    callback: func[Target, Target] = accept
    return callback(value[0])


def array_conditional(value: Source, peer: Target, flag: i32) -> Target:
    return value[0] if flag else peer


def array_conditional_reversed(value: Source, peer: Target, flag: i32) -> Target:
    return peer if flag else value[0]


def array_sequence(value: Source, peer: Target, flag: i32) -> Target:
    holder: Carrier = (value[0],)
    return holder[0]


def array_mapping(value: Source, peer: Target, flag: i32) -> Target:
    holder: NamedCarrier = {'value': value[0]}
    return holder.value


def array_constructor(value: Source, peer: Target, flag: i32) -> Target:
    return Target(value[0])


def probe_scalar(flag: i32) -> i64:
    value: Source = Source(first)
    peer: Target = Target(second)
    return i64(convert(value, peer, flag))


def probe_pointer(flag: i32) -> i64:
    left: i32 = first
    right: i32 = second
    value: Source = Source(ptr(left))
    peer: Target = Target(ptr(right))
    result = convert(value, peer, flag)
    return ptr[i32](result)[0]


def probe_indirection(flag: i32) -> i64:
    left: i32 = first
    right: i32 = second
    left_pointer = ptr(left)
    right_pointer = ptr(right)
    value: Source = Source(ptr(left_pointer))
    peer: Target = Target(ptr(right_pointer))
    result = convert(value, peer, flag)
    return ptr[ptr[i32]](result)[0][0]


def probe_record(flag: i32) -> i64:
    value: Source = (SourceField(first),)
    peer: Target = (TargetField(second),)
    result = convert(value, peer, flag)
    return i64(result[0])


def probe_pointer_record(flag: i32) -> i64:
    left: i32 = first
    right: i32 = second
    value: Source = (SourceField(ptr(left)),)
    peer: Target = (TargetField(ptr(right)),)
    result = convert(value, peer, flag)
    return ptr[i32](result[0])[0]


def probe_array_pointer(flag: i32) -> i64:
    left: array[i32, 2] = [first, 19]
    right: array[i32, 2] = [second, 23]
    value: Source = Source(ptr(left))
    peer: Target = Target(ptr(right))
    result = convert(value, peer, flag)
    return ptr[array[i32, 2]](result)[0][0]


def probe_decay(flag: i32) -> i64:
    value: Source = [first, 19]
    other: array[i32, 2] = [second, 23]
    peer: Target = Target(other)
    result = convert(ptr(value), peer, flag)
    return ptr[i32](result)[0]


def probe_decay_2d(flag: i32) -> i64:
    value: Source = [[first, 19], [29, 31]]
    other: array[i32, 2, 2] = [[second, 23], [41, 43]]
    peer: Target = Target(other)
    result = convert(ptr(value), peer, flag)
    return ptr[array[i32, 2]](result)[0][0]


def probe_callback(flag: i32) -> i64:
    value: Source = increment
    peer: Target = peer_callback
    result = convert(value, peer, flag)
    return result(36)


def probe_enum(flag: i32) -> i64:
    value: Source = Source(0, first)
    peer: Target = Target(0, second)
    result = convert(value, peer, flag)
    match result:
        case (Target.Value, payload):
            return payload
        case _:
            return -999


def probe_union(flag: i32) -> i64:
    value: Source
    value.value = first
    peer: Target
    peer.value = second
    result = convert(value, peer, flag)
    return result.value


def probe_simd(flag: i32) -> i64:
    value: Source = Source(first)
    peer: Target = Target(second)
    result = convert(value, peer, flag)
    return simd_reduce_add(result)


def function_ast(function):
    lines, start = inspect.getsourcelines(function)
    node = ast.parse(textwrap.dedent(''.join(lines))).body[0]
    return ast.increment_lineno(node, start - 1)


ENTRY_FORMS = {
    'declaration': function_ast(entry_declaration),
    'assignment': function_ast(entry_assignment),
    'return': function_ast(entry_return),
    'argument': function_ast(entry_argument),
    'indirect_argument': function_ast(entry_indirect_argument),
    'conditional': function_ast(entry_conditional),
    'conditional_reversed': function_ast(entry_conditional_reversed),
    'sequence': function_ast(entry_sequence),
    'mapping': function_ast(entry_mapping),
    'constructor': function_ast(entry_constructor),
}

ARRAY_FORMS = {
    'declaration': function_ast(array_declaration),
    'assignment': function_ast(array_assignment),
    'return': function_ast(array_return),
    'argument': function_ast(array_argument),
    'indirect_argument': function_ast(array_indirect_argument),
    'conditional': function_ast(array_conditional),
    'conditional_reversed': function_ast(array_conditional_reversed),
    'sequence': function_ast(array_sequence),
    'mapping': function_ast(array_mapping),
    'constructor': function_ast(array_constructor),
}

PROBE_FORMS = {
    'scalar': function_ast(probe_scalar),
    'pointer': function_ast(probe_pointer),
    'indirection': function_ast(probe_indirection),
    'record': function_ast(probe_record),
    'pointer_record': function_ast(probe_pointer_record),
    'array_pointer': function_ast(probe_array_pointer),
    'decay': function_ast(probe_decay),
    'decay_2d': function_ast(probe_decay_2d),
    'callback': function_ast(probe_callback),
    'enum': function_ast(probe_enum),
    'union': function_ast(probe_union),
    'simd': function_ast(probe_simd),
}

ACCEPT_FORM = function_ast(accept)
ARRAY_FAMILIES = ('decay', 'decay_2d')


def conversion_context(case):
    source = ptr[case.source] if case.family in ARRAY_FAMILIES else case.source
    return {**globals(), 'Source': source, 'Target': case.target,
            'Carrier': struct[case.target],
            'NamedCarrier': struct['value': case.target]}


def compile_internal(form, name, namespace, parameters, result):
    return meta.compile_ast(
        form, name=name, user_globals=namespace,
        param_types=parameters, return_type=result,
        source_file=__file__, linkage='internal', attrs={'noinline'},
    )


class TypeConversionMatrixTest(DeferredTestCase):
    def setUp(self):
        session = CompileSession()
        session.__enter__()
        self.addCleanup(session.__exit__, None, None, None)

    def test_all_conversion_entries(self):
        accepted = rejected = 0
        for number, case in enumerate(CASES):
            namespace = conversion_context(case)
            namespace['accept'] = compile_internal(
                ACCEPT_FORM, f'frontend_accept_{number}', namespace,
                {'value': case.target}, case.target)
            forms = ARRAY_FORMS if case.family in ARRAY_FAMILIES else ENTRY_FORMS
            for entry in ENTRIES:
                with self.subTest(relation=case.name, entry=entry):
                    compiler = LLVMCompiler(user_globals=namespace)
                    form = copy.deepcopy(forms[entry])
                    if case.allowed(entry):
                        compiler.compile_function_from_ast(form)
                        binding.parse_assembly(compiler.get_ir()).verify()
                        accepted += 1
                    else:
                        with self.assertRaisesRegex(TypeError, case.rejection(entry)):
                            compiler.compile_function_from_ast(form)
                        rejected += 1
        self.assertEqual((accepted, rejected), (420, 330))

    def test_all_successful_values_in_one_native_batch(self):
        checks = []
        labels = []
        callbacks = {
            'increment': compile_internal(
                function_ast(increment), 'matrix_increment', globals(),
                {'value': i32}, i32),
            'peer_callback': compile_internal(
                function_ast(peer_callback), 'matrix_peer_callback', globals(),
                {'value': i32}, i32),
        }
        for number, case in enumerate(CASES):
            entries = [entry for entry in ENTRIES if case.allowed(entry)]
            if not entries:
                continue
            namespace = {**conversion_context(case), **callbacks}
            namespace['accept'] = compile_internal(
                ACCEPT_FORM, f'accept_{number}', namespace,
                {'value': case.target}, case.target)
            forms = ARRAY_FORMS if case.family in ARRAY_FAMILIES else ENTRY_FORMS
            for entry in entries:
                converter = compile_internal(
                    forms[entry], f'convert_{number}_{entry}', namespace,
                    {'value': namespace['Source'], 'peer': case.target, 'flag': i32},
                    case.target)
                probe_context = {
                    **namespace, 'Source': case.source, 'convert': converter,
                    'first': case.source_value, 'second': case.peer_value,
                }
                if case.family in ('record', 'pointer_record'):
                    probe_context.update(SourceField=case.source._field_types[0],
                                         TargetField=case.target._field_types[0])
                probe = compile_internal(
                    PROBE_FORMS[case.family], f'probe_{number}_{entry}',
                    probe_context, {'flag': i32}, i64)
                expected = [case.expected, case.expected]
                if entry == 'conditional':
                    expected[0] = case.peer_expected
                elif entry == 'conditional_reversed':
                    expected[1] = case.peer_expected
                checks.append((probe, expected[0], expected[1]))
                labels.append((case.name, entry))
        self.assertEqual(len(checks), 420)

        @compile
        def run_matrix(flag: i32) -> i32:
            for index, row in enumerate(checks):
                probe, expected_false, expected_true = row
                wanted: i64 = i64(expected_true) if flag else i64(expected_false)
                if probe(flag) != wanted:
                    return index + 1
            return 0

        for flag in (0, 1):
            with self.subTest(flag=flag):
                failure = int(run_matrix(flag))
                if failure:
                    relation, entry = labels[failure - 1]
                    self.fail(f'{relation}/{entry}: unexpected native value')


if __name__ == '__main__':
    unittest.main()
