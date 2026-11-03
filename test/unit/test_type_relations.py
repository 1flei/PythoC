"""PythoC type objects are the sole source for identity and conversion."""

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from llvmlite import ir

from pythoc import array, const, func, i32, i64, ptr, refined, void, volatile
from pythoc.builtin_entities.enum import _create_enum_type
from pythoc.builtin_entities.python_type import PythonType
from pythoc.builtin_entities.struct import create_struct_type
from pythoc.session import CompileSession
from pythoc.type_descriptor import can_convert, same_type, type_key
from pythoc.type_graph import validate_type_graph
from pythoc.type_id import get_type_id
from pythoc.type_converter import ImplicitCoercer, TypeConverter
from pythoc.valueref import wrap_value
from pythoc.build.artifact_identity import verify_artifact_identity
from pythoc.utils.naming import normalize_suffix


class TypeRelationsTest(unittest.TestCase):
    def setUp(self):
        self.session = CompileSession()
        self.session.__enter__()
        self.addCleanup(self.session.__exit__, None, None, None)

    def test_long_id_iff_exact_type_for_representative_constructors(self):
        types = (i32, i64, const[i32], volatile[i32], ptr[i32],
                 ptr[const[i32]], ptr[volatile[i32]],
                 refined[i32, 'checked'], PythonType.wrap(1),
                 PythonType.wrap(True), PythonType.wrap('1'))
        for left in types:
            for right in types:
                with self.subTest(left=left, right=right):
                    self.assertEqual(get_type_id(left) == get_type_id(right),
                                     same_type(left, right))
        self.assertEqual(get_type_id(const[i32]), const[i32].get_type_id())
        self.assertNotEqual(get_type_id(PythonType.wrap(1)),
                            get_type_id(PythonType.wrap(True)))
        self.assertNotEqual(get_type_id(PythonType.wrap(float('nan'))),
                            get_type_id(PythonType.wrap(float('nan'))))

    def test_same_display_name_is_not_pointer_compatibility(self):
        left = create_struct_type([i32], ['value'],
                                  python_class=type('SameName', (), {}))
        right = create_struct_type([i64], ['value'],
                                   python_class=type('SameName', (), {}))
        self.assertEqual(left.get_name(), right.get_name())
        self.assertFalse(same_type(left, right))
        self.assertFalse(ImplicitCoercer.are_compatible_pointers(ptr[left], ptr[right]))
        self.assertTrue(can_convert(ptr[left], ptr[right], mode='explicit'))

    def test_pointee_qualifiers_are_directional(self):
        self.assertTrue(can_convert(ptr[i32], ptr[const[i32]]))
        self.assertFalse(can_convert(ptr[const[i32]], ptr[i32]))
        self.assertFalse(can_convert(ptr[volatile[i32]], ptr[i32]))
        self.assertTrue(can_convert(ptr[const[i32]], ptr[i32], mode='explicit'))

    def test_refinement_requires_constraint_proof_through_qualifier(self):
        positive = refined[i32, 'positive']
        negative = refined[i32, 'negative']
        self.assertFalse(can_convert(i32, const[positive]))
        self.assertFalse(can_convert(positive, negative))
        self.assertFalse(can_convert(const[positive], const[negative], mode='explicit'))
        self.assertTrue(can_convert(positive, i32))
        self.assertTrue(can_convert(positive, const[positive]))

    def test_unapplied_func_constructor_is_not_a_data_pointer_type(self):
        self.assertFalse(same_type(func, ptr[void]))
        self.assertNotEqual(get_type_id(func), get_type_id(ptr[void]))
        self.assertFalse(can_convert(func, ptr[void]))
        with self.assertRaisesRegex(TypeError, 'specialized func'):
            func.get_llvm_type()

    def test_type_recognition_does_not_probe_arbitrary_attribute_protocols(self):
        from pythoc.type_descriptor import specialization_key
        from pythoc import effect
        namespace = effect
        self.assertEqual(specialization_key(namespace), specialization_key(namespace))
        self.assertNotEqual(specialization_key(namespace)[0], 'pc-type')

    def test_function_names_are_identity_but_not_conversion(self):
        named = func['value': i32, i32]
        unnamed = func[i32, i32]
        self.assertFalse(same_type(named, unnamed))
        self.assertTrue(can_convert(named, unnamed))
        self.assertFalse(can_convert(func[i64, i32], unnamed))

    def test_callable_wrapper_admission_checks_signature(self):
        callable_type = func[i32, i32]
        wrapper = SimpleNamespace(
            _is_compiled=True,
            _func_info=SimpleNamespace(callable_pc_type=callable_type),
        )
        source = PythonType.wrap(wrapper)
        self.assertTrue(can_convert(source, callable_type))
        self.assertFalse(can_convert(source, func[i64, i64]))

    def test_null_pointer_is_valid_function_pointer_value(self):
        null = wrap_value(ir.Constant(ir.PointerType(ir.IntType(8)), None),
                          kind='value', type_hint=ptr[void])
        self.assertTrue(can_convert(ptr[void], func[i32, i32], value=null))
        self.assertFalse(can_convert(ptr[void], func[i32, i32]))

    def test_ternary_pointer_common_type_keeps_qualifiers(self):
        self.assertTrue(same_type(
            TypeConverter.infer_unified_pc_type(ptr[const[i32]], ptr[i32]),
            ptr[const[i32]]))
        self.assertTrue(same_type(
            TypeConverter.infer_unified_pc_type(ptr[i32], ptr[const[i32]]),
            ptr[const[i32]]))

    def test_numeric_common_type_is_symmetric_and_backend_independent(self):
        from pythoc import i8, u8, u32, u64, f32, f64
        from pythoc.type_descriptor import common_numeric_type
        cases = [(i32, u32, u32), (i64, u32, i64), (i32, u64, u64),
                 (i8, u8, u8), (const[i32], refined[i32, 'positive'], i32),
                 (f32, f64, f64), (f32, i64, f32)]
        for left, right, expected in cases:
            with self.subTest(left=left, right=right):
                with patch.object(i32, 'get_llvm_type',
                                  side_effect=AssertionError('backend called')):
                    self.assertTrue(same_type(common_numeric_type(left, right), expected))
                    self.assertTrue(same_type(common_numeric_type(right, left), expected))
                    self.assertTrue(same_type(
                        TypeConverter.infer_unified_pc_type(left, right), expected))
        self.assertIsNone(common_numeric_type(ptr[i32], f64))

    def test_pointer_comparison_uses_semantic_pointees(self):
        from pythoc.type_descriptor import can_compare_pointers
        self.assertTrue(can_compare_pointers(ptr[const[i32]], ptr[volatile[i32]]))
        self.assertFalse(can_compare_pointers(ptr[refined[i32, 'checked']], ptr[i32]))
        with patch.object(i32, 'get_llvm_type',
                          side_effect=AssertionError('backend called')):
            self.assertTrue(can_compare_pointers(ptr[i32], ptr[i32]))

    def test_void_pointer_qualifiers_and_common_selection_share_relations(self):
        from pythoc.type_descriptor import common_value_type, can_subtract_pointers
        from pythoc import volatile
        combined = ptr[volatile[const[i32]]]
        for source in (ptr[const[i32]], const[array[i32, 2]]):
            self.assertTrue(can_convert(source, ptr[const[void]]))
            self.assertFalse(can_convert(source, ptr[void]))
        self.assertFalse(can_convert(ptr[const[void]], ptr[i32]))
        with patch.object(i32, 'get_llvm_type', side_effect=AssertionError('backend called')):
            for left, right in ((ptr[const[i32]], ptr[volatile[i32]]),
                                (ptr[const[void]], ptr[volatile[i32]])):
                self.assertTrue(same_type(common_value_type(left, right), combined))
                self.assertTrue(same_type(common_value_type(right, left), combined))
            self.assertTrue(can_subtract_pointers(ptr[const[i32]], ptr[volatile[i32]]))
            self.assertFalse(can_subtract_pointers(ptr[i32], ptr[i64]))
            self.assertFalse(can_subtract_pointers(ptr[i32], ptr[void]))
        self.assertFalse(can_convert(func[i32, i32], ptr[i32]))
        self.assertTrue(can_convert(func[i32, i32], ptr[void]))
        self.assertTrue(can_convert(func[i32, i32], ptr[i32], mode='explicit'))

    def test_stable_identity_and_artifact_collision_across_processes(self):
        import os
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'identity_process.py'
            path.write_text(
                "import json, sys\n"
                "from pythoc import array, const, i32, i64, ptr, refined\n"
                "from pythoc.type_id import get_type_id\n"
                "from pythoc.std.refinement import nonnull, positive, in_range\n"
                "from pythoc.std.float_types import f128\n"
                "from pythoc.builtin_entities.struct import create_struct_type\n"
                "from pythoc.build.artifact_identity import verify_artifact_identity\n"
                "record = create_struct_type([ptr[i32]], ['value'])\n"
                "types = [i32, ptr[const[i32]], array[i32, 3], record, refined[i32, 'checked'], "
                "nonnull(ptr[i32])[1], positive(i32)[1], in_range(i32, 0, 10)[1], f128]\n"
                "opaque = refined[i32, lambda value: value > 0]\n"
                "suffix = ptr[i64] if sys.argv[2] == 'collision' else ptr[i32]\n"
                "verify_artifact_identity(sys.argv[1] + '/short.o', sys.argv[1] + '/short.so', "
                "scope=None, suffix=suffix, effect_suffix=None)\n"
                "print(json.dumps([[get_type_id(t) for t in types], get_type_id(opaque)]))\n",
                encoding='ascii',
            )
            environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
            command = [sys.executable, str(path), directory]
            first = subprocess.run(command + ['first'], env=environment,
                                   capture_output=True, text=True, check=True)
            second = subprocess.run(command + ['repeat'], env=environment,
                                    capture_output=True, text=True, check=True)
            import json
            first_ids, first_opaque = json.loads(first.stdout)
            second_ids, second_opaque = json.loads(second.stdout)
            self.assertEqual(first_ids, second_ids)
            self.assertNotEqual(first_opaque, second_opaque)
            collision = subprocess.run(command + ['collision'], env=environment,
                                       capture_output=True, text=True)
            self.assertNotEqual(collision.returncode, 0)
            self.assertIn('Artifact filename collision', collision.stderr)

    def test_refinement_erasure_composes_with_underlying_conversion(self):
        checked = refined[i32, 'checked']
        self.assertTrue(can_convert(checked, i64))
        self.assertTrue(can_convert(checked, i64, mode='explicit'))
        self.assertFalse(can_convert(checked, refined[i64, 'checked']))
        self.assertFalse(can_convert(i64, checked))

    def test_constant_integer_casts_decode_source_bits_once(self):
        from pythoc import i8, u8, u64, f64
        converter = TypeConverter(SimpleNamespace(module=ir.Module()))
        for source, value, target, expected in (
                (i8, -1, i64, -1), (i8, -128, i64, -128),
                (i8, -1, u64, (1 << 64) - 1), (u8, -1, i64, 255),
                (i8, -1, f64, -1.0)):
            with self.subTest(source=source, target=target):
                ref = wrap_value(ir.Constant(source.get_llvm_type(), value),
                                 kind='value', type_hint=source)
                self.assertEqual(converter.convert(ref, target).ir_value.constant,
                                 expected)

    def test_cyclic_pyconst_identity_is_finite_and_distinct(self):
        first, second = [], []
        first.append(first)
        second.append(second)
        left = PythonType.wrap(first, is_constant=True)
        right = PythonType.wrap(second, is_constant=True)
        self.assertEqual(get_type_id(left), get_type_id(left))
        self.assertFalse(same_type(left, right))
        self.assertNotEqual(get_type_id(left), get_type_id(right))

    def test_compatibility_alias_uses_authoritative_direction(self):
        from pythoc import i24, struct
        self.assertEqual(i32.is_compatible_with(i24), can_convert(i32, i24))
        left, right = struct[i32], struct[ptr[i32]]
        left._structure_hash = right._structure_hash = 0
        self.assertFalse(left.is_compatible_with(right))
        self.assertEqual(left.is_compatible_with(right), can_convert(left, right))

    def test_pyconst_checks_interpret_signed_and_boolean_constants(self):
        from pythoc import i8, bool as pc_bool
        signed = wrap_value(ir.Constant(i8.get_llvm_type(), 232),
                            kind='value', type_hint=i8)
        boolean = wrap_value(ir.Constant(pc_bool.get_llvm_type(), 1),
                             kind='value', type_hint=pc_bool)
        self.assertTrue(can_convert(i8, PythonType.wrap(-24, is_constant=True),
                                    value=signed))
        self.assertTrue(can_convert(pc_bool, PythonType.wrap(True, is_constant=True),
                                    value=boolean))

    def test_implicit_zero_fill_cannot_forge_proofs(self):
        from pythoc import linear, struct
        from pythoc.type_descriptor import can_zero_initialize
        checked = refined[i32, 'checked']
        self.assertTrue(can_zero_initialize(array[i32, 2]))
        self.assertFalse(can_zero_initialize(array[checked, 2]))
        self.assertFalse(can_zero_initialize(struct[linear]))
        self.assertFalse(can_convert(PythonType.wrap([]), array[checked, 1]))
        self.assertFalse(can_convert(PythonType.wrap(()), struct[checked]))
        self.assertFalse(can_convert(PythonType.wrap(0), struct[checked]))
        self.assertTrue(can_convert(PythonType.wrap([]), array[i32, 1]))

    def test_integer_index_and_varargs_categories_use_type_metadata(self):
        from pythoc import i8, u1, u8, u16, f32, f64, bool as pc_bool
        from pythoc.type_descriptor import is_integer_operand, default_argument_promotion
        for source in (i8, u1, u8, const[u16], pc_bool, refined[i32, 'checked']):
            self.assertTrue(is_integer_operand(source))
        self.assertFalse(is_integer_operand(f64))
        self.assertFalse(is_integer_operand(PythonType.wrap(1.5)))
        for source in (i8, u1, u8, const[u16], pc_bool):
            self.assertIs(default_argument_promotion(source), i32)
        self.assertIs(default_argument_promotion(const[f32]), f64)
        self.assertIsNone(default_argument_promotion(i64))

    def test_structural_heads_use_the_same_exact_values_as_ids(self):
        malformed = type('BooleanExtent', (array,),
                         dict(element_type=i32, dimensions=(True,)))
        regular = array[i32, 1]
        self.assertNotEqual(get_type_id(malformed), get_type_id(regular))
        self.assertFalse(same_type(malformed, regular))

    def test_array_and_simd_reject_boolean_extents_and_proof_lanes(self):
        from pythoc import simd
        with self.assertRaises(TypeError):
            array[i32, True]
        with self.assertRaises(TypeError):
            simd[i32, True]
        with self.assertRaises(TypeError):
            simd[refined[i32, 'checked'], 4]

    def test_array_decay_preserves_pointee_storage_qualifiers(self):
        from pythoc.type_descriptor import array_decay_type
        source = const[array[i32, 3]]
        self.assertTrue(same_type(array_decay_type(source), ptr[const[i32]]))
        self.assertFalse(can_convert(source, ptr[i32]))
        self.assertTrue(can_convert(source, ptr[const[i32]]))
        self.assertTrue(same_type(array_decay_type(volatile[array[i32, 2, 3]]),
                                  ptr[volatile[array[i32, 3]]]))

    def test_atomic_admission_preserves_storage_and_refinement_rules(self):
        from pythoc import i24, f64
        from pythoc.type_descriptor import atomic_value_is_supported
        self.assertTrue(atomic_value_is_supported(i32, 'add'))
        self.assertTrue(atomic_value_is_supported(const[i32], 'load'))
        self.assertFalse(atomic_value_is_supported(const[i32], 'store'))
        self.assertFalse(atomic_value_is_supported(i24, 'load'))
        self.assertFalse(atomic_value_is_supported(f64, 'load'))
        checked = refined[i32, 'checked']
        self.assertTrue(atomic_value_is_supported(checked, 'store'))
        self.assertFalse(atomic_value_is_supported(checked, 'add'))
        self.assertTrue(atomic_value_is_supported(ptr[i32], 'xchg'))
        self.assertFalse(atomic_value_is_supported(ptr[i32], 'add'))

    def test_conversion_materializes_a_scalar_place_before_same_type_fast_path(self):
        module = ir.Module()
        function = ir.Function(module, ir.FunctionType(ir.VoidType(), []), name='place')
        builder = ir.IRBuilder(function.append_basic_block('entry'))
        address = builder.alloca(i32.get_llvm_type())
        value = wrap_value(address, kind='address', type_hint=i32, address=address)
        result = TypeConverter(SimpleNamespace(module=module, builder=builder)).convert(value, i32)
        self.assertEqual(result.ir_value.type, i32.get_llvm_type())
        self.assertEqual(result.ir_value.opname, 'load')

    def test_ordinary_cast_cannot_erase_linear_storage_or_carriers(self):
        from pythoc import linear, struct
        from pythoc.type_descriptor import contains_linear_storage
        token = wrap_value(ir.Constant(linear.get_llvm_type(), []),
                           kind='value', type_hint=linear)
        carrier = PythonType.wrap([token])
        self.assertTrue(contains_linear_storage(struct[linear]))
        self.assertFalse(contains_linear_storage(ptr[linear]))
        self.assertFalse(can_convert(linear, void, mode='explicit'))
        self.assertFalse(can_convert(carrier, void, mode='explicit'))
        for mode in ('implicit', 'explicit'):
            self.assertFalse(can_convert(linear, PythonType.wrap([], is_constant=True),
                                         mode=mode, value=token))
        self.assertFalse(can_convert(struct[linear], struct[i32]))
        cycle = [token]
        cycle.append(cycle)
        self.assertTrue(contains_linear_storage(PythonType.wrap(cycle)))

    def test_nested_specialization_keys_share_type_identity(self):
        from pythoc.type_descriptor import specialization_key
        from pythoc.meta.normalize import normalize_factory_key
        first, second = {'value': [ptr[i32]]}, {'value': [ptr[i32]]}
        self.assertEqual(specialization_key(first), specialization_key(second))
        self.assertEqual(normalize_suffix(first).identity, specialization_key(first))
        self.assertEqual(normalize_factory_key(first), normalize_factory_key(second))
        self.assertNotEqual(specialization_key(first), specialization_key({'value': [ptr[i64]]}))

    def test_cyclic_specialization_identity_is_finite_but_factory_is_rejected(self):
        from pythoc.type_descriptor import specialization_key
        from pythoc.meta.normalize import normalize_factory_key
        cycle = []
        cycle.append(cycle)
        self.assertEqual(normalize_suffix(cycle).identity, specialization_key(cycle))
        with tempfile.TemporaryDirectory() as directory:
            obj, so = str(Path(directory) / 'cycle.o'), str(Path(directory) / 'cycle.so')
            for _ in range(2):
                verify_artifact_identity(obj, so, scope=None, suffix=cycle, effect_suffix=None)
        with self.assertRaisesRegex(TypeError, 'cyclic factory input'):
            normalize_factory_key(cycle)

    def test_selection_preserves_refinement_while_arithmetic_forgets_it(self):
        from pythoc.type_descriptor import common_numeric_type
        checked = refined[i32, 'checked']
        self.assertTrue(same_type(TypeConverter.infer_unified_pc_type(checked, checked), checked))
        self.assertTrue(same_type(TypeConverter.infer_unified_pc_type(const[checked], checked), checked))
        self.assertIs(common_numeric_type(checked, checked), i32)

    def test_selection_preserves_weaker_existing_proof_in_both_orders(self):
        from pythoc.type_descriptor import common_numeric_type, common_value_type
        weaker = refined[i32, 'checked']
        stronger = refined[i32, 'checked', 'extra']
        equivalent = refined[i32, 'extra', 'checked']
        unrelated = refined[i32, 'unrelated']
        for left, right in ((weaker, stronger), (stronger, weaker)):
            with self.subTest(left=left, right=right):
                self.assertTrue(same_type(common_value_type(left, right), weaker))
                self.assertTrue(same_type(common_numeric_type(left, right), i32))
        self.assertTrue(same_type(common_value_type(stronger, equivalent),
                                  common_value_type(equivalent, stronger)))
        self.assertTrue(same_type(common_value_type(weaker, unrelated), i32))

    def test_anonymous_unions_do_not_inherit_record_field_conversions(self):
        from pythoc import struct, u32, union
        source = union['value': i32]
        identical = union['value': i32]
        different = union['value': u32]
        record = struct['value': i32]
        self.assertTrue(can_convert(source, identical))
        for first, second in ((source, different), (different, source),
                              (source, record), (record, source)):
            for mode in ('implicit', 'explicit'):
                with self.subTest(source=first, target=second, mode=mode):
                    self.assertFalse(can_convert(first, second, mode=mode))

    def test_constructor_labels_dimensions_and_type_arguments_are_checked(self):
        from pythoc import struct, union, simd
        invalid = (lambda: struct['same': i32, 'same': i64],
                   lambda: union['same': i32, 'same': i64],
                   lambda: func['same': i32, 'same': i64, i32],
                   lambda: func[i32, 'result': i32],
                   lambda: ptr[i32, -1], lambda: ptr[i32, True],
                   lambda: array['element': i32, 3], lambda: array[int, 3],
                   lambda: ptr[int], lambda: const[int], lambda: const[i32, i64],
                   lambda: const['named': i32], lambda: refined[int, 'tag'],
                   lambda: refined[void, 'tag'], lambda: refined['named': i32, 'tag'],
                   lambda: create_struct_type([void]), lambda: func[void, i32],
                   lambda: simd['element': i32, 4])
        for constructor in invalid:
            with self.assertRaises(TypeError):
                constructor()
        self.assertTrue(same_type(ptr[i32, None, 3], ptr[array[i32, 3]]))
        self.assertTrue(same_type(func[None], func[void]))

    def test_enum_tags_are_representable_and_payload_discriminants_unique(self):
        from pythoc import i8, f64
        with self.assertRaisesRegex(TypeError, 'not representable'):
            _create_enum_type([('Overflow', None, 128)], i8)
        with self.assertRaisesRegex(TypeError, 'distinct tag values'):
            _create_enum_type([('First', i32, 0), ('Second', f64, 0)], i32)
        with self.assertRaisesRegex(TypeError, 'enum tag type must be an integer'):
            _create_enum_type([('Value', None, 0)], f64)
        aliases = _create_enum_type([('First', None, 0), ('Alias', None, 0)], i32)
        self.assertEqual(aliases.First, aliases.Alias)
        self.assertTrue(same_type(_create_enum_type([('Empty', None, 0)], i32),
                                  _create_enum_type([('Empty', void, 0)], i32)))

    def test_provisional_record_method_uses_authoritative_identity(self):
        import importlib
        from pythoc import compile as pc_compile
        decorators = importlib.import_module('pythoc.decorators.structs')
        original = decorators.create_struct_type
        observed = []
        def capture(*args, **kwargs):
            proxy = kwargs.get('python_class')
            if proxy is not None:
                observed.append(proxy.get_type_id(set()))
                self.assertEqual(observed[-1], get_type_id(proxy))
            return original(*args, **kwargs)
        with patch.object(decorators, 'create_struct_type', side_effect=capture):
            for field in (i32, i64):
                proxy = type('ProvisionalIdentity', (),
                             {'__annotations__': {'value': field}, '__module__': __name__})
                pc_compile(proxy)
        self.assertEqual(len(observed), 2)
        self.assertNotEqual(*observed)

    def test_cimport_enum_underlying_type_drives_emission_and_layout(self):
        from pythoc import u64
        from pythoc.cimport_ir import CDeclIR, CTypeIR, CModuleIR, CEnumValueIR
        from pythoc.cimport_emitter import emit_pythoc_module, _LayoutModel
        declaration = CDeclIR('enum', 'Wide', type=CTypeIR('primitive', 'u64'),
                              values=[CEnumValueIR('Large', 1 << 40)])
        self.assertIn('@enum(u64)', emit_pythoc_module(CModuleIR([declaration]), 'c'))
        self.assertEqual(_LayoutModel([declaration]).type_layout(CTypeIR('enum', 'Wide')),
                         (u64.get_size_bytes(), u64.get_alignment()))

    def test_storage_qualifiers_survive_refinement_in_both_nesting_orders(self):
        from pythoc import struct, static
        from pythoc.ir_helpers import is_const, is_volatile, is_static
        from pythoc.type_descriptor import array_decay_type, contains_const_storage
        inside = refined[const[i32], 'inside_const']
        self.assertTrue(is_const(inside))
        from pythoc import mark_type_defined
        from pythoc.builtin_entities.refined import _create_refined_type_class
        mark_type_defined('ReadonlyStorageAlias', const[i32])
        quoted = _create_refined_type_class('ReadonlyStorageAlias', [], ['quoted_storage'])
        self.assertTrue(is_const(quoted))
        mixed = volatile[refined[static[const[i32]], 'mixed_storage']]
        self.assertTrue(is_const(mixed))
        self.assertTrue(is_volatile(mixed))
        self.assertTrue(is_static(mixed))
        record = struct['fixed': const[i32], 'mutable': i32]
        self.assertFalse(is_const(record))
        self.assertTrue(contains_const_storage(record))
        self.assertFalse(contains_const_storage(ptr[record]))
        array_proof = refined[const[array[i32, 2]], 'array_checked']
        self.assertTrue(same_type(array_decay_type(array_proof), ptr[const[i32]]))

    def test_zero_initialization_resolves_quoted_storage_types(self):
        from pythoc import mark_type_defined
        from pythoc.type_descriptor import can_zero_initialize
        ordinary = create_struct_type([i32], ['value'])
        proof = create_struct_type([refined[i32, 'quoted_checked']], ['value'])
        mark_type_defined('ZeroOrdinary', ordinary)
        mark_type_defined('ZeroProof', proof)
        self.assertTrue(can_zero_initialize('ZeroOrdinary'))
        self.assertFalse(can_zero_initialize('ZeroProof'))
        self.assertFalse(can_zero_initialize('ZeroUnknown'))

    def test_storage_checks_follow_values_and_stop_at_pointer_edges(self):
        from pythoc import linear
        from pythoc.type_descriptor import (
            contains_linear_storage, contains_refinement_storage,
            validate_static_storage,
        )
        checked = refined[i32, 'checked']
        record = create_struct_type([checked], ['value'])
        self.assertTrue(contains_refinement_storage(const[record]))
        self.assertFalse(contains_refinement_storage(ptr[record]))
        self.assertFalse(contains_refinement_storage(func[checked, i32]))
        self.assertTrue(contains_linear_storage(create_struct_type([linear], ['token'])))
        self.assertFalse(contains_linear_storage(ptr[linear]))
        with self.assertRaisesRegex(TypeError, 'zero-initialize'):
            validate_static_storage(record, zero_initialized=True)
        with self.assertRaisesRegex(TypeError, 'linear'):
            validate_static_storage(linear)
        validate_static_storage(ptr[record], zero_initialized=True)

    def test_standard_predicate_identity_uses_family_and_current_type_arguments(self):
        from pythoc.std.refinement import nonnull, positive, nonnegative, in_range
        first_pred, first = nonnull(ptr[i32])
        second_pred, second = nonnull(ptr[i32])
        self.assertTrue(same_type(first, second))
        self.assertEqual(get_type_id(first), get_type_id(second))
        self.assertTrue(can_convert(first, second))
        self.assertTrue(can_convert(second, first))
        self.assertFalse(same_type(first, nonnull(ptr[i64])[1]))
        self.assertFalse(same_type(positive(i32)[1], nonnegative(i32)[1]))
        self.assertTrue(same_type(in_range(i32, 0, 10)[1], in_range(i32, 0, 10)[1]))
        self.assertFalse(can_convert(in_range(i32, 0, 10)[1], in_range(i32, 1, 10)[1]))
        self.assertNotEqual(get_type_id(refined[i32, lambda value: value > 0]),
                            get_type_id(refined[i32, lambda value: value > 0]))
        self.assertNotEqual(get_type_id(refined), get_type_id(ptr[void]))

    def test_standard_predicate_type_edges_close_recursive_records(self):
        from pythoc.std.refinement import nonnull
        records, proofs = [], []
        for _ in range(2):
            record = create_struct_type([], ['next'],
                                        python_class=type('PredicateNode', (), {}))
            record._field_types = [None]
            proof = nonnull(ptr[record])[1]
            record._field_types = [proof]
            records.append(record)
            proofs.append(proof)
        self.assertTrue(same_type(*records))
        self.assertTrue(same_type(*proofs))
        self.assertEqual(get_type_id(proofs[0]), get_type_id(proofs[1]))
        self.assertTrue(can_convert(*proofs))

    def test_recursion_unrolling_has_one_key(self):
        node = create_struct_type([], ['next', 'value'],
                                  python_class=type('Node', (), {}))
        quoted = ptr[ptr['Node']]
        concrete = ptr[ptr[node]]
        from pythoc.forward_ref import mark_type_defined
        mark_type_defined('Node', node)
        node._field_types = [quoted, i32]
        self.assertEqual(type_key(quoted), type_key(concrete))
        self.assertEqual(get_type_id(quoted), get_type_id(concrete))

    def test_incomplete_declaration_has_provisional_identity(self):
        record = create_struct_type([], ['value'],
                                    python_class=type('Pending', (), {}))
        record._field_types = [None]
        before = get_type_id(ptr[record])
        with self.assertRaisesRegex(TypeError, 'Incomplete PythoC value type'):
            validate_type_graph(record)
        record._field_types = [i32]
        self.assertNotEqual(before, get_type_id(ptr[record]))

    def test_incomplete_same_name_declarations_remain_distinct(self):
        first = create_struct_type([], ['value'],
                                   python_class=type('Pending', (), {}))
        second = create_struct_type([], ['value'],
                                    python_class=type('Pending', (), {}))
        first._field_types = [None]
        second._field_types = [None]
        self.assertFalse(same_type(first, second))
        self.assertNotEqual(get_type_id(first), get_type_id(second))

    def test_identity_does_not_query_backend(self):
        with patch.object(i32, 'get_llvm_type',
                          side_effect=AssertionError('backend called')):
            self.assertTrue(same_type(ptr[i32], ptr[i32]))
            self.assertTrue(can_convert(ptr[i32], ptr[const[i32]]))

    def test_short_artifact_path_verifies_full_key(self):
        with tempfile.TemporaryDirectory() as directory:
            obj = str(Path(directory) / 'short.o')
            so = str(Path(directory) / 'short.so')
            Path(obj).write_bytes(b'old-object')
            verify_artifact_identity(obj, so, scope=None, suffix=ptr[i32],
                                     effect_suffix=None)
            self.assertFalse(Path(obj).exists())
            self.assertEqual(len(list(Path(directory).glob('short.o.unverified-*'))), 1)
            verify_artifact_identity(obj, so, scope=None, suffix=ptr[i32],
                                     effect_suffix=None)
            with self.assertRaisesRegex(TypeError, 'filename collision'):
                verify_artifact_identity(obj, so, scope=None, suffix=ptr[i64],
                                         effect_suffix=None)

    def test_normalized_suffix_retains_type_identity(self):
        small = normalize_suffix((ptr[i32],))
        large = normalize_suffix((ptr[i64],))
        self.assertNotEqual(small.identity, large.identity)
        self.assertIs(normalize_suffix(small), small)

    def test_same_semantic_pointer_can_need_backend_bitcast(self):
        record = create_struct_type([i32], ['value'],
                                    python_class=type('LateRecord', (), {}))
        pointer = ptr[record]
        module = ir.Module(name='type_relation_pointer')
        function = ir.Function(module, ir.FunctionType(ir.VoidType(), []),
                               name='test')
        builder = ir.IRBuilder(function.append_basic_block('entry'))
        visitor = SimpleNamespace(builder=builder, module=module)
        source = wrap_value(ir.Constant(ir.PointerType(ir.IntType(8)), None),
                            kind='value', type_hint=pointer)
        converted = TypeConverter(visitor).convert(source, pointer)
        self.assertEqual(converted.ir_value.type,
                         pointer.get_llvm_type(module.context))

    def test_value_dependent_constructors_and_array_decay(self):
        result = _create_enum_type([('Ok', i32, 0), ('Empty', None, 1)], i32)
        tagged = create_struct_type([PythonType.wrap(0), i64],
                                    ['tag', 'payload'])
        wrong_tag = create_struct_type([PythonType.wrap(9), i64],
                                       ['tag', 'payload'])
        self.assertTrue(can_convert(tagged, result))
        self.assertFalse(can_convert(wrong_tag, result))
        self.assertTrue(can_convert(PythonType.wrap((0, 4)), result))
        self.assertFalse(can_convert(PythonType.wrap((1, 4)), result))
        self.assertFalse(can_convert(PythonType.wrap((9,)), result))
        self.assertTrue(can_convert(array[i32, 3], ptr[i32]))
        self.assertTrue(can_convert(array[i32, 2, 3], ptr[array[i32, 3]]))
        self.assertFalse(can_convert(array[i32, 3], ptr[i64]))
        self.assertTrue(can_convert(PythonType.wrap([1, 2]), array[i32, 2]))
        self.assertFalse(can_convert(PythonType.wrap([1, 2, 3]), array[i32, 2]))

    def test_backend_cast_categories_do_not_authorize_type_conversion(self):
        from pythoc import f32, void
        self.assertFalse(can_convert(ptr[i32], f32, mode='explicit'))
        self.assertFalse(can_convert(f32, ptr[i32], mode='explicit'))
        self.assertFalse(can_convert(i32, ptr[i32], mode='implicit'))
        self.assertTrue(can_convert(i32, ptr[i32], mode='explicit'))
        self.assertTrue(can_convert(PythonType.wrap('hello'), ptr[void]))

    def test_string_literal_adapts_to_signed_and_unsigned_c_char(self):
        from pythoc import bool as pc_bool, i8, u8

        literal = PythonType.wrap('hello')
        for byte in (i8, u8):
            for pointee in (byte, const[byte], volatile[byte]):
                with self.subTest(pointee=pointee):
                    self.assertTrue(can_convert(literal, ptr[pointee]))
            other = u8 if byte is i8 else i8
            self.assertFalse(can_convert(ptr[byte], ptr[other]))
            self.assertFalse(can_convert(array[byte, 4], ptr[other]))
        for pointee in (pc_bool, i32, refined[u8, 'checked']):
            with self.subTest(pointee=pointee):
                self.assertFalse(can_convert(literal, ptr[pointee]))

    def test_full_id_tracks_additional_composites(self):
        result = _create_enum_type([('Ok', i32, 0), ('Err', i64, 1)], i32)
        near = _create_enum_type([('Ok', i32, 0), ('Err', i64, 2)], i32)
        types = (array[i32, 3], array[i32, 4], func[i32, i64],
                 func['x': i32, i64], result, near,
                 create_struct_type([i32, i64], ['x', 'y']),
                 create_struct_type([i32, i64], ['x', 'z']))
        for left in types:
            for right in types:
                with self.subTest(left=left, right=right):
                    self.assertEqual(get_type_id(left) == get_type_id(right),
                                     same_type(left, right))


if __name__ == '__main__':
    unittest.main()
