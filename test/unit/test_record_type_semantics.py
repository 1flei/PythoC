"""Record identity and validity are decided without backend materialization."""

import unittest
from unittest.mock import patch

from pythoc import array, compile, const, func, i32, i64, ptr, refined, u32, union
from pythoc.builtin_entities.struct import create_struct_type
from pythoc.forward_ref import mark_type_defined
from pythoc.session import CompileSession
from pythoc.target_layout import make_aggregate_layout_config
from pythoc.type_graph import (
    canonical_record_type, prepare_type, record_definition, validate_type_graph,
)
from pythoc.type_id import get_type_id
from pythoc.type_resolver import TypeReference, TypeResolver


def recursive_views(value_type=i32, other_type=i32):
    owner = type('RecordView', (), {})
    first = create_struct_type([], ['next', 'value'], python_class=owner)
    second = create_struct_type([], ['next', 'value'], python_class=owner)
    first._field_types = [ptr[second], value_type]
    second._field_types = [ptr[first], other_type]
    return first, second


class RecordTypeSemanticsTest(unittest.TestCase):
    def setUp(self):
        self.session = CompileSession()
        self.session.__enter__()
        self.addCleanup(self.session.__exit__, None, None, None)

    def test_recursive_factory_fields_bind_their_own_declaration(self):
        def make_node(value_type):
            @compile
            class FactoryNode:
                next: ptr['FactoryNode']
                value: value_type

            return FactoryNode

        small = make_node(i32)
        wide = make_node(i64)
        self.assertIs(small._field_types[0].pointee_type, small)
        self.assertIs(wide._field_types[0].pointee_type, wide)
        self.assertNotEqual(get_type_id(small), get_type_id(wide))

    def test_recursive_views_have_one_pythoc_identity(self):
        first, second = recursive_views()
        with patch.object(
            i32, 'get_llvm_type', side_effect=AssertionError('backend called')):
            self.assertIs(canonical_record_type(first), canonical_record_type(second))
            self.assertEqual(get_type_id(first), get_type_id(second))
            self.assertEqual(get_type_id(ptr[first]), get_type_id(ptr[second]))

    def test_conflicting_views_are_rejected_in_the_type_layer(self):
        for other in (i64, u32, const[i32], refined[i32, 'tag']):
            with self.subTest(other=other):
                first, _ = recursive_views(other_type=other)
                with self.assertRaisesRegex(
                    TypeError, 'Conflicting PythoC record views'):
                    canonical_record_type(first)

    def test_field_names_are_semantic_even_with_the_same_storage(self):
        first, second = recursive_views()
        second._field_names = ['next', 'different']
        with self.assertRaisesRegex(TypeError, 'Conflicting PythoC record views'):
            get_type_id(first)

    def test_field_count_is_not_truncated_by_comparison(self):
        first, second = recursive_views()
        second._field_types.append(i32)
        with self.assertRaisesRegex(TypeError, 'Incomplete PythoC record definition'):
            validate_type_graph(first)

    def test_layout_configuration_is_part_of_the_declaration(self):
        first, second = recursive_views()
        second._layout_config = make_aggregate_layout_config(abi_alignment=64)
        with self.assertRaisesRegex(TypeError, 'Conflicting PythoC record views'):
            validate_type_graph(first)

    def test_direct_type_id_and_ffi_entry_points_validate_views(self):
        for operation in (lambda t: t.get_type_id(), lambda t: t.get_ctypes_type()):
            with self.subTest(operation=operation):
                first, _ = recursive_views(other_type=u32)
                with self.assertRaisesRegex(
                    TypeError, 'Conflicting PythoC record views'):
                    operation(first)

    def test_separately_created_compatible_views_are_checked(self):
        owner = type('SequentialView', (), {})
        first = create_struct_type([i32], ['value'], python_class=owner)
        second = create_struct_type([u32], ['value'], python_class=owner)
        get_type_id(first)
        with self.assertRaisesRegex(TypeError, 'Conflicting PythoC record views'):
            get_type_id(second)

    def test_distinct_same_name_declarations_can_have_different_types(self):
        first = create_struct_type(
            [i32], ['value'], python_class=type('Generic', (), {}))
        second = create_struct_type(
            [u32], ['value'], python_class=type('Generic', (), {}))
        self.assertIsNot(canonical_record_type(first), canonical_record_type(second))
        self.assertNotEqual(get_type_id(first), get_type_id(second))

    def test_user_type_named_void_is_not_the_void_builtin(self):
        first = create_struct_type(
            [i32], ['value'], python_class=type('void', (), {}))
        second = create_struct_type(
            [i64], ['value'], python_class=type('void', (), {}))
        a = create_struct_type(
            [ptr[first]], ['next'], python_class=type('Wrapper', (), {}))
        b = create_struct_type(
            [ptr[second]], ['next'], python_class=type('Wrapper', (), {}))
        self.assertIsNot(canonical_record_type(a), canonical_record_type(b))
        self.assertNotEqual(get_type_id(a), get_type_id(b))

    def test_function_parameter_names_are_part_of_record_identity(self):
        signatures = (
            func['x':i32, 'y':i32, i32],
            func['y':i32, 'x':i32, i32],
            func['x':i32, i32, i32],
            func[i32, 'x':i32, i32],
            func[i32, i32, i32],
        )
        records = [create_struct_type(
            [signature], ['callback'], python_class=type('Callbacks', (), {}))
            for signature in signatures]
        self.assertEqual(len({get_type_id(t) for t in records}), len(records))
        self.assertEqual(len({canonical_record_type(t) for t in records}),
                         len(records))

    def test_conflicting_function_parameter_names_in_views_are_rejected(self):
        owner = type('CallbackViews', (), {})
        first = create_struct_type(
            [func['x':i32, i32]], ['callback'], python_class=owner)
        second = create_struct_type(
            [func['y':i32, i32]], ['callback'], python_class=owner)
        validate_type_graph(first)
        with self.assertRaisesRegex(TypeError, 'Conflicting PythoC record views'):
            validate_type_graph(second)

    def test_matching_named_and_unnamed_function_types_stay_canonical(self):
        for signature in (lambda: func['x':i32, i32], lambda: func[i32, i32]):
            with self.subTest(signature=signature):
                first = create_struct_type(
                    [signature()], ['callback'],
                    python_class=type('MatchingCallbacks', (), {}))
                second = create_struct_type(
                    [signature()], ['callback'],
                    python_class=type('MatchingCallbacks', (), {}))
                self.assertIs(canonical_record_type(first),
                              canonical_record_type(second))
                self.assertEqual(get_type_id(first), get_type_id(second))

    def test_omitted_parameter_names_match_explicit_unnamed_slots(self):
        for names, signature in (
            ([], func[i32, i32, i32]),
            ([None, None], func[i32, i32, i32]),
            (['x'], func['x':i32, i32, i32]),
        ):
            with self.subTest(names=names):
                equivalent = type('NormalizedFunction', (func,), {
                    '_pc_specialized': True,
                    'param_types': (i32, i32),
                    'param_names': names,
                    'return_type': i32,
                })
                first = create_struct_type(
                    [signature], ['callback'],
                    python_class=type('NormalizedCallbacks', (), {}))
                second = create_struct_type(
                    [equivalent], ['callback'],
                    python_class=type('NormalizedCallbacks', (), {}))
                self.assertEqual(get_type_id(signature), get_type_id(equivalent))
                self.assertIs(canonical_record_type(first),
                              canonical_record_type(second))

    def test_equivalent_recursive_factories_share_identity(self):
        first, _ = recursive_views()
        second, _ = recursive_views()
        self.assertIs(canonical_record_type(first), canonical_record_type(second))
        self.assertEqual(get_type_id(first), get_type_id(second))

    def test_mutual_pointer_recursion_is_valid(self):
        first = create_struct_type([], ['next'], python_class=type('First', (), {}))
        second = create_struct_type([], ['next'], python_class=type('Second', (), {}))
        first._field_types = [ptr[second]]
        second._field_types = [ptr[first]]
        validate_type_graph(first)

    def test_by_value_recursion_is_rejected_before_layout(self):
        for wrap in (lambda t: t, lambda t: array[t, 1], lambda t: union[t, i32]):
            with self.subTest(wrap=wrap):
                record = create_struct_type(
                    [], ['value'], python_class=type('Invalid', (), {}))
                record._field_types = [wrap(record)]
                with self.assertRaisesRegex(TypeError, 'Illegal by-value recursion'):
                    record.get_layout()

    def test_function_pointer_does_not_store_its_argument(self):
        record = create_struct_type(
            [], ['callback'], python_class=type('Callback', (), {}))
        record._field_types = [func[record, i32]]
        validate_type_graph(record)

    def test_late_resolution_rechecks_new_views(self):
        first, second = recursive_views(other_type=i64)
        holder = create_struct_type(
            [ptr['Later']], ['next'], python_class=type('Holder', (), {}))
        validate_type_graph(holder)
        mark_type_defined('Later', first)
        with self.assertRaisesRegex(TypeError, 'Conflicting PythoC record views'):
            validate_type_graph(holder)

    def test_completion_preserves_canonical_declaration_identity(self):
        holder = create_struct_type(
            [ptr['Pending']], ['next'], python_class=type('Holder', (), {}))
        before = record_definition(holder)
        mark_type_defined('Pending', create_struct_type([i32], ['value']))
        self.assertIs(record_definition(holder), before)

    def test_completion_rechecks_previously_distinct_declarations(self):
        pending = create_struct_type(
            [i32], ['value'], python_class=type('Pending', (), {}))
        quoted = create_struct_type(
            [ptr['Pending']], ['next'], python_class=type('Holder', (), {}))
        concrete = create_struct_type(
            [ptr[pending]], ['next'], python_class=type('Holder', (), {}))
        self.assertIsNot(canonical_record_type(quoted),
                         canonical_record_type(concrete))
        mark_type_defined('Pending', pending)
        self.assertIs(canonical_record_type(quoted),
                      canonical_record_type(concrete))
        self.assertEqual(get_type_id(quoted), get_type_id(concrete))

    def test_completion_canonicalizes_before_either_query_order(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse), CompileSession():
                target = create_struct_type([i32], ['value'],
                         python_class=type('PendingTarget', (), {}))
                first = create_struct_type([ptr['PendingTarget']], ['next'],
                        python_class=type('CompletedView', (), {}))
                second = create_struct_type([ptr[target]], ['next'],
                         python_class=type('CompletedView', (), {}))
                canonical_record_type(first)
                canonical_record_type(second)
                mark_type_defined('PendingTarget', target)
                left, right = (second, first) if reverse else (first, second)
                self.assertIs(canonical_record_type(left), canonical_record_type(right))

    def test_prepared_expression_refreshes_its_index_dependents(self):
        from pythoc.type_resolver import TypeReference, TypeResolver

        class Constructor:
            def __class_getitem__(cls, value):
                return i64

        reference = TypeReference('Constructor[i32]', {
            'Constructor': Constructor, 'i32': i32})
        first = create_struct_type([ptr[reference]], ['next'],
                python_class=type('ExpressionView', (), {}))
        second = create_struct_type([ptr[i64]], ['next'],
                 python_class=type('ExpressionView', (), {}))
        canonical_record_type(first)
        canonical_record_type(second)
        TypeResolver.resolve_type(reference, freeze=True)
        self.assertIs(canonical_record_type(second), canonical_record_type(first))

    def test_union_decorator_and_its_type_are_the_same_field_type(self):
        wrapped = union(type('Payload', (), {
            '__annotations__': {'x': i32}, '__module__': __name__}))
        owner = type('Container', (), {})
        first = create_struct_type([wrapped], ['data'], python_class=owner)
        second = create_struct_type([wrapped._union_type], ['data'], python_class=owner)
        self.assertIs(record_definition(first), record_definition(second))

    def test_semantic_failure_does_not_poison_a_corrected_definition(self):
        first, second = recursive_views(other_type=i64)
        with self.assertRaises(TypeError):
            validate_type_graph(first)
        second._field_types = [ptr[first], i32]
        self.assertEqual(get_type_id(first), get_type_id(second))

    def test_declaration_checks_are_session_local(self):
        owner = type('SessionRecord', (), {})
        first = create_struct_type([i32], ['value'], python_class=owner)
        second = create_struct_type([u32], ['value'], python_class=owner)
        validate_type_graph(first)
        with CompileSession():
            validate_type_graph(second)
        with self.assertRaisesRegex(TypeError, 'Conflicting PythoC record views'):
            validate_type_graph(second)

    def test_unchanged_unresolved_pointer_reuses_record_validation(self):
        reference = TypeReference('PendingCacheTarget', {})
        record = create_struct_type(
            [ptr[reference], i32], ['next', 'value'],
            python_class=type('CacheHolder', (), {}))
        graph = self.session.type_graph
        with patch.object(graph, '_collect', wraps=graph._collect) as collect:
            prepare_type(record)
            prepare_type(record)
        self.assertEqual(collect.call_count, 1)

    def test_anonymous_record_reuses_preparation_and_validation(self):
        reference = TypeReference('PendingAnonymousCacheTarget', {})
        record = create_struct_type([ptr[reference], i32], ['next', 'value'])
        graph = self.session.type_graph
        with patch.object(graph, '_collect', wraps=graph._collect) as collect:
            prepare_type(record)
            validate_type_graph(record)
            prepare_type(record)
            validate_type_graph(record)
        self.assertEqual(collect.call_count, 1)

    def test_new_staged_binding_still_checks_reachable_views(self):
        for holder_owner in (type('StagedCacheHolder', (), {}), None):
            with self.subTest(named=holder_owner is not None):
                owner = type('StagedCacheTarget', (), {})
                first = create_struct_type([i32], ['value'], python_class=owner)
                conflict = create_struct_type([i64], ['value'], python_class=owner)
                namespace = {}
                reference = TypeReference('StagedCacheAlias', namespace)
                holder = create_struct_type(
                    [ptr[first], ptr[reference]], ['first', 'pending'],
                    python_class=holder_owner)
                prepare_type(holder)
                namespace['StagedCacheAlias'] = conflict
                with self.assertRaisesRegex(TypeError, 'Conflicting PythoC record views'):
                    with self.session.type_graph.binding_transaction():
                        TypeResolver.resolve_type(reference, freeze=True)
                        prepare_type(holder)
                namespace['StagedCacheAlias'] = first
                prepare_type(holder)

    def test_named_materialization_prepares_the_declaration_once(self):
        from llvmlite import ir
        from pythoc import type_graph

        owner = type('MaterializedOnce', (), {})
        record = create_struct_type(
            [ptr['UnfinishedMaterializationTarget'], i32],
            ['next', 'value'], python_class=owner)
        context = ir.Context()
        with patch.object(type_graph, 'prepare_type',
                          wraps=type_graph.prepare_type) as prepare:
            first = record.get_llvm_type(context)
            second = record.get_llvm_type(context)
        self.assertIs(first, second)
        calls = [call for call in prepare.call_args_list if call.args[0] is record]
        self.assertEqual(len(calls), 2)

    def test_staged_validation_does_not_rescan_all_pending_bindings(self):
        for size in (8, 16, 32):
            with self.subTest(size=size), CompileSession() as session:
                references = [TypeReference(f'PendingTarget{index}', {})
                              for index in range(size)]
                records = [create_struct_type(
                    [ptr[reference], i32], ['next', 'value'],
                    python_class=type(f'PendingHolder{index}', (), {}))
                    for index, reference in enumerate(references)]
                for record in records:
                    prepare_type(record)
                graph = session.type_graph
                with graph.binding_transaction():
                    for reference in references:
                        TypeResolver.resolve_type(reference, required=False)
                    with patch.object(graph.type_bindings, 'get',
                                      wraps=graph.type_bindings.get) as lookup:
                        for record in records:
                            validate_type_graph(record)
                    self.assertLessEqual(lookup.call_count, 4 * size)

    def test_rolled_back_bindings_can_be_retried_with_another_type(self):
        namespace = {}
        reference = TypeReference('RollbackTarget', namespace)
        record = create_struct_type([ptr[reference]], ['next'])
        prepare_type(record)
        graph = self.session.type_graph
        namespace['RollbackTarget'] = i32
        with self.assertRaisesRegex(RuntimeError, 'abort preparation'):
            with graph.binding_transaction():
                TypeResolver.resolve_type(reference, freeze=True)
                prepare_type(record)
                self.assertIs(graph.get_binding(reference).value, i32)
                raise RuntimeError('abort preparation')
        self.assertIsNone(graph.get_binding(reference).value)
        namespace['RollbackTarget'] = i64
        prepare_type(record)
        self.assertIs(TypeResolver.resolve_type(reference), i64)

    def test_binding_completed_during_validation_requires_revalidation(self):
        owner = type('LateTarget', (), {})
        first = create_struct_type([i32], ['value'], python_class=owner)
        conflict = create_struct_type([i64], ['value'], python_class=owner)
        namespace = {}
        reference = TypeReference('LateTargetAlias', namespace)
        holder = create_struct_type(
            [ptr[first], ptr[reference]], ['first', 'late'])
        graph = self.session.type_graph
        collect = graph._collect

        def collect_then_bind(root):
            result = collect(root)
            namespace['LateTargetAlias'] = conflict
            TypeResolver.resolve_type(reference)
            return result

        with patch.object(graph, '_collect', side_effect=collect_then_bind):
            validate_type_graph(holder)
        with self.assertRaisesRegex(TypeError, 'Conflicting PythoC record views'):
            validate_type_graph(holder)


if __name__ == '__main__':
    unittest.main()
