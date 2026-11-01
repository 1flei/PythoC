"""Tests for per-module LLVM type materialization of struct types.

A struct class compiled into two different LLVM modules must yield two
distinct IdentifiedStructType objects, each owned by its module's context,
with identical field layout.  Self-referential structs must set their body
independently in every module.
"""

import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from llvmlite import ir

from pythoc.builtin_entities import i32, i64, u32, f64, ptr
from pythoc.builtin_entities.struct import create_struct_type
from pythoc.type_id import get_type_id
from pythoc.forward_ref import mark_type_defined
from pythoc.session import CompileSession
from pythoc.builder.llvm_c_builder import LLVMCBuilder
from pythoc.type_graph import canonical_record_type
from pythoc.builder.llvm_simple_builder import LLVMBuilder
from pythoc.literal_protocol import assemble_aggregate_value


class Point:
    pass


def _make_point_type():
    return create_struct_type([i32, f64], ['x', 'y'], python_class=Point)


def _make_module(name):
    """Module with its own context (one context per module)."""
    return ir.Module(name, context=ir.Context())


class TestStructTypePerModule(unittest.TestCase):

    def test_identified_type_is_per_module(self):
        point_type = _make_point_type()
        mod1 = _make_module('m1')
        mod2 = _make_module('m2')

        t1 = point_type.get_llvm_type(mod1.context)
        t2 = point_type.get_llvm_type(mod2.context)

        self.assertIsNot(t1, t2)
        self.assertIs(t1.context, mod1.context)
        self.assertIs(t2.context, mod2.context)
        # Same field layout in both modules.
        self.assertEqual([str(e) for e in t1.elements],
                         [str(e) for e in t2.elements])

    def test_same_module_lookup_is_stable(self):
        point_type = _make_point_type()
        mod = _make_module('m')

        t1 = point_type.get_llvm_type(mod.context)
        t2 = point_type.get_llvm_type(mod.context)

        self.assertIs(t1, t2)

    def test_self_referential_struct_per_module(self):
        class Node:
            pass

        node_type = create_struct_type([], ['next'], python_class=Node)
        node_type._field_types = [ptr[node_type]]

        mod1 = _make_module('n1')
        mod2 = _make_module('n2')
        t1 = node_type.get_llvm_type(mod1.context)
        t2 = node_type.get_llvm_type(mod2.context)

        self.assertIsNot(t1, t2)
        # Body must be set in both modules (no cross-module flag leaks).
        self.assertIsNotNone(t1.elements)
        self.assertIsNotNone(t2.elements)
        # Pointer storage does not encode a semantic pointee or recurse.
        self.assertEqual(t1.elements[0], ir.PointerType(ir.IntType(8)))
        self.assertEqual(t2.elements[0], ir.PointerType(ir.IntType(8)))
        self.assertIs(ptr[node_type].get_llvm_type(mod1.context).pointee, t1)
        self.assertIs(ptr[node_type].get_llvm_type(mod2.context).pointee, t2)

    def test_field_map_frozen_across_modules(self):
        point_type = _make_point_type()
        mod1 = _make_module('f1')
        mod2 = _make_module('f2')

        idx1 = point_type._get_llvm_field_index(1, mod1.context)
        idx2 = point_type._get_llvm_field_index(1, mod2.context)

        self.assertEqual(idx1, 1)
        self.assertEqual(idx2, 1)

    def test_recursive_views_of_one_identified_type_share_its_body(self):
        class RecursiveView:
            pass

        first = create_struct_type([], ['next', 'value'], python_class=RecursiveView)
        second = create_struct_type([], ['next', 'value'], python_class=RecursiveView)
        first._field_types = [ptr[second], i32]
        second._field_types = [ptr[first], i32]

        for name in ('views_one', 'views_two'):
            module = _make_module(name)
            llvm_type = first.get_llvm_type(module.context)
            self.assertIs(second.get_llvm_type(module.context), llvm_type)
            self.assertEqual(llvm_type.elements[0], ir.PointerType(ir.IntType(8)))
            self.assertEqual(str(llvm_type.elements[1]), 'i32')

    def test_conflicting_recursive_views_fail_before_materialization(self):
        class ConflictingView:
            pass

        first = create_struct_type([], ['next', 'value'], python_class=ConflictingView)
        second = create_struct_type([], ['next', 'value'], python_class=ConflictingView)
        first._field_types = [ptr[second], i32]
        second._field_types = [ptr[first], i64]
        module = _make_module('conflicting_views')
        with self.assertRaisesRegex(TypeError, 'Conflicting PythoC record views'):
            first.get_llvm_type(module.context)
        self.assertEqual(module.context.identified_types, {})

    def test_same_name_independent_declarations_keep_their_own_bodies(self):
        module = _make_module('generic_declarations')
        small = create_struct_type(
            [i32], ['value'], python_class=type('Generic', (), {}))
        wide = create_struct_type(
            [i64], ['value'], python_class=type('Generic', (), {}))
        unsigned = create_struct_type(
            [u32], ['value'], python_class=type('Generic', (), {}))
        a, b, c = [t.get_llvm_type(module.context) for t in (small, wide, unsigned)]
        self.assertEqual(str(a.elements[0]), 'i32')
        self.assertEqual(str(b.elements[0]), 'i64')
        self.assertEqual(str(c.elements[0]), 'i32')
        self.assertIsNot(a, b)
        self.assertIs(a, c)
        self.assertNotEqual(get_type_id(small), get_type_id(unsigned))

    def test_semantic_identity_does_not_accept_backend_types(self):
        with self.assertRaises(TypeError):
            get_type_id(ir.IntType(32))
        with self.assertRaises(TypeError):
            get_type_id(ir.Context().get_identified_type('Point'))

    def test_forward_completion_does_not_rematerialize_the_owner(self):
        with CompileSession():
            context = ir.Context()
            holder = create_struct_type(
                [ptr['Pending']], ['next'], python_class=type('Holder', (), {}))
            before = holder.get_llvm_type(context)
            pending = create_struct_type(
                [i32], ['value'], python_class=type('Pending', (), {}))
            mark_type_defined('Pending', pending)
            self.assertIs(holder.get_llvm_type(context), before)
            self.assertEqual(before.elements[0], ir.PointerType(ir.IntType(8)))
            self.assertEqual(pending.get_llvm_type(context).elements, (ir.IntType(32),))
            self.assertEqual(len(context.identified_types), 2)

    def test_foreign_backend_name_does_not_define_pythoc_identity(self):
        context = ir.Context()
        foreign = context.get_identified_type('Point')
        foreign.set_body(ir.IntType(8))
        point = _make_point_type().get_llvm_type(context)
        self.assertIsNot(point, foreign)
        self.assertEqual(str(point.elements[0]), 'i32')

    def test_unowned_opaque_name_is_not_claimed(self):
        context = ir.Context()
        foreign = context.get_identified_type('Point')
        point = _make_point_type().get_llvm_type(context)
        self.assertIsNot(point, foreign)
        self.assertTrue(foreign.is_opaque)
        self.assertEqual(str(point.elements[0]), 'i32')

    def test_pointer_constants_match_their_field_storage_type(self):
        module = _make_module('constant_storage')
        global_value = ir.GlobalVariable(module, ir.IntType(32), 'value')
        holder = create_struct_type(
            [ptr[i32]], ['next'], python_class=type('ConstantHolder', (), {}))
        constant = assemble_aggregate_value(
            SimpleNamespace(module=module), holder, [global_value])
        self.assertEqual(constant.constant[0].type, constant.type.elements[0])
        self.assertEqual(constant.constant[0].type,
                         ir.PointerType(ir.IntType(8)))

    def test_storage_adaptation_cannot_truncate_an_array(self):
        module = _make_module('invalid_storage')
        function = ir.Function(module, ir.FunctionType(ir.VoidType(), []), 'f')
        builder = LLVMBuilder(ir.IRBuilder(function.append_basic_block('entry')))
        target = ir.LiteralStructType([ir.ArrayType(ir.IntType(32), 2)])
        source = ir.Constant(ir.ArrayType(ir.IntType(32), 3), [1, 2, 3])
        with self.assertRaisesRegex(TypeError, 'array length'):
            builder.insert_value(ir.Constant(target, ir.Undefined), source, 0)

    def test_late_equivalent_views_keep_compatible_existing_signatures(self):
        with CompileSession():
            pending = create_struct_type(
                [i32], ['value'], python_class=type('Pending', (), {}))
            first = create_struct_type(
                [ptr['Pending']], ['next'],
                python_class=type('Container', (), {}))
            second = create_struct_type(
                [ptr[pending]], ['next'],
                python_class=type('Container', (), {}))
            module = _make_module('late_equivalence')
            a = first.get_llvm_type(module.context)
            b = second.get_llvm_type(module.context)
            mark_type_defined('Pending', pending)
            self.assertIs(canonical_record_type(first),
                          canonical_record_type(second))
            builder = LLVMCBuilder()
            declared = builder.declare_function(
                module, 'consume', [a.as_pointer()], ir.IntType(32))
            redeclared = builder.declare_function(
                module, 'consume', [b.as_pointer()], ir.IntType(32),
                existing_func=declared.ir_function)
            self.assertIs(declared.ir_function, redeclared.ir_function)

    def test_backend_failure_can_retry_the_same_declaration(self):
        context = ir.Context()
        point_type = _make_point_type()
        with patch.object(i32, 'get_llvm_type', side_effect=RuntimeError('failed')):
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                point_type.get_llvm_type(context)
        record = point_type.get_llvm_type(context)
        self.assertEqual(str(record.elements[0]), 'i32')
        self.assertIs(point_type.get_llvm_type(context), record)

    def test_concurrent_recursive_views_in_one_context(self):
        owner = type('ConcurrentViews', (), {})
        first = create_struct_type([], ['next', 'value'], python_class=owner)
        second = create_struct_type([], ['next', 'value'], python_class=owner)
        first._field_types = [ptr[second], i32]
        second._field_types = [ptr[first], i32]
        context = ir.Context()
        results, errors = [], []

        def worker(view):
            try:
                results.append(view.get_llvm_type(context))
            except Exception as error:
                errors.append(error)

        threads = [threading.Thread(target=worker, args=(t,)) for t in (first, second)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertIs(results[0], results[1])
        self.assertEqual(results[0].elements[0], ir.PointerType(ir.IntType(8)))

    def test_default_modules_share_llvmlite_global_context(self):
        # Status quo: ir.Module() without an explicit context uses llvmlite's
        # global context, so all LLVMCompiler modules share identified types.
        # get_llvm_type() must preserve that behavior under this regime.
        point_type = _make_point_type()
        mod1 = ir.Module('g1')
        mod2 = ir.Module('g2')
        if mod1.context is not mod2.context:
            self.skipTest('llvmlite no longer uses a shared global context')

        t1 = point_type.get_llvm_type(mod1.context)
        t2 = point_type.get_llvm_type(mod2.context)
        self.assertIs(t1, t2)
        self.assertIsNotNone(t1.elements)

    def test_concurrent_materialization_into_two_modules(self):
        point_type = _make_point_type()
        mods = [_make_module('c1'), _make_module('c2')]
        results = [None, None]
        errors = []

        def worker(i):
            try:
                results[i] = point_type.get_llvm_type(mods[i].context)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertIsNot(results[0], results[1])
        self.assertIs(results[0].context, mods[0].context)
        self.assertIs(results[1].context, mods[1].context)
        self.assertIsNotNone(results[0].elements)
        self.assertIsNotNone(results[1].elements)


if __name__ == '__main__':
    unittest.main()
