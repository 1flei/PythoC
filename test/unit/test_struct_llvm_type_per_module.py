"""Tests for per-module LLVM type materialization of struct types.

A struct class compiled into two different LLVM modules must yield two
distinct IdentifiedStructType objects, each owned by its module's context,
with identical field layout.  Self-referential structs must set their body
independently in every module.
"""

import threading
import unittest

from llvmlite import ir

from pythoc.builtin_entities import i32, f64, ptr
from pythoc.builtin_entities.struct import create_struct_type


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
        # The single field is a pointer to the module-local identified type.
        self.assertIs(t1.elements[0].pointee, t1)
        self.assertIs(t2.elements[0].pointee, t2)

    def test_field_map_frozen_across_modules(self):
        point_type = _make_point_type()
        mod1 = _make_module('f1')
        mod2 = _make_module('f2')

        idx1 = point_type._get_llvm_field_index(1, mod1.context)
        idx2 = point_type._get_llvm_field_index(1, mod2.context)

        self.assertEqual(idx1, 1)
        self.assertEqual(idx2, 1)

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
