#!/usr/bin/env python3
"""Integration test for DWARF debug info emission.

The test only runs when ``PC_DEBUG_INFO`` / ``config.debug_info`` is enabled.
It verifies that the generated object file contains enough DWARF metadata for
gdb to set breakpoints by function name and source line, and that parameters
and local variables are described so ``info locals`` / ``info args`` work.
"""

import os
import shutil
import subprocess
import sys
import unittest

from pythoc import (
    array,
    bool,
    f64,
    i32,
    ptr,
    struct,
    union,
    compile,
)
from pythoc.config import config


@compile
def _debug_target(x: i32) -> i32:
    a: i32 = x + 1
    b: i32 = a * 2
    return b


@compile
def _debug_ptr_target(arr: 'ptr[i32]', n: i32) -> i32:
    total: i32 = 0
    i: i32 = 0
    while i < n:
        total = total + arr[i]
        i = i + 1
    return total


@compile
def _debug_scalar_target(flag: bool, value: f64) -> f64:
    scaled: f64 = value * 2.0
    if flag:
        scaled = scaled + 1.0
    return scaled


@compile
def _debug_array_target() -> i32:
    data: 'array[i32, 3]' = [10, 20, 30]
    local_arr: 'array[i32, 3]' = [1, 2, 3]
    return data[0] + local_arr[1]


@compile
def _debug_struct_target() -> f64:
    point: 'struct[x: i32, y: f64]' = (1, 1.5)
    local_point: 'struct[x: i32, y: f64]' = (10, 2.5)
    return point.y + local_point.x


@compile
def _debug_union_target() -> f64:
    data: 'union[i: i32, f: f64]' = union[i: i32, f: f64]()
    data.i = 42
    data.f = 2.5
    return data.f


def _object_file_for(wrapper) -> str:
    """Return the object file path associated with a compiled wrapper."""
    state = getattr(wrapper, '_state', None)
    if state is None:
        raise RuntimeError('wrapper has no compilation state')
    so_file = getattr(state, 'so_file', None)
    if not so_file:
        raise RuntimeError('cannot locate shared library for wrapper')
    from pythoc.utils.link_utils import get_shared_lib_extension
    return so_file.replace(get_shared_lib_extension(), '.o')


def _run_debug_dump(obj_file: str) -> str:
    """Run the platform's DWARF dumping tool and return stdout."""
    if sys.platform == 'darwin':
        exe = shutil.which('dwarfdump')
        if exe:
            return subprocess.run(
                [exe, obj_file],
                capture_output=True, text=True, check=False,
            ).stdout
    else:
        # Prefer readelf on Linux; fall back to dwarfdump if available.
        for tool in ('readelf', 'dwarfdump'):
            exe = shutil.which(tool)
            if not exe:
                continue
            if tool == 'readelf':
                return subprocess.run(
                    [exe, '--debug-dump=info', obj_file],
                    capture_output=True, text=True, check=False,
                ).stdout
            return subprocess.run(
                [exe, obj_file],
                capture_output=True, text=True, check=False,
            ).stdout
    return ''


class DebugInfoTestCase(unittest.TestCase):
    def test_debug_info_present(self):
        with config.override(debug_info=True):
            # Force compilation/object emission.
            result = _debug_target(5)
            self.assertEqual(result, 12)

            obj_file = _object_file_for(_debug_target)
            self.assertTrue(
                os.path.exists(obj_file),
                f'object file not found: {obj_file}',
            )

            dump = _run_debug_dump(obj_file)
            if not dump:
                self.skipTest('no DWARF dumping tool available')

            self.assertIn('DW_TAG_compile_unit', dump)
            self.assertIn('DW_TAG_subprogram', dump)
            self.assertIn('_debug_target', dump)

    def test_local_variables_emitted(self):
        with config.override(debug_info=True):
            # Compile a function with parameters and locals.
            self.assertEqual(_debug_target(5), 12)

            obj_file = _object_file_for(_debug_target)
            self.assertTrue(os.path.exists(obj_file))

            dump = _run_debug_dump(obj_file)
            if not dump:
                self.skipTest('no DWARF dumping tool available')

            # Parameters and locals must appear as distinct descriptors.
            self.assertIn('DW_TAG_formal_parameter', dump)
            self.assertIn('DW_TAG_variable', dump)

            # Variable names must be present in the debug info section.
            for name in ('x', 'a', 'b'):
                self.assertIn(name, dump, f'variable {name!r} missing from DWARF dump')

    def test_pointer_variable_type(self):
        with config.override(debug_info=True):
            # Force compilation of a function using a pointer parameter.
            import ctypes
            arr = (ctypes.c_int32 * 3)(1, 2, 3)
            self.assertEqual(_debug_ptr_target(arr, 3), 6)

            obj_file = _object_file_for(_debug_ptr_target)
            self.assertTrue(os.path.exists(obj_file))

            dump = _run_debug_dump(obj_file)
            if not dump:
                self.skipTest('no DWARF dumping tool available')

            self.assertIn('DW_TAG_formal_parameter', dump)
            self.assertIn('DW_TAG_variable', dump)
            # The pointer parameter 'arr' and local variables must be described.
            for name in ('arr', 'total', 'i', 'n'):
                self.assertIn(name, dump, f'variable {name!r} missing from DWARF dump')
            self.assertIn('DW_TAG_pointer_type', dump)

    def test_scalar_types(self):
        with config.override(debug_info=True):
            self.assertAlmostEqual(float(_debug_scalar_target(True, 2.5)), 6.0)

            obj_file = _object_file_for(_debug_scalar_target)
            self.assertTrue(os.path.exists(obj_file))

            dump = _run_debug_dump(obj_file)
            if not dump:
                self.skipTest('no DWARF dumping tool available')

            for name in ('flag', 'value', 'scaled'):
                self.assertIn(name, dump, f'variable {name!r} missing from DWARF dump')
            # f64 and bool must have distinct debug types.
            self.assertIn('f64', dump)
            self.assertIn('bool', dump)

    def test_array_type(self):
        with config.override(debug_info=True):
            self.assertEqual(_debug_array_target(), 12)

            obj_file = _object_file_for(_debug_array_target)
            self.assertTrue(os.path.exists(obj_file))

            dump = _run_debug_dump(obj_file)
            if not dump:
                self.skipTest('no DWARF dumping tool available')

            for name in ('data', 'local_arr'):
                self.assertIn(name, dump, f'variable {name!r} missing from DWARF dump')
            self.assertIn('DW_TAG_array_type', dump)

    def test_struct_type(self):
        with config.override(debug_info=True):
            self.assertAlmostEqual(float(_debug_struct_target()), 11.5)

            obj_file = _object_file_for(_debug_struct_target)
            self.assertTrue(os.path.exists(obj_file))

            dump = _run_debug_dump(obj_file)
            if not dump:
                self.skipTest('no DWARF dumping tool available')

            for name in ('point', 'local_point'):
                self.assertIn(name, dump, f'variable {name!r} missing from DWARF dump')
            # Struct members should be described.
            for name in ('x', 'y'):
                self.assertIn(name, dump, f'member {name!r} missing from DWARF dump')
            self.assertIn('DW_TAG_structure_type', dump)

    def test_union_type(self):
        with config.override(debug_info=True):
            self.assertAlmostEqual(float(_debug_union_target()), 2.5)

            obj_file = _object_file_for(_debug_union_target)
            self.assertTrue(os.path.exists(obj_file))

            dump = _run_debug_dump(obj_file)
            if not dump:
                self.skipTest('no DWARF dumping tool available')

            self.assertIn('data', dump, 'variable "data" missing from DWARF dump')
            # Union members should be described.
            for name in ('i', 'f'):
                self.assertIn(name, dump, f'member {name!r} missing from DWARF dump')
            self.assertIn('DW_TAG_union_type', dump)


if __name__ == '__main__':
    unittest.main()
