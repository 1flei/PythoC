# -*- coding: utf-8 -*-
"""
End-to-end tests for cimport extern global variables.

Covers:
- Read and write of an extern global from @compile functions
- Effects of writes visible across calls and to C functions
- Header-declared extern global backed by a source file
- Python-side read/write via the module attribute (.value)
- static C globals and thread-local globals produce lazy errors on access

Note: @compile wrappers are defined at module level because pythoc requires
all @compile definitions to precede the first native call from this module.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from pythoc import compile, i32


def _clang_backend_available() -> bool:
    try:
        from pythoc.cimport_clang import is_clang_backend_available
    except Exception:
        return False
    return is_clang_backend_available()


def _cc_available() -> bool:
    try:
        from pythoc.utils.cc_utils import find_available_cc
        find_available_cc()
    except RuntimeError:
        return False
    return True


_BACKEND_AVAILABLE = _clang_backend_available() and _cc_available()

# =============================================================================
# Module-level fixtures: cimport + @compile wrappers
# =============================================================================

_fixture_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'build', 'test', 'cimport_globals'))
os.makedirs(_fixture_dir, exist_ok=True)


def _write_fixture(name: str, content: str) -> str:
    path = os.path.join(_fixture_dir, name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path


if _BACKEND_AVAILABLE:
    from pythoc.cimport import cimport

    # --- Case 1: global counter + accessor functions in one C source ---
    _case1_path = _write_fixture('counter.c', '''
int global_counter = 10;
void inc_counter(void) { global_counter += 1; }
int read_counter(void) { return global_counter; }
''')
    _case1_mod = cimport(_case1_path, compile_sources=True)
    inc_counter = _case1_mod.inc_counter
    read_counter = _case1_mod.read_counter
    global_counter = _case1_mod.global_counter

    @compile
    def bump_counter() -> i32:
        inc_counter()
        global_counter += 100
        return global_counter

    @compile
    def overwrite_counter() -> i32:
        global_counter = 7
        return read_counter()

    @compile
    def reset_counter(v: i32) -> i32:
        global_counter = v
        return read_counter()

    # --- Case 2: header-declared extern global backed by a source ---
    _case2_header = _write_fixture('gvar.h', '''
extern int gvar;
int gvar_get(void);
void gvar_set(int v);
''')
    _case2_source = _write_fixture('gvar.c', '''
#include "gvar.h"
int gvar = 3;
int gvar_get(void) { return gvar; }
void gvar_set(int v) { gvar = v; }
''')
    _case2_mod = cimport(_case2_header, sources=[_case2_source],
                         compile_sources=True, include_dirs=[_fixture_dir])
    gvar = _case2_mod.gvar
    gvar_get = _case2_mod.gvar_get
    gvar_set = _case2_mod.gvar_set

    @compile
    def read_gvar() -> i32:
        return gvar

    @compile
    def add_to_gvar(x: i32) -> i32:
        gvar = gvar + x
        return gvar

    # --- Case 3: Python-side read/write roundtrip ---
    _case3_path = _write_fixture('pyside.c', '''
int pyside_value = 123;
int pyside_get(void) { return pyside_value; }
''')
    _case3_mod = cimport(_case3_path, compile_sources=True)
    pyside_value = _case3_mod.pyside_value
    pyside_get = _case3_mod.pyside_get

    @compile
    def read_pyside() -> i32:
        return pyside_value

    @compile
    def write_pyside() -> i32:
        pyside_value = 66
        return pyside_value

    # --- Case 4: a second module whose @compile code lives in its own
    # group .so, used to prove two groups share one storage copy ---
    _helper_path = _write_fixture('cimport_globals_helper.py', f'''\
from pythoc import compile, i32
from pythoc.cimport import cimport

_helper_mod = cimport({_case1_path!r}, compile_sources=True)
global_counter = _helper_mod.global_counter


@compile
def helper_read_counter() -> i32:
    return global_counter
''')


class CimportGlobalsBase(unittest.TestCase):
    def setUp(self):
        if not _BACKEND_AVAILABLE:
            self.skipTest("clang/libclang bindings or C compiler not available")
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write(self, name: str, content: str) -> str:
        path = os.path.join(self.temp_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path


class TestCimportGlobalReadWrite(CimportGlobalsBase):
    """Read/write an extern global from @compile code."""

    def test_compiled_read_write_visible_to_c(self):
        reset_counter(10)
        self.assertEqual(bump_counter(), 111)
        # Writes from compiled code are visible on later calls
        self.assertEqual(bump_counter(), 212)
        self.assertEqual(read_counter(), 212)

    def test_compiled_overwrite_and_c_write(self):
        reset_counter(10)
        self.assertEqual(overwrite_counter(), 7)
        # Writes from C code are visible to compiled reads
        self.assertEqual(bump_counter(), 108)


class TestCimportHeaderExternGlobal(CimportGlobalsBase):
    """Header-declared extern global backed by a compiled source."""

    def test_header_extern_with_source(self):
        self.assertEqual(read_gvar(), 3)
        self.assertEqual(add_to_gvar(4), 7)
        self.assertEqual(gvar_get(), 7)
        gvar_set(42)
        self.assertEqual(read_gvar(), 42)


class TestCimportGlobalPythonSide(CimportGlobalsBase):
    """Python-side read/write of the global via the module attribute."""

    def test_python_side_read_write(self):
        # Compile + execute once so the group .so (with the data symbol)
        # is loaded RTLD_GLOBAL into the process.
        self.assertEqual(read_pyside(), 123)

        # Python-side read via ctypes.in_dll
        self.assertEqual(_case3_mod.pyside_value.value, 123)

        # Python-side write is visible to compiled reads
        _case3_mod.pyside_value.value = 55
        self.assertEqual(read_pyside(), 55)
        self.assertEqual(pyside_get(), 55)

        # Compiled writes are visible to Python-side reads
        self.assertEqual(write_pyside(), 66)
        self.assertEqual(_case3_mod.pyside_value.value, 66)


class TestCimportCrossGroupSingleCopy(CimportGlobalsBase):
    """Two group .so files share one storage copy of the same extern global.

    Regression test: registry link objects (cimport compile_sources
    products) used to be statically linked into every group .so, so each
    group privately held its own copy of a cimported global's definition
    and reads/writes diverged depending on which group .so observed them.
    """

    def test_two_group_libraries_share_one_storage(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            'cimport_globals_helper', _helper_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)

        reset_counter(10)
        # helper_read_counter lives in the helper module's own group .so.
        self.assertEqual(helper.helper_read_counter(), 10)
        # A write from this module's group .so must be visible there.
        self.assertEqual(bump_counter(), 111)
        self.assertEqual(helper.helper_read_counter(), 111)


class TestCimportUnsupportedGlobals(CimportGlobalsBase):
    """static and thread-local C globals are lazy errors, not wrong bindings."""

    def test_static_global_lazy_error(self):
        from pythoc.cimport import cimport

        source = self._write("static_g.c", """
static int file_scope_counter = 1;
int public_counter = 2;
""")
        mod = cimport(source, compile_sources=True)
        self.assertTrue(hasattr(mod, "public_counter"))
        with self.assertRaises(RuntimeError) as ctx:
            mod.file_scope_counter
        self.assertIn("static global", str(ctx.exception))

    def test_thread_local_global_lazy_error(self):
        from pythoc.cimport import cimport

        source = self._write("tls_g.c", """
_Thread_local int tls_counter = 1;
int plain_counter = 2;
""")
        mod = cimport(source, compile_sources=True)
        self.assertTrue(hasattr(mod, "plain_counter"))
        with self.assertRaises(RuntimeError) as ctx:
            mod.tls_counter
        self.assertIn("thread-local", str(ctx.exception))

    def test_unsupported_global_type_lazy_error(self):
        from pythoc.cimport import cimport

        # long double on x86_64 has no matching pythoc type; the global must
        # degrade to a lazy error like other unsupported declarations.
        header = self._write("ld_global.h", """
extern long double ld_global;
extern int plain_global;
""")
        mod = cimport(header, lib="c", target="x86_64-unknown-linux-gnu")
        self.assertTrue(hasattr(mod, "plain_global"))
        with self.assertRaises(RuntimeError) as ctx:
            mod.ld_global
        self.assertIn("long double", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
