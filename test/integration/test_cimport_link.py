# -*- coding: utf-8 -*-
"""
Tests for cimport Phase 4: linklibrary, system include discovery, and
bindings cache keying.

Covers:
- linklibrary(.so): dlopen into the process so JIT-compiled code and
  Python-side extern calls resolve the symbols, plus registration for AOT
  linking
- cimport(..., libraries=[...]) convenience wiring
- linklibrary(.a): static archives resolve through the extern-objects
  bundle
- linklibrary(.bc/.ll): rejected with a clear error (bitcode linking is
  not supported)
- System headers importable by name (cimport('stdio.h')) via host
  toolchain include discovery
- PC_CIMPORT_INCLUDE_PATH env var
- Bindings cache keyed on parse options: same header with different
  defines/target produces distinct modules

Note: @compile wrappers are defined at module level because pythoc requires
all @compile definitions to precede the first native call from this module.
"""
from __future__ import annotations

import os
import platform
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
# Module-level fixtures: shared/static libraries, cimport modules, wrappers
# =============================================================================

_fixture_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'build', 'test', 'cimport_link'))
os.makedirs(_fixture_dir, exist_ok=True)


def _write_fixture(name: str, content: str) -> str:
    path = os.path.join(_fixture_dir, name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path


if _BACKEND_AVAILABLE:
    from pythoc import linklibrary
    from pythoc.cimport import cimport
    from pythoc.utils.cc_utils import compile_c_to_object
    from pythoc.utils.link_utils import archive_files, link_files

    def _build_shared(name: str, source: str) -> str:
        src = _write_fixture(name + '.c', source)
        obj = compile_c_to_object(src, os.path.join(_fixture_dir, name + '.o'))
        return link_files(
            [obj], os.path.join(_fixture_dir, 'lib' + name + '.so'),
            shared=True, link_objects=[])

    # --- Shared library loaded via linklibrary() ---
    _add_so = _build_shared(
        'linklib_add',
        'int pc_link_add(int a, int b) { return a + b + 1; }\n')
    linklibrary(_add_so)
    _add_mod = cimport(
        _write_fixture('linklib_add.h', 'int pc_link_add(int a, int b);\n'),
        lib='')

    @compile
    def call_link_add(a: i32, b: i32) -> i32:
        return _add_mod.pc_link_add(a, b)

    # --- Shared library loaded via cimport(libraries=[...]) ---
    _sub_so = _build_shared(
        'linklib_sub',
        'int pc_link_sub(int a, int b) { return a - b; }\n')
    _sub_mod = cimport(
        _write_fixture('linklib_sub.h', 'int pc_link_sub(int a, int b);\n'),
        lib='', libraries=[_sub_so])

    @compile
    def call_link_sub(a: i32, b: i32) -> i32:
        return _sub_mod.pc_link_sub(a, b)

    # --- Static archive loaded via linklibrary() ---
    _mul_src = _write_fixture(
        'linklib_mul.c',
        'int pc_link_mul(int a, int b) { return a * b; }\n')
    _mul_obj = compile_c_to_object(
        _mul_src, os.path.join(_fixture_dir, 'linklib_mul.o'))
    _mul_a = archive_files(
        [_mul_obj], os.path.join(_fixture_dir, 'liblinklib_mul.a'))
    linklibrary(_mul_a)
    _mul_mod = cimport(
        _write_fixture('linklib_mul.h', 'int pc_link_mul(int a, int b);\n'),
        lib='')

    @compile
    def call_link_mul(a: i32, b: i32) -> i32:
        return _mul_mod.pc_link_mul(a, b)


@unittest.skipUnless(_BACKEND_AVAILABLE, 'clang backend or C compiler unavailable')
class TestLinklibrary(unittest.TestCase):
    def test_shared_lib_jit_call(self):
        self.assertEqual(call_link_add(1, 2), 4)

    def test_shared_lib_python_side_call(self):
        self.assertEqual(_add_mod.pc_link_add(3, 4), 8)

    def test_libraries_kwarg_jit_call(self):
        self.assertEqual(call_link_sub(10, 3), 7)

    def test_registered_for_aot_link(self):
        from pythoc.registry import get_unified_registry
        libs = get_unified_registry().get_link_libraries()
        self.assertIn(os.path.abspath(_add_so), libs)

    def test_idempotent_reload(self):
        self.assertEqual(linklibrary(_add_so), _add_so)

    def test_static_archive_jit_call(self):
        self.assertEqual(call_link_mul(6, 7), 42)

    def test_static_archive_python_side_call(self):
        self.assertEqual(_mul_mod.pc_link_mul(2, 5), 10)

    def test_bitcode_rejected(self):
        path = _write_fixture('fake_bitcode.bc', 'BC')
        with self.assertRaises(ValueError) as ctx:
            linklibrary(path)
        self.assertIn('bitcode', str(ctx.exception))

    def test_missing_library_raises(self):
        with self.assertRaises(FileNotFoundError):
            linklibrary(os.path.join(_fixture_dir, 'definitely_missing.so'))


@unittest.skipUnless(_BACKEND_AVAILABLE, 'clang backend or C compiler unavailable')
class TestSystemIncludes(unittest.TestCase):
    def setUp(self):
        self._saved_env = os.environ.get('PC_CIMPORT_INCLUDE_PATH')

    def tearDown(self):
        if self._saved_env is None:
            os.environ.pop('PC_CIMPORT_INCLUDE_PATH', None)
        else:
            os.environ['PC_CIMPORT_INCLUDE_PATH'] = self._saved_env

    def test_system_header_by_name(self):
        mod = cimport('stdio.h', lib='c')
        self.assertTrue(hasattr(mod, 'puts'))
        self.assertTrue(hasattr(mod, 'printf'))

    def test_env_include_path(self):
        custom_dir = tempfile.mkdtemp(dir=_fixture_dir)
        try:
            with open(os.path.join(custom_dir, 'env_only_hdr.h'), 'w',
                      encoding='utf-8') as f:
                f.write('#define ENV_MAGIC 41\nint env_only_fn(int x);\n')
            os.environ['PC_CIMPORT_INCLUDE_PATH'] = custom_dir
            mod = cimport('env_only_hdr.h', lib='c')
            self.assertEqual(mod.ENV_MAGIC, 41)
            self.assertTrue(hasattr(mod, 'env_only_fn'))
        finally:
            shutil.rmtree(custom_dir, ignore_errors=True)


@unittest.skipUnless(_BACKEND_AVAILABLE, 'clang backend or C compiler unavailable')
class TestCimportCacheKey(unittest.TestCase):
    def test_defines_produce_distinct_bindings(self):
        from pythoc.cimport import cimport
        header = _write_fixture('cache_key_defs.h', '''
#ifndef PC_CACHE_KEY_FOO
#define PC_CACHE_KEY_FOO 0
#endif
enum CacheKeyDefs { PC_CACHE_KEY_VAL = PC_CACHE_KEY_FOO };
''')
        mod1 = cimport(header, lib='c', defines=['PC_CACHE_KEY_FOO=1'])
        mod2 = cimport(header, lib='c', defines=['PC_CACHE_KEY_FOO=2'])
        self.assertIsNot(mod1, mod2)
        self.assertEqual(mod1.PC_CACHE_KEY_VAL, 1)
        self.assertEqual(mod2.PC_CACHE_KEY_VAL, 2)

    def test_targets_do_not_collide(self):
        from pythoc.cimport import cimport
        header = _write_fixture('cache_key_target.h', '''
typedef long double pc_cache_ld_t;
typedef int pc_cache_plain_t;
''')
        host_arch = platform.machine().lower()
        if host_arch in ('arm64', 'aarch64'):
            cross_target = 'x86_64-unknown-linux-gnu'
        else:
            cross_target = 'aarch64-unknown-linux-gnu'

        mod_host = cimport(header, lib='c')
        mod_cross = cimport(header, lib='c', target=cross_target)
        self.assertIsNot(mod_host, mod_cross)
        self.assertTrue(hasattr(mod_host, 'pc_cache_plain_t'))
        self.assertTrue(hasattr(mod_cross, 'pc_cache_plain_t'))

        def probe(mod):
            try:
                ty = mod.pc_cache_ld_t
                return getattr(ty, '__name__', repr(ty))
            except RuntimeError:
                return 'unsupported'

        # long double maps differently across targets (f64/f128/unsupported);
        # stale cache reuse would show identical bindings for both modules.
        self.assertNotEqual(probe(mod_host), probe(mod_cross))


if __name__ == '__main__':
    unittest.main()
