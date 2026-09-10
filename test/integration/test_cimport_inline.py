# -*- coding: utf-8 -*-
"""
End-to-end tests for cimport of header-defined inline/static functions.

A function defined in a header as ``static inline`` (or plain ``static``,
or C99 ``inline`` without ``extern``) has no external symbol in any library.
cimport generates a stub .c with a forwarding wrapper per such function,
compiles it with the system C compiler, and registers the object for
linking, so the functions are callable from @compile code without
compile_sources=True and without a library.

Covers:
- static inline / C99 inline / plain static functions, called from @compile
- pointer arguments and struct-by-value arguments through the cc wrapper
- variadic static inline -> lazy error via _unsupported_symbols
- kind='source' with compile_sources keeps existing behavior (no stub)
- wrapper symbol disambiguation across headers with same-named functions
- header edits invalidate bindings and the stub object

Note: @compile wrappers are defined at module level because pythoc requires
all @compile definitions to precede the first native call from this module.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from pythoc import compile, i32, ptr


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
    os.path.dirname(__file__), '..', '..', 'build', 'test', 'cimport_inline'))
os.makedirs(_fixture_dir, exist_ok=True)


def _write_fixture(name: str, content: str) -> str:
    path = os.path.join(_fixture_dir, name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path


if _BACKEND_AVAILABLE:
    from pythoc.cimport import cimport

    # --- Case 1: header-only static inline / C99 inline / plain static ---
    _case1_header = _write_fixture('inl_math.h', '''
static inline int inl_add(int a, int b) { return a + b; }
inline int inl_mul(int a, int b) { return a * b; }
static int inl_sub(int a, int b) { return a - b; }
static inline int inl_scale(const int *p, int k) { return *p * k; }

struct InlPoint { int x; int y; };
static inline struct InlPoint inl_mkpoint(int x, int y) {
    struct InlPoint p = {x, y};
    return p;
}
static inline int inl_dotsum(struct InlPoint p) { return p.x + p.y; }
''')
    _case1_mod = cimport(_case1_header)
    inl_add = _case1_mod.inl_add
    inl_mul = _case1_mod.inl_mul
    inl_sub = _case1_mod.inl_sub
    inl_scale = _case1_mod.inl_scale
    inl_mkpoint = _case1_mod.inl_mkpoint
    inl_dotsum = _case1_mod.inl_dotsum
    InlPoint = _case1_mod.InlPoint

    @compile
    def call_inl_add(a: i32, b: i32) -> i32:
        return inl_add(a, b)

    @compile
    def call_inl_mul(a: i32, b: i32) -> i32:
        return inl_mul(a, b)

    @compile
    def call_inl_sub(a: i32, b: i32) -> i32:
        return inl_sub(a, b)

    @compile
    def call_inl_scale(v: i32, k: i32) -> i32:
        return inl_scale(ptr(v), k)

    @compile
    def call_inl_struct(x: i32, y: i32) -> i32:
        p: InlPoint = inl_mkpoint(x, y)
        return inl_dotsum(p)

    # --- Case 2: .c source whose non-static functions use static inline ---
    _case2_source = _write_fixture('inl_src.c', '''
static inline int triple(int x) { return x * 3; }
static int hidden_helper(int x) { return x + 1; }
int use_inline(int x) { return triple(x) + hidden_helper(x); }
''')
    _case2_mod = cimport(_case2_source, compile_sources=True)
    use_inline = _case2_mod.use_inline

    @compile
    def call_use_inline(x: i32) -> i32:
        return use_inline(x)

    # --- Case 3: two headers defining a same-named static inline function ---
    _case3_dir_a = os.path.join(_fixture_dir, 'dup_a')
    _case3_dir_b = os.path.join(_fixture_dir, 'dup_b')
    os.makedirs(_case3_dir_a, exist_ok=True)
    os.makedirs(_case3_dir_b, exist_ok=True)
    _case3_header_a = _write_fixture(
        os.path.join('dup_a', 'dup.h'),
        'static inline int add(int a, int b) { return a + b; }\n')
    _case3_header_b = _write_fixture(
        os.path.join('dup_b', 'dup.h'),
        'static inline int add(int a, int b) { return 10 * (a + b); }\n')
    _case3_mod_a = cimport(_case3_header_a)
    _case3_mod_b = cimport(_case3_header_b)
    dup_add_a = _case3_mod_a.add
    dup_add_b = _case3_mod_b.add

    @compile
    def call_dup_a(a: i32, b: i32) -> i32:
        return dup_add_a(a, b)

    @compile
    def call_dup_b(a: i32, b: i32) -> i32:
        return dup_add_b(a, b)


class CimportInlineBase(unittest.TestCase):
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


class TestCimportInlineCalls(CimportInlineBase):
    """Header-defined functions are callable from @compile with no lib and
    no compile_sources (JIT side)."""

    def test_static_inline(self):
        self.assertEqual(call_inl_add(3, 4), 7)
        self.assertEqual(call_inl_add(-10, 10), 0)

    def test_c99_inline(self):
        self.assertEqual(call_inl_mul(6, 7), 42)

    def test_plain_static(self):
        self.assertEqual(call_inl_sub(10, 3), 7)
        self.assertEqual(call_inl_sub(5, 10), -5)

    def test_pointer_argument(self):
        self.assertEqual(call_inl_scale(21, 2), 42)

    def test_struct_by_value(self):
        self.assertEqual(call_inl_struct(20, 22), 42)

    def test_wrapper_bindings_use_generated_symbol(self):
        # The Python-facing name stays the original; the C symbol is the
        # wrapper in the cc-compiled stub object.
        self.assertEqual(inl_add.func_name, 'inl_add')
        self.assertTrue(inl_add.c_name.startswith('__pythoc_inl_'))
        self.assertTrue(inl_add.c_name.endswith('_inl_add'))


class TestCimportSourceWithInline(CimportInlineBase):
    """A .c imported with compile_sources compiles as its own translation
    unit: no wrapper stub is generated (it would duplicate symbols), and
    static inline functions keep working inside their own TU."""

    def test_non_static_function_using_statics(self):
        self.assertEqual(call_use_inline(5), 21)

    def test_no_stub_generated(self):
        from pythoc.cimport_wrappers import stub_paths
        cache_dir = os.path.dirname(_case2_mod.__file__)
        bindings_file = os.path.basename(_case2_mod.__file__)
        cached_name = bindings_file[len('bindings_clang_'):-len('.py')]
        stub_c, _ = stub_paths(cache_dir, cached_name)
        self.assertFalse(os.path.exists(stub_c))


class TestCimportInlineCollision(CimportInlineBase):
    """Same-named static inline functions from different headers get
    distinct wrapper symbols (path hash in the wrapper prefix)."""

    def test_distinct_symbols_and_results(self):
        self.assertNotEqual(dup_add_a.c_name, dup_add_b.c_name)
        self.assertEqual(call_dup_a(1, 2), 3)
        self.assertEqual(call_dup_b(1, 2), 30)


class TestCimportInlineUnsupported(CimportInlineBase):
    """Signatures that cannot be wrapped produce lazy errors."""

    def test_variadic_static_inline_lazy_error(self):
        from pythoc.cimport import cimport

        header = self._write("va_inl.h", """
static inline int va_sum(int first, ...) { return first; }
static inline int va_ok(int a, int b) { return a + b; }
""")
        mod = cimport(header)
        self.assertTrue(hasattr(mod, "va_ok"))
        with self.assertRaises(RuntimeError) as ctx:
            mod.va_sum
        self.assertIn("variadic", str(ctx.exception))


class TestCimportInlineCacheInvalidation(CimportInlineBase):
    """Editing the header regenerates bindings and rebuilds the stub .o."""

    def test_header_change_rebuilds_stub(self):
        import time
        from pythoc.cimport import cimport
        from pythoc.cimport_wrappers import stub_paths

        header = self._write("inl_redef.h",
                             "static inline int rd_fn(int x) { return x + 1; }\n")
        mod1 = cimport(header)
        self.assertTrue(hasattr(mod1, "rd_fn"))
        cache_dir = os.path.dirname(mod1.__file__)
        # The stub shares the bindings' cache-keyed name:
        # bindings_clang_<name>_<hash>.py -> <name>_<hash>_pythoc_inl.[co]
        bindings_file = os.path.basename(mod1.__file__)
        self.assertTrue(bindings_file.startswith('bindings_clang_'))
        cached_name = bindings_file[len('bindings_clang_'):-len('.py')]
        stub_c, stub_o = stub_paths(cache_dir, cached_name)
        self.assertTrue(os.path.exists(stub_c))
        self.assertTrue(os.path.exists(stub_o))
        # Backdate the stub object so a rebuild is observable via mtime.
        os.utime(stub_o, (1000000, 1000000))

        # Rewrite the header with a strictly newer mtime for both bindings
        # and stub-object freshness checks.
        future = time.time() + 2
        with open(header, 'w', encoding='utf-8') as f:
            f.write("static inline int rd_fn(int x) { return x + 2; }\n"
                    "static inline int rd_fn2(int x) { return x + 3; }\n")
        os.utime(header, (future, future))

        mod2 = cimport(header)
        self.assertTrue(hasattr(mod2, "rd_fn2"))
        # Stub source and object were regenerated after the header edit.
        with open(stub_c, encoding='utf-8') as f:
            self.assertIn('rd_fn2', f.read())
        self.assertGreater(os.path.getmtime(stub_o), 1000000)


if __name__ == "__main__":
    unittest.main()
