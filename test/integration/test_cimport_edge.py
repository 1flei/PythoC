# -*- coding: utf-8 -*-
"""
Edge-case tests for cimport with libclang backend.

Tests advanced C type coverage:
- Pointer to pointer (int**)
- Function pointer typedefs
- Opaque struct (forward declarations)
- Variadic functions
- Nested struct pointers
- include_dirs / defines parameters
- Enum with negative / unsigned values
- Bitfields
- Empty struct
- Global extern variables
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest


def _clang_backend_available() -> bool:
    try:
        from pythoc.cimport_clang import is_clang_backend_available
    except Exception:
        return False
    return is_clang_backend_available()


class TestCimportPointers(unittest.TestCase):
    """Test cimport with various pointer types."""

    def setUp(self):
        if not _clang_backend_available():
            self.skipTest("clang/libclang Python bindings are not available")
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write(self, name: str, content: str) -> str:
        path = os.path.join(self.temp_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_pointer_to_pointer(self):
        """int** should become ptr[ptr[i32]]"""
        from pythoc.cimport import cimport

        header = self._write("pp.h", "int **pp_func(void);")
        mod = cimport(header, lib="c")
        self.assertTrue(hasattr(mod, "pp_func"))

    def test_const_pointer(self):
        """const int* should be parsed correctly"""
        from pythoc.cimport import cimport

        header = self._write("cptr.h", "int cptr_func(const int *p);")
        mod = cimport(header, lib="c")
        self.assertTrue(hasattr(mod, "cptr_func"))

    def test_void_pointer(self):
        """void* should become ptr[void]"""
        from pythoc.cimport import cimport

        header = self._write("vptr.h", "void *vptr_func(void);")
        mod = cimport(header, lib="c")
        self.assertTrue(hasattr(mod, "vptr_func"))


class TestCimportFunctionPointers(unittest.TestCase):
    """Test cimport with function pointer typedefs."""

    def setUp(self):
        if not _clang_backend_available():
            self.skipTest("clang/libclang Python bindings are not available")
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write(self, name: str, content: str) -> str:
        path = os.path.join(self.temp_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_function_pointer_typedef(self):
        """typedef void (*callback_t)(int) should generate func type"""
        from pythoc.cimport import cimport

        header = self._write("fptr.h", """
typedef void (*callback_t)(int);
void register_cb(callback_t cb);
""")
        mod = cimport(header, lib="c")
        self.assertTrue(hasattr(mod, "callback_t"))
        self.assertTrue(hasattr(mod, "register_cb"))

    def test_function_pointer_with_return(self):
        """typedef int (*mapper_t)(int, int)"""
        from pythoc.cimport import cimport

        header = self._write("fptr_ret.h", """
typedef int (*mapper_t)(int, int);
mapper_t get_mapper(void);
""")
        mod = cimport(header, lib="c")
        self.assertTrue(hasattr(mod, "mapper_t"))
        self.assertTrue(hasattr(mod, "get_mapper"))


class TestCimportOpaqueStruct(unittest.TestCase):
    """Test cimport with forward declarations / opaque structs."""

    def setUp(self):
        if not _clang_backend_available():
            self.skipTest("clang/libclang Python bindings are not available")
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write(self, name: str, content: str) -> str:
        path = os.path.join(self.temp_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_opaque_struct_pointer(self):
        """Forward-declared struct used via pointer only"""
        from pythoc.cimport import cimport

        header = self._write("opaque.h", """
struct OpaqueCtx;
struct OpaqueCtx *opaque_create(void);
void opaque_destroy(struct OpaqueCtx *ctx);
""")
        mod = cimport(header, lib="c")
        self.assertTrue(hasattr(mod, "opaque_create"))
        self.assertTrue(hasattr(mod, "opaque_destroy"))


class TestCimportVariadic(unittest.TestCase):
    """Test cimport with variadic functions."""

    def setUp(self):
        if not _clang_backend_available():
            self.skipTest("clang/libclang Python bindings are not available")
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write(self, name: str, content: str) -> str:
        path = os.path.join(self.temp_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_variadic_function(self):
        """int printf(const char *fmt, ...) should be parsed"""
        from pythoc.cimport import cimport

        header = self._write("variadic.h", "int my_printf(const char *fmt, ...);")
        mod = cimport(header, lib="c")
        self.assertTrue(hasattr(mod, "my_printf"))


class TestCimportIncludeDirsAndDefines(unittest.TestCase):
    """Test cimport with include_dirs and defines parameters."""

    def setUp(self):
        if not _clang_backend_available():
            self.skipTest("clang/libclang Python bindings are not available")
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write(self, name: str, content: str) -> str:
        path = os.path.join(self.temp_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_include_dirs(self):
        """cimport should resolve #include from custom include_dirs"""
        from pythoc.cimport import cimport

        # Create an include directory with a header
        inc_dir = os.path.join(self.temp_dir, "myincludes")
        os.makedirs(inc_dir)
        with open(os.path.join(inc_dir, "mylib.h"), "w") as f:
            f.write("int mylib_add(int a, int b);\n")

        # Main header includes the custom one
        header = self._write("with_include.h", """
#include "mylib.h"
int with_include_func(void);
""")
        mod = cimport(header, lib="c", include_dirs=[inc_dir])
        self.assertTrue(hasattr(mod, "with_include_func"))

    def test_defines(self):
        """cimport should pass -D defines to clang"""
        from pythoc.cimport import cimport

        header = self._write("defines.h", """
#ifdef USE_FEATURE_X
int feature_x_enabled(void);
#else
int feature_x_disabled(void);
#endif
""")
        mod = cimport(header, lib="c", defines=["USE_FEATURE_X"])
        self.assertTrue(hasattr(mod, "feature_x_enabled"))
        self.assertFalse(hasattr(mod, "feature_x_disabled"))


class TestCimportEnums(unittest.TestCase):
    """Test cimport with various enum forms."""

    def setUp(self):
        if not _clang_backend_available():
            self.skipTest("clang/libclang Python bindings are not available")
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write(self, name: str, content: str) -> str:
        path = os.path.join(self.temp_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_enum_with_negative_values(self):
        """Enum with negative explicit values"""
        from pythoc.cimport import cimport

        header = self._write("neg_enum.h", """
enum ErrorCode {
    ERR_NONE = 0,
    ERR_NOT_FOUND = -1,
    ERR_DENIED = -2,
    ERR_AFTER,
};
""")
        mod = cimport(header, lib="c")
        self.assertTrue(hasattr(mod, "ErrorCode"))
        # Value correctness: negative tags must survive, and the following
        # auto-numbered constant continues from the negative tag.
        self.assertEqual(mod.ErrorCode.ERR_NONE, 0)
        self.assertEqual(mod.ErrorCode.ERR_NOT_FOUND, -1)
        self.assertEqual(mod.ErrorCode.ERR_DENIED, -2)
        self.assertEqual(mod.ErrorCode.ERR_AFTER, -1)
        # C enum constants are also visible in the enclosing (module) scope
        self.assertEqual(mod.ERR_NOT_FOUND, -1)
        self.assertEqual(mod.ERR_AFTER, -1)

    def test_enum_with_large_values(self):
        """Enum with large unsigned-like values"""
        from pythoc.cimport import cimport

        header = self._write("large_enum.h", """
enum Flags {
    FLAG_A = 1,
    FLAG_B = 0x100,
    FLAG_C = 0xFFFFFFFF,
};
""")
        mod = cimport(header, lib="c")
        self.assertTrue(hasattr(mod, "Flags"))
        # Module-level constants mirror the clang-folded enum values
        self.assertEqual(mod.FLAG_A, 1)
        self.assertEqual(mod.FLAG_B, 0x100)
        self.assertIn(mod.FLAG_C, (0xFFFFFFFF, -1))


class TestCimportNestedStruct(unittest.TestCase):
    """Test cimport with structs containing pointers to other structs."""

    def setUp(self):
        if not _clang_backend_available():
            self.skipTest("clang/libclang Python bindings are not available")
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write(self, name: str, content: str) -> str:
        path = os.path.join(self.temp_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_struct_with_pointer_to_struct(self):
        """Struct containing a pointer to another struct.
        Note: self-referential structs (ptr[Node] in Node) are a known
        limitation of the current emitter. Test only non-recursive case."""
        from pythoc.cimport import cimport

        header = self._write("nested.h", """
struct Inner {
    int value;
};

struct Outer {
    struct Inner *data;
    int count;
};
""")
        mod = cimport(header, lib="c")
        self.assertTrue(hasattr(mod, "Inner"))
        self.assertTrue(hasattr(mod, "Outer"))

    def test_struct_with_array_field(self):
        """Struct containing a fixed-size array field"""
        from pythoc.cimport import cimport

        header = self._write("arr_struct.h", """
struct Buffer {
    int data[16];
    int len;
};
""")
        mod = cimport(header, lib="c")
        self.assertTrue(hasattr(mod, "Buffer"))


class TestCimportBitfields(unittest.TestCase):
    """Test cimport with bitfield struct fields."""

    def setUp(self):
        if not _clang_backend_available():
            self.skipTest("clang/libclang Python bindings are not available")
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write(self, name: str, content: str) -> str:
        path = os.path.join(self.temp_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_struct_with_bitfield(self):
        """Struct with bitfield fields degrades to opaque storage.

        Bitfields cannot be expressed as pythoc fields without producing a
        wrong ABI layout, so the struct degrades to an opaque record with
        matching size: no full-width fields are emitted, field access fails,
        and the type stays usable as a pointer pointee."""
        from pythoc.cimport import cimport

        header = self._write("bitfield.h", """
struct Flags {
    unsigned int a : 1;
    unsigned int b : 3;
    unsigned int c : 4;
};
struct Flags *flags_make(void);
""")
        mod = cimport(header, lib="c")
        self.assertTrue(hasattr(mod, "Flags"))
        # No wrong full-width fields are emitted
        self.assertFalse(mod.Flags.has_field("a"))
        self.assertFalse(mod.Flags.has_field("b"))
        self.assertFalse(mod.Flags.has_field("c"))
        self.assertTrue(mod.Flags.has_field("_storage"))
        # Opaque layout still matches clang's size
        self.assertEqual(mod.Flags.get_size_bytes(), 4)
        # Functions taking ptr to the record are still bound
        self.assertTrue(hasattr(mod, "flags_make"))


class TestCimportMacros(unittest.TestCase):
    """Test cimport extraction of numeric macro constants."""

    def setUp(self):
        if not _clang_backend_available():
            self.skipTest("clang/libclang Python bindings are not available")
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write(self, name: str, content: str) -> str:
        path = os.path.join(self.temp_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_numeric_macros(self):
        """Simple numeric macros become module-level Python constants"""
        from pythoc.cimport import cimport

        header = self._write("macros.h", """
#define FOO 42
#define NEG -3.4
#define HEX 0x1F
#define BIG 1UL
#define OCT 010
""")
        mod = cimport(header, lib="c")
        self.assertEqual(mod.FOO, 42)
        self.assertAlmostEqual(mod.NEG, -3.4)
        self.assertEqual(mod.HEX, 0x1F)
        self.assertEqual(mod.BIG, 1)
        self.assertEqual(mod.OCT, 8)

    def test_undef_redefine_uses_final_value(self):
        """A macro redefined after #undef uses the final definition"""
        from pythoc.cimport import cimport

        header = self._write("redef.h", """
#define REDEF_VALUE 1
#undef REDEF_VALUE
#define REDEF_VALUE 2
""")
        mod = cimport(header, lib="c")
        self.assertEqual(mod.REDEF_VALUE, 2)

    def test_non_numeric_macros_skipped(self):
        """Expression, string, empty and function-like macros are skipped"""
        from pythoc.cimport import cimport

        header = self._write("skip_macros.h", """
#define EXPR (1 << 2)
#define STR "hello"
#define EMPTY
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define KEPT 7
""")
        mod = cimport(header, lib="c")
        self.assertFalse(hasattr(mod, "EXPR"))
        self.assertFalse(hasattr(mod, "STR"))
        self.assertFalse(hasattr(mod, "EMPTY"))
        self.assertFalse(hasattr(mod, "MAX"))
        self.assertEqual(mod.KEPT, 7)

    def test_declaration_wins_over_macro(self):
        """A macro whose name collides with a declaration is skipped"""
        from pythoc.cimport import cimport

        header = self._write("collision.h", """
int collide_fn(void);
#define collide_fn 123
""")
        mod = cimport(header, lib="c")
        self.assertFalse(isinstance(mod.collide_fn, int))


class TestCimportLongDouble(unittest.TestCase):
    """Test cimport handling of long double across targets."""

    def setUp(self):
        if not _clang_backend_available():
            self.skipTest("clang/libclang Python bindings are not available")
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write(self, name: str, content: str) -> str:
        path = os.path.join(self.temp_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    _HEADER = """
long double ld_identity(long double x);
typedef long double ld_t;
struct LDBox {
    int tag;
    long double val;
};
int ld_plain_fn(int x);
"""

    def test_long_double_unsupported_on_x86_64(self):
        """x86_64 long double (fp80) yields lazy errors, not wrong bindings"""
        from pythoc.cimport import cimport

        header = self._write("ld_x86_64.h", self._HEADER)
        mod = cimport(header, lib="c", target="x86_64-unknown-linux-gnu")

        # Import itself succeeds; unsupported symbols fail lazily on access
        self.assertTrue(hasattr(mod, "ld_plain_fn"))
        with self.assertRaises(RuntimeError) as ctx:
            mod.ld_identity
        self.assertIn("long double", str(ctx.exception))
        with self.assertRaises(RuntimeError) as ctx:
            mod.ld_t
        self.assertIn("long double", str(ctx.exception))

        # Structs with unsupported fields degrade to opaque storage
        self.assertTrue(hasattr(mod, "LDBox"))
        self.assertFalse(mod.LDBox.has_field("val"))
        self.assertTrue(mod.LDBox.has_field("_storage"))
        self.assertEqual(mod.LDBox.get_size_bytes(), 32)

    def test_long_double_fp128_on_aarch64(self):
        """aarch64 long double (fp128) maps to pythoc f128"""
        from pythoc.cimport import cimport

        header = self._write("ld_aarch64.h", self._HEADER)
        mod = cimport(header, lib="c", target="aarch64-unknown-linux-gnu")
        self.assertTrue(hasattr(mod, "ld_identity"))
        self.assertTrue(hasattr(mod, "ld_t"))
        # Struct with an f128 field keeps real fields
        self.assertTrue(mod.LDBox.has_field("val"))


class TestCimportGlobals(unittest.TestCase):
    """Test cimport with extern global variables."""

    def setUp(self):
        if not _clang_backend_available():
            self.skipTest("clang/libclang Python bindings are not available")
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write(self, name: str, content: str) -> str:
        path = os.path.join(self.temp_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_extern_global(self):
        """Extern global variables become extern_global bindings"""
        from pythoc.cimport import cimport
        from pythoc.decorators.extern import ExternGlobal

        header = self._write("globals.h", """
extern int global_counter;
extern void *global_handle;
""")
        mod = cimport(header, lib="c")
        self.assertIsInstance(mod.global_counter, ExternGlobal)
        self.assertEqual(mod.global_counter.c_name, "global_counter")
        self.assertIsInstance(mod.global_handle, ExternGlobal)
        self.assertEqual(mod.global_handle.c_name, "global_handle")


class TestCimportClangArgs(unittest.TestCase):
    """Test cimport with clang_args parameter."""

    def setUp(self):
        if not _clang_backend_available():
            self.skipTest("clang/libclang Python bindings are not available")
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write(self, name: str, content: str) -> str:
        path = os.path.join(self.temp_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_clang_args_extra_flags(self):
        """Extra clang args should be passed through"""
        from pythoc.cimport import cimport

        header = self._write("clang_args.h", "int clang_args_func(void);")
        # -Wno-everything should not affect parsing
        mod = cimport(header, lib="c", clang_args=["-Wno-everything"])
        self.assertTrue(hasattr(mod, "clang_args_func"))


class TestCimportExport(unittest.TestCase):
    """Test cimport export / export_all parameters."""

    def setUp(self):
        if not _clang_backend_available():
            self.skipTest("clang/libclang Python bindings are not available")
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write(self, name: str, content: str) -> str:
        path = os.path.join(self.temp_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_export_specific_symbols(self):
        """export parameter should inject specified symbols into caller globals"""
        from pythoc.cimport import cimport

        header = self._write("export_sel.h", """
int export_a(void);
int export_b(void);
int export_c(void);
""")
        caller_ns = {}
        mod = cimport(header, lib="c", export=["export_a", "export_b"])
        self.assertTrue(hasattr(mod, "export_a"))
        self.assertTrue(hasattr(mod, "export_b"))
        self.assertTrue(hasattr(mod, "export_c"))

    def test_export_nonexistent_raises(self):
        """Exporting a nonexistent symbol should raise AttributeError"""
        from pythoc.cimport import cimport

        header = self._write("export_fail.h", "int export_only_this(void);")
        with self.assertRaises(AttributeError):
            cimport(header, lib="c", export=["nonexistent_symbol"])


class TestCimportEndToEnd(unittest.TestCase):
    """End-to-end cimport tests: parse + compile + call."""

    def setUp(self):
        if not _clang_backend_available():
            self.skipTest("clang/libclang Python bindings are not available")
        try:
            from pythoc.utils.cc_utils import find_available_cc
            find_available_cc()
        except RuntimeError:
            self.skipTest("C compiler not available")
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write(self, name: str, content: str) -> str:
        path = os.path.join(self.temp_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_cimport_pointer_roundtrip(self):
        """Cimport a function returning and taking pointers, call it"""
        from pythoc import compile, i32, ptr, void
        from pythoc.cimport import cimport
        from pythoc.registry import get_unified_registry

        saved = list(get_unified_registry().get_link_objects())
        source = self._write("ptr_rt.c", """
int ptr_rt_deref(const int *p) { return *p; }
""")
        try:
            mod = cimport(source, compile_sources=True)
            ptr_rt_deref = mod.ptr_rt_deref

            @compile
            def test_ptr_rt() -> i32:
                x: i32 = 42
                return ptr_rt_deref(ptr(x))

            self.assertEqual(test_ptr_rt(), 42)
        finally:
            get_unified_registry().clear_link_objects()
            for obj in saved:
                get_unified_registry().add_link_object(obj)

    def test_cimport_struct_roundtrip(self):
        """Cimport a struct + function operating on it, verify module has both"""
        from pythoc.cimport import cimport

        source = self._write("struct_rt.c", """
struct Vec2 { int x; int y; };
int vec2_dot(struct Vec2 a, struct Vec2 b) {
    return a.x * b.x + a.y * b.y;
}
""")
        # Compile to .o but don't try to call (e2e calling is in test_cimport.py)
        mod = cimport(source, lib="c")
        self.assertTrue(hasattr(mod, "Vec2"))
        self.assertTrue(hasattr(mod, "vec2_dot"))

    def test_cimport_double_pointer_roundtrip(self):
        """int** binding: C writes through a double pointer, reads it back"""
        from pythoc import compile, i32, ptr
        from pythoc.cimport import cimport
        from pythoc.registry import get_unified_registry

        saved = list(get_unified_registry().get_link_objects())
        source = self._write("pp_rt.c", """
void pp_rt_write(int **pp, int v) { **pp = v; }
int pp_rt_read(int **pp) { return **pp + 1; }
""")
        try:
            mod = cimport(source, compile_sources=True)
            pp_rt_write = mod.pp_rt_write
            pp_rt_read = mod.pp_rt_read

            @compile(suffix="cimport_e2e_pp")
            def pp_roundtrip() -> i32:
                x: i32 = 10
                p: ptr[i32] = ptr(x)
                pp: ptr[ptr[i32]] = ptr(p)
                pp_rt_write(pp, 77)
                return pp_rt_read(pp)

            self.assertEqual(pp_roundtrip(), 78)
        finally:
            get_unified_registry().clear_link_objects()
            for obj in saved:
                get_unified_registry().add_link_object(obj)

    def test_cimport_opaque_struct_pointer_roundtrip(self):
        """Opaque struct pointer created in C, handed back to C functions"""
        from pythoc import compile, i32, ptr
        from pythoc.cimport import cimport
        from pythoc.registry import get_unified_registry

        saved = list(get_unified_registry().get_link_objects())
        header = self._write("opaque_rt.h", """
struct OpaqueRt;
struct OpaqueRt *opaque_rt_create(int seed);
int opaque_rt_get(struct OpaqueRt *ctx);
void opaque_rt_destroy(struct OpaqueRt *ctx);
""")
        source = self._write("opaque_rt.c", """
#include <stdlib.h>
#include "opaque_rt.h"
struct OpaqueRt { int v; };
struct OpaqueRt *opaque_rt_create(int seed) {
    struct OpaqueRt *ctx = malloc(sizeof(struct OpaqueRt));
    ctx->v = seed * 3;
    return ctx;
}
int opaque_rt_get(struct OpaqueRt *ctx) { return ctx->v; }
void opaque_rt_destroy(struct OpaqueRt *ctx) { free(ctx); }
""")
        try:
            mod = cimport(header, sources=[source], compile_sources=True,
                          include_dirs=[self.temp_dir])
            OpaqueRt = mod.OpaqueRt
            opaque_rt_create = mod.opaque_rt_create
            opaque_rt_get = mod.opaque_rt_get
            opaque_rt_destroy = mod.opaque_rt_destroy

            @compile(suffix="cimport_e2e_opaque")
            def opaque_roundtrip(seed: i32) -> i32:
                ctx: ptr[OpaqueRt] = opaque_rt_create(seed)
                v: i32 = opaque_rt_get(ctx)
                opaque_rt_destroy(ctx)
                return v

            self.assertEqual(opaque_roundtrip(7), 21)
        finally:
            get_unified_registry().clear_link_objects()
            for obj in saved:
                get_unified_registry().add_link_object(obj)

    def test_cimport_fnptr_typedef_callback(self):
        """fn-ptr typedef: C applies an @compile callback through it"""
        from pythoc import compile, i32
        from pythoc.cimport import cimport
        from pythoc.registry import get_unified_registry

        saved = list(get_unified_registry().get_link_objects())
        header = self._write("cb_rt.h", """
typedef int (*cb_rt_t)(int);
int cb_rt_apply(cb_rt_t fn, int x);
""")
        source = self._write("cb_rt.c", """
#include "cb_rt.h"
int cb_rt_apply(cb_rt_t fn, int x) { return fn(x) + fn(x + 1); }
""")
        try:
            mod = cimport(header, sources=[source], compile_sources=True,
                          include_dirs=[self.temp_dir])
            cb_rt_apply = mod.cb_rt_apply

            @compile(suffix="cimport_e2e_cb_inner")
            def cb_double(x: i32) -> i32:
                return x * 2

            @compile(suffix="cimport_e2e_cb")
            def cb_roundtrip(x: i32) -> i32:
                return cb_rt_apply(cb_double, x)

            # fn(5) + fn(6) = 10 + 12
            self.assertEqual(cb_roundtrip(5), 22)
        finally:
            get_unified_registry().clear_link_objects()
            for obj in saved:
                get_unified_registry().add_link_object(obj)


if __name__ == "__main__":
    unittest.main()
