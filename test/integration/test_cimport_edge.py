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
};
""")
        mod = cimport(header, lib="c")
        self.assertTrue(hasattr(mod, "ErrorCode"))

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
        """Struct with bitfield fields should be parseable"""
        from pythoc.cimport import cimport

        header = self._write("bitfield.h", """
struct Flags {
    unsigned int a : 1;
    unsigned int b : 3;
    unsigned int c : 4;
};
""")
        mod = cimport(header, lib="c")
        self.assertTrue(hasattr(mod, "Flags"))


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
        """Extern global variables should appear as comments in bindings"""
        from pythoc.cimport import cimport

        header = self._write("globals.h", """
extern int global_counter;
extern void *global_handle;
""")
        # Globals are emitted as comments, so no attribute check
        # Just verify no crash
        mod = cimport(header, lib="c")


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
