from __future__ import annotations

import importlib
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


class TestCimportBackendSelection(unittest.TestCase):
    def setUp(self):
        self._saved_env = {
            key: os.environ.get(key)
            for key in ("PC_CIMPORT_BACKEND", "PC_ENABLE_CLANG_CIMPORT")
        }

    def tearDown(self):
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_default_backend_is_auto(self):
        ci = importlib.import_module("pythoc.cimport")

        self.assertEqual(ci._normalize_cimport_backend(None), "auto")

    def test_env_backend_overrides_legacy_enable_flag(self):
        ci = importlib.import_module("pythoc.cimport")

        os.environ["PC_ENABLE_CLANG_CIMPORT"] = "1"
        os.environ["PC_CIMPORT_BACKEND"] = "native"

        self.assertEqual(ci._normalize_cimport_backend(None), "native")

    def test_legacy_enable_flag_requests_clang(self):
        ci = importlib.import_module("pythoc.cimport")

        os.environ.pop("PC_CIMPORT_BACKEND", None)
        os.environ["PC_ENABLE_CLANG_CIMPORT"] = "1"

        self.assertEqual(ci._normalize_cimport_backend(None), "clang")

    def test_auto_selects_clang_when_available(self):
        ci = importlib.import_module("pythoc.cimport")

        old = ci._clang_backend_available
        try:
            ci._clang_backend_available = lambda: True
            self.assertEqual(ci._select_cimport_backend("auto"), "clang")
        finally:
            ci._clang_backend_available = old

    def test_invalid_backend_is_rejected(self):
        ci = importlib.import_module("pythoc.cimport")

        with self.assertRaises(ValueError):
            ci._normalize_cimport_backend("bogus")


class TestCimportNativeBackend(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.header_path = os.path.join(self.temp_dir, "native_backend.h")
        with open(self.header_path, "w", encoding="utf-8") as f:
            f.write("int native_backend_add(int a, int b);\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_explicit_native_backend_generates_bindings(self):
        from pythoc.cimport import cimport

        mod = cimport(self.header_path, lib="c", backend="native")

        self.assertTrue(hasattr(mod, "native_backend_add"))


class TestCimportClangFrontend(unittest.TestCase):
    def setUp(self):
        if not _clang_backend_available():
            self.skipTest("clang/libclang Python bindings are not available")
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(getattr(self, "temp_dir", ""), ignore_errors=True)

    def _write(self, name: str, content: str) -> str:
        path = os.path.join(self.temp_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_clang_frontend_normalized_ir(self):
        from pythoc.cimport_clang import parse_to_ir

        header_path = self._write(
            "clang_frontend_ir.h",
            """
typedef int clang_int_t;

struct ClangPoint {
    int x;
    int y;
};

union ClangValue {
    int i;
    double d;
};

enum ClangColor {
    CLANG_RED = 1,
    CLANG_GREEN,
    CLANG_BLUE = 7
};

extern int clang_global;
int clang_add(clang_int_t a, int b);
""",
        )

        module = parse_to_ir(header_path, cflags=["-x", "c", "-std=c11"])
        decls = {(decl.kind, decl.name): decl for decl in module.declarations}

        self.assertIn(("typedef", "clang_int_t"), decls)
        self.assertEqual(decls[("typedef", "clang_int_t")].type.kind, "primitive")
        self.assertEqual(decls[("typedef", "clang_int_t")].type.name, "i32")

        point = decls[("struct", "ClangPoint")]
        self.assertEqual([field.name for field in point.fields], ["x", "y"])
        self.assertEqual([field.type.name for field in point.fields], ["i32", "i32"])

        value = decls[("union", "ClangValue")]
        self.assertEqual([field.name for field in value.fields], ["i", "d"])
        self.assertEqual([field.type.name for field in value.fields], ["i32", "f64"])

        color = decls[("enum", "ClangColor")]
        self.assertEqual(
            [(value.name, value.value) for value in color.values],
            [("CLANG_RED", 1), ("CLANG_GREEN", 2), ("CLANG_BLUE", 7)],
        )

        global_var = decls[("var", "clang_global")]
        self.assertEqual(global_var.type.name, "i32")

        func = decls[("function", "clang_add")]
        self.assertEqual(func.type.return_type.name, "i32")
        self.assertEqual([param.name for param in func.type.params], ["a", "b"])
        self.assertEqual(func.type.params[0].type.kind, "typedef")
        self.assertEqual(func.type.params[0].type.name, "clang_int_t")
        self.assertEqual(func.type.params[1].type.name, "i32")

    def test_clang_backend_cimport_imports_generated_module(self):
        from pythoc.cimport import cimport

        header_path = self._write(
            "clang_backend_module.h",
            """
typedef int clang_mod_int_t;

struct ClangModulePoint {
    int x;
    int y;
};

enum ClangModuleColor {
    CLANG_MODULE_RED = 4,
    CLANG_MODULE_BLUE = 8
};

int clang_module_add(clang_mod_int_t a, int b);
""",
        )

        mod = cimport(
            header_path,
            lib="c",
            backend="clang",
            cflags=["-x", "c", "-std=c11"],
            prefix="clang_backend_module",
        )

        self.assertTrue(hasattr(mod, "clang_mod_int_t"))
        self.assertTrue(hasattr(mod, "ClangModulePoint"))
        self.assertTrue(hasattr(mod, "ClangModuleColor"))
        self.assertTrue(hasattr(mod, "clang_module_add"))

    def test_clang_backend_c_source_end_to_end_call(self):
        try:
            from pythoc.utils.cc_utils import find_available_cc

            find_available_cc()
        except RuntimeError as exc:
            self.skipTest(str(exc))

        from pythoc import compile, i32
        from pythoc.cimport import cimport
        from pythoc.registry import get_unified_registry

        saved_link_objects = list(get_unified_registry().get_link_objects())
        source_path = self._write(
            "clang_backend_e2e.c",
            """
int clang_backend_e2e_add(int a, int b) {
    return a + b;
}
""",
        )

        try:
            mod = cimport(
                source_path,
                backend="clang",
                compile_sources=True,
                cflags=["-x", "c", "-std=c11"],
                prefix="clang_backend_e2e",
            )
            clang_backend_e2e_add = mod.clang_backend_e2e_add

            @compile
            def clang_backend_e2e_call(a: i32, b: i32) -> i32:
                return clang_backend_e2e_add(a, b)

            self.assertEqual(clang_backend_e2e_call(12, 30), 42)
        finally:
            get_unified_registry().clear_link_objects()
            for obj in saved_link_objects:
                get_unified_registry().add_link_object(obj)
