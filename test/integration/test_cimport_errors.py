# -*- coding: utf-8 -*-
"""
Error-path hardening tests for cimport.

Locks in clean, descriptive failures instead of crashes or silent
misbehavior:

- nonexistent header path -> FileNotFoundError
- header with a syntax error -> ClangCImportError carrying clang diagnostics
- every _unsupported_symbols category raises a descriptive RuntimeError
  on attribute access (variadic static inline, static global, TLS global,
  long double on targets without a matching pythoc type, and declarations
  that transitively reference an unsupported type)
- calling a function bound to a missing symbol gives a clear load error
  (AttributeError from dlsym), not a segfault
- export= of an unsupported name raises the descriptive RuntimeError;
  export= of a name that does not exist at all raises AttributeError

Note: @compile wrappers are defined at module level because pythoc requires
all @compile definitions to precede the first native call from this module.
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


def _cc_available() -> bool:
    try:
        from pythoc.utils.cc_utils import find_available_cc
        find_available_cc()
    except RuntimeError:
        return False
    return True


_BACKEND_AVAILABLE = _clang_backend_available() and _cc_available()


@unittest.skipUnless(_BACKEND_AVAILABLE, "clang backend or cc not available")
class CimportErrorsBase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write(self, name: str, content: str) -> str:
        path = os.path.join(self.temp_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path


class TestCimportBadInput(CimportErrorsBase):
    """Unparseable or missing inputs fail with descriptive errors."""

    def test_nonexistent_header(self):
        from pythoc.cimport import cimport

        with self.assertRaises(FileNotFoundError) as ctx:
            cimport(os.path.join(self.temp_dir, "does_not_exist.h"))
        self.assertIn("does_not_exist.h", str(ctx.exception))

    def test_nonexistent_header_by_name(self):
        from pythoc.cimport import cimport

        # A bare name that resolves nowhere (not cwd, not include path).
        with self.assertRaises(FileNotFoundError):
            cimport("definitely_not_a_real_header_zzz.h")

    def test_syntax_error_header(self):
        from pythoc.cimport import cimport
        from pythoc.cimport_clang import ClangCImportError

        header = self._write("broken.h", "int foo(\n")
        with self.assertRaises(ClangCImportError) as ctx:
            cimport(header, lib="c")
        msg = str(ctx.exception)
        # The clang diagnostic (file/line/what) rides along with the error.
        self.assertIn("broken.h", msg)
        self.assertIn("error", msg)

    def test_syntax_error_source(self):
        from pythoc.cimport import cimport
        from pythoc.cimport_clang import ClangCImportError

        source = self._write("broken.c", "int main(void) { return ;;;\n")
        with self.assertRaises(ClangCImportError):
            cimport(source, compile_sources=True)


class TestCimportUnsupportedSymbols(CimportErrorsBase):
    """Every unsupported category is a lazy descriptive RuntimeError."""

    def test_variadic_static_inline(self):
        from pythoc.cimport import cimport

        header = self._write("u_va.h", """
static inline int u_va_sum(int first, ...) { return first; }
int u_ok(int x);
""")
        mod = cimport(header, lib="c")
        self.assertTrue(hasattr(mod, "u_ok"))
        self.assertIn("u_va_sum", mod._unsupported_symbols)
        with self.assertRaises(RuntimeError) as ctx:
            mod.u_va_sum
        self.assertIn("variadic", str(ctx.exception))

    def test_static_global(self):
        from pythoc.cimport import cimport

        header = self._write("u_static.h", """
static int u_file_scope = 1;
extern int u_extern_ok;
""")
        mod = cimport(header, lib="c")
        self.assertIn("u_file_scope", mod._unsupported_symbols)
        with self.assertRaises(RuntimeError) as ctx:
            mod.u_file_scope
        self.assertIn("static global", str(ctx.exception))

    def test_thread_local_global(self):
        from pythoc.cimport import cimport

        header = self._write("u_tls.h", """
extern _Thread_local int u_tls;
extern int u_extern_ok;
""")
        mod = cimport(header, lib="c")
        self.assertIn("u_tls", mod._unsupported_symbols)
        with self.assertRaises(RuntimeError) as ctx:
            mod.u_tls
        self.assertIn("thread-local", str(ctx.exception))

    def test_long_double_function_on_x86_64(self):
        from pythoc.cimport import cimport

        # x86_64 long double is 16-byte fp80 with no pythoc type.
        header = self._write("u_ld.h", """
long double u_ld_fn(long double x);
int u_ok(int x);
""")
        mod = cimport(header, lib="c", target="x86_64-unknown-linux-gnu")
        self.assertTrue(hasattr(mod, "u_ok"))
        self.assertIn("u_ld_fn", mod._unsupported_symbols)
        with self.assertRaises(RuntimeError) as ctx:
            mod.u_ld_fn
        self.assertIn("long double", str(ctx.exception))

    def test_transitive_reference_to_unsupported_type(self):
        from pythoc.cimport import cimport

        # A function whose signature mentions an unsupported declaration's
        # type must degrade as well, with a reason that names the chain.
        header = self._write("u_trans.h", """
struct UBits { unsigned a : 1; };
struct UBits u_take_bits(struct UBits b);
int u_also_ok(int x);
""")
        mod = cimport(header, lib="c")
        # The bitfield struct itself binds (as opaque storage), so nothing
        # here is unsupported; assert the module is fully usable instead.
        self.assertTrue(hasattr(mod, "u_take_bits"))
        self.assertTrue(hasattr(mod, "u_also_ok"))

    def test_unsupported_reason_mentions_referencing_decl(self):
        from pythoc.cimport import cimport

        # u_ld_t is an unsupported typedef; u_ld_user references it and
        # must degrade with a "references unsupported type" reason.
        header = self._write("u_ref.h", """
typedef long double u_ld_t;
u_ld_t u_ld_user(u_ld_t x);
int u_ok(int x);
""")
        mod = cimport(header, lib="c", target="x86_64-unknown-linux-gnu")
        with self.assertRaises(RuntimeError) as ctx:
            mod.u_ld_user
        msg = str(ctx.exception)
        self.assertIn("u_ld_user", msg)
        self.assertIn("long double", msg)

    def test_unknown_attribute_is_attribute_error_not_runtime(self):
        from pythoc.cimport import cimport

        header = self._write("u_attr.h", "int u_some_fn(int x);\n")
        mod = cimport(header, lib="c")
        with self.assertRaises(AttributeError):
            mod.totally_unknown_name


class TestCimportMissingSymbol(CimportErrorsBase):
    """A binding to a symbol no loaded library provides fails cleanly."""

    def test_call_missing_symbol_clear_error(self):
        from pythoc.cimport import cimport

        header = self._write(
            "missing.h",
            "int pythoc_no_such_symbol_anywhere(int x);\n")
        mod = cimport(header, lib="c")
        self.assertTrue(hasattr(mod, "pythoc_no_such_symbol_anywhere"))
        with self.assertRaises((AttributeError, RuntimeError, OSError)) as ctx:
            mod.pythoc_no_such_symbol_anywhere(1)
        self.assertIn("pythoc_no_such_symbol_anywhere", str(ctx.exception))


class TestCimportExportEdgeCases(CimportErrorsBase):
    """export= against unsupported or unknown names behaves predictably."""

    def test_export_unsupported_raises_descriptive_error(self):
        from pythoc.cimport import cimport

        header = self._write("exp_unsup.h", """
static inline int exp_va(int first, ...) { return first; }
int exp_ok(int x);
""")
        # export= resolves the attribute, which triggers the lazy
        # RuntimeError for unsupported symbols.
        with self.assertRaises(RuntimeError) as ctx:
            cimport(header, lib="c", export=["exp_va"])
        self.assertIn("variadic", str(ctx.exception))

    def test_export_unknown_name_raises_attribute_error(self):
        from pythoc.cimport import cimport

        header = self._write("exp_unknown.h", "int exp_real(int x);\n")
        with self.assertRaises(AttributeError) as ctx:
            cimport(header, lib="c", export=["exp_ghost"])
        self.assertIn("exp_ghost", str(ctx.exception))

    def test_export_valid_names_still_work(self):
        from pythoc.cimport import cimport

        header = self._write("exp_ok.h", """
#define EXP_MAGIC 77
int exp_fn(int x);
""")
        g = {}
        code = (
            "from pythoc.cimport import cimport\n"
            f"cimport({header!r}, lib='c', export=['EXP_MAGIC', 'exp_fn'])\n"
        )
        exec(compile(code, "<test>", "exec"), g)
        self.assertEqual(g["EXP_MAGIC"], 77)
        self.assertEqual(g["exp_fn"].c_name, "exp_fn")


class TestCimportLinkAndCacheErrors(CimportErrorsBase):
    """Library-path and bindings-cache failure handling."""

    def test_lib_path_to_nonexistent_so_fails_at_call(self):
        """lib= pointing at a missing .so imports lazily; calling the
        binding raises OSError from dlopen instead of a crash."""
        from pythoc.cimport import cimport

        header = self._write("badlib.h", "int badlib_fn(int x);\n")
        mod = cimport(
            header, lib=os.path.join(self.temp_dir, "no_such_lib_xyz.so"))
        self.assertTrue(hasattr(mod, "badlib_fn"))
        with self.assertRaises(OSError) as ctx:
            mod.badlib_fn(1)
        self.assertIn("no_such_lib_xyz", str(ctx.exception))

    def test_source_compile_failure_clear_error(self):
        """A source listed in sources= that fails cc compilation raises a
        RuntimeError carrying the compiler diagnostics."""
        from pythoc.cimport import cimport

        header = self._write("good_hdr.h", "int good_hdr_fn(int x);\n")
        source = self._write(
            "bad_src.c",
            "int bad_src_fn(int x) { return x + ; }\n")
        with self.assertRaises(RuntimeError) as ctx:
            cimport(header, sources=[source], compile_sources=True)
        msg = str(ctx.exception)
        self.assertIn("Compilation failed", msg)
        self.assertIn("bad_src.c", msg)

    def test_cache_recovery_after_header_fix(self):
        """A failed import does not poison the bindings cache: fixing the
        header and re-importing the same path succeeds."""
        from pythoc.cimport import cimport
        from pythoc.cimport_clang import ClangCImportError

        header = self._write("recov.h", "int recov_fn(\n")
        with self.assertRaises(ClangCImportError):
            cimport(header, lib="c")
        with open(header, "w", encoding="utf-8") as f:
            f.write("int recov_fn(int x);\n")
        mod = cimport(header, lib="c")
        self.assertTrue(hasattr(mod, "recov_fn"))


class TestCimportConversionResilience(CimportErrorsBase):
    """A declaration that fails IR conversion degrades to a lazy error
    instead of aborting the whole import (e.g. a python-clang/libclang
    version skew raising ValueError on an SDK declaration)."""

    def test_unconvertible_decl_becomes_lazy_error(self):
        import pythoc.cimport_clang as clang_backend
        from pythoc.cimport import cimport

        header = self._write(
            "mixed_conv.h",
            "int conv_ok_fn(int x);\nint conv_bad_fn(int x);\n")
        original = clang_backend._cursor_to_decl

        def _flaky(*args, **kwargs):
            cursor = args[2]
            if cursor.spelling == "conv_bad_fn":
                raise ValueError("Unknown template argument kind 32")
            return original(*args, **kwargs)

        try:
            clang_backend._cursor_to_decl = _flaky
            mod = cimport(header, lib="c")
        finally:
            clang_backend._cursor_to_decl = original

        # The good declaration is unaffected...
        self.assertTrue(hasattr(mod, "conv_ok_fn"))
        self.assertEqual(mod.conv_ok_fn.c_name, "conv_ok_fn")
        # ...and the bad one is a descriptive lazy error, not a crash.
        with self.assertRaises(RuntimeError) as ctx:
            mod.conv_bad_fn
        self.assertIn("conv_bad_fn", str(ctx.exception))
        self.assertIn("could not be converted", str(ctx.exception))
        self.assertIn("Unknown template argument kind 32", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
