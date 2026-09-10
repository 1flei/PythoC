# -*- coding: utf-8 -*-
"""
Layout-fidelity and namespace tests for cimport.

Covers:
- Structs whose C layout pythoc's natural struct layout cannot reproduce
  (packed, _Alignas over-alignment, flexible array members) degrade to
  opaque storage with the correct size.
- Correctly-laid-out structs keep field bindings (no over-degradation),
  with offsets checked against the C layout.
- C dual namespaces: a tag type colliding with a typedef/function/enum
  constant keeps the plain name for the ordinary identifier; the tag is
  emitted as struct_Foo / union_Foo / enum_Foo.
- __asm__ labels: the binding keeps the C name while the linked symbol
  uses the asm label (glibc __REDIRECT scenario).
- Cross-import type identity: the same header imported twice yields
  structurally interchangeable types.
- K&R no-proto declarations `int f();` bind as zero-arg.

Note: @compile wrappers are defined at module level because pythoc requires
all @compile definitions to precede the first native call from this module.
"""
from __future__ import annotations

import inspect
import os
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

_fixture_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'build', 'test', 'cimport_layout'))
os.makedirs(_fixture_dir, exist_ok=True)


def _write_fixture(name: str, content: str) -> str:
    path = os.path.join(_fixture_dir, name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path


if _BACKEND_AVAILABLE:
    from pythoc import offsetof
    from pythoc.cimport import cimport

    # =================================================================
    # Item 1: layout-mismatch structs degrade to opaque
    # =================================================================
    _layout_header = _write_fixture('layout.h', '''
#pragma pack(push, 1)
struct Packed { int a; char b; };
#pragma pack(pop)
struct Normal { int a; double b; char c; };
struct Inner { int x; int y; };
struct Outer { char tag; struct Inner in_; int arr[3]; };
struct OverAligned { char c; _Alignas(16) int x; };
struct Flex { int n; char data[]; };
union UAligned { char c; _Alignas(16) int x; };
union UNormal { int i; double d; };
struct Bits { unsigned a : 1; };
struct OuterBits { char c; struct Bits bits; int tail; };
struct WithEnum { enum Tag { TAG_A, TAG_B } e; int x; };
void packed_fill(struct Packed *p, int a, char b);
int packed_sum(const struct Packed *p);
int normal_sum(const struct Normal *p);
int outer_sum(const struct Outer *p);
''')
    _layout_source = _write_fixture('layout.c', '''
#include "layout.h"
void packed_fill(struct Packed *p, int a, char b) { p->a = a; p->b = b; }
int packed_sum(const struct Packed *p) { return p->a + p->b; }
int normal_sum(const struct Normal *p) { return p->a + (int)p->b + p->c; }
int outer_sum(const struct Outer *p) {
    return p->tag + p->in_.x + p->in_.y + p->arr[0] + p->arr[1] + p->arr[2];
}
''')
    _layout_mod = cimport(_layout_header, sources=[_layout_source],
                          compile_sources=True, include_dirs=[_fixture_dir])
    Packed = _layout_mod.Packed
    Normal = _layout_mod.Normal
    Outer = _layout_mod.Outer
    packed_fill = _layout_mod.packed_fill
    packed_sum = _layout_mod.packed_sum
    normal_sum = _layout_mod.normal_sum
    outer_sum = _layout_mod.outer_sum

    @compile
    def packed_roundtrip(a: i32, b: i32) -> i32:
        s: Packed
        packed_fill(ptr(s), a, b)
        return packed_sum(ptr(s))

    @compile
    def normal_roundtrip(a: i32, b: i32, c: i32) -> i32:
        s: Normal
        s.a = a
        s.b = b
        s.c = c
        return normal_sum(ptr(s))

    @compile
    def outer_roundtrip() -> i32:
        s: Outer
        s.tag = 1
        s.in_.x = 2
        s.in_.y = 3
        s.arr[0] = 4
        s.arr[1] = 5
        s.arr[2] = 6
        return outer_sum(ptr(s))

    # =================================================================
    # Item 2: C dual namespaces (tag vs ordinary identifiers)
    # =================================================================
    _ns_header = _write_fixture('namespaces.h', '''
struct Foo { int v; };
typedef int Foo;
struct Bar { int v; };
typedef struct Bar Bar;
struct Dup { int st; };
int Dup(int x);
enum Color { COLOR_RED, COLOR_GREEN };
typedef double Color;
int foo_read(struct Foo *p);
''')
    _ns_source = _write_fixture('namespaces.c', '''
#include "namespaces.h"
int Dup(int x) { return x + 1; }
int foo_read(struct Foo *p) { return p->v; }
''')
    _ns_mod = cimport(_ns_header, sources=[_ns_source],
                      compile_sources=True, include_dirs=[_fixture_dir])
    struct_Foo = _ns_mod.struct_Foo
    foo_read = _ns_mod.foo_read
    Dup_fn = _ns_mod.Dup

    @compile
    def foo_roundtrip(v: i32) -> i32:
        s: struct_Foo
        s.v = v
        return foo_read(ptr(s))

    @compile
    def dup_call(x: i32) -> i32:
        return Dup_fn(x)

    # =================================================================
    # Item 3: __asm__ label symbol names
    # =================================================================
    _asm_header = _write_fixture('asmlabel.h', '''
int asm_fn(void) __asm__("alt_asm_fn");
extern int asm_var __asm__("alt_asm_var");
''')
    _asm_source = _write_fixture('asmlabel.c', '''
int alt_asm_fn(void) { return 77; }
int alt_asm_var = 55;
''')
    _asm_mod = cimport(_asm_header, sources=[_asm_source],
                       compile_sources=True, include_dirs=[_fixture_dir])
    asm_fn = _asm_mod.asm_fn
    asm_var = _asm_mod.asm_var

    @compile
    def call_asm_fn() -> i32:
        return asm_fn()

    @compile
    def read_asm_var() -> i32:
        return asm_var

    # =================================================================
    # Item 4: cross-import type identity
    # =================================================================
    _twice_header = _write_fixture('twice.h', '''
struct Point { int x; int y; };
int point_sum(struct Point *p);
''')
    _twice_source = _write_fixture('twice.c', '''
#include "twice.h"
int point_sum(struct Point *p) { return p->x + p->y; }
''')
    _twice_a = cimport(_twice_header, sources=[_twice_source],
                       compile_sources=True, include_dirs=[_fixture_dir])
    _twice_b = cimport(_twice_header, sources=[_twice_source],
                       compile_sources=True, include_dirs=[_fixture_dir])
    PointA = _twice_a.Point
    point_sum_b = _twice_b.point_sum

    @compile
    def cross_import_sum(x: i32, y: i32) -> i32:
        p: PointA
        p.x = x
        p.y = y
        return point_sum_b(ptr(p))

    # =================================================================
    # Item 5: K&R no-proto functions bind as zero-arg
    # =================================================================
    _knr_header = _write_fixture('knr.h', '''
int knr_noargs();
''')
    _knr_source = _write_fixture('knr.c', '''
int knr_noargs(void) { return 42; }
''')
    _knr_mod = cimport(_knr_header, sources=[_knr_source],
                       compile_sources=True, include_dirs=[_fixture_dir])
    knr_noargs = _knr_mod.knr_noargs

    @compile
    def call_knr() -> i32:
        return knr_noargs()

    @compile
    def normal_offsets() -> i32:
        return (offsetof(Normal, "a") + offsetof(Normal, "b") * 16
                + offsetof(Normal, "c") * 256)

    @compile
    def outer_offsets() -> i32:
        return offsetof(Outer, "in_") + offsetof(Outer, "arr") * 16


@unittest.skipUnless(_BACKEND_AVAILABLE, "clang backend or cc not available")
class TestCimportLayoutDegradation(unittest.TestCase):
    """Layout-mismatch structs degrade to opaque; matching ones keep fields."""

    def test_packed_struct_is_opaque(self):
        self.assertFalse(Packed.has_field("a"))
        self.assertFalse(Packed.has_field("b"))
        self.assertTrue(Packed.has_field("_storage"))
        # C size is 5 (packed); pythoc's natural layout would be 8.
        self.assertEqual(Packed.get_size_bytes(), 5)

    def test_packed_struct_abi_roundtrip(self):
        # C writes/reads the packed layout through a pointer to the opaque
        # storage; a wrong size or layout would corrupt or misread.
        self.assertEqual(packed_roundtrip(7, 3), 10)
        self.assertEqual(packed_roundtrip(-4, 2), -2)

    def test_overaligned_struct_is_opaque(self):
        OverAligned = _layout_mod.OverAligned
        self.assertFalse(OverAligned.has_field("x"))
        self.assertTrue(OverAligned.has_field("_storage"))
        self.assertEqual(OverAligned.get_size_bytes(), 32)

    def test_flexible_array_member_is_opaque(self):
        # char data[] would be emitted as ptr[char] at a wrong (aligned)
        # offset; the whole record degrades instead.
        Flex = _layout_mod.Flex
        self.assertFalse(Flex.has_field("data"))
        self.assertTrue(Flex.has_field("_storage"))

    def test_overaligned_union_is_opaque(self):
        UAligned = _layout_mod.UAligned
        self.assertTrue(UAligned.has_field("_storage"))
        self.assertEqual(UAligned.get_size_bytes(), 16)

    def test_normal_union_keeps_fields(self):
        self.assertTrue(_layout_mod.UNormal.has_field("i"))
        self.assertTrue(_layout_mod.UNormal.has_field("d"))
        self.assertEqual(_layout_mod.UNormal.get_size_bytes(), 8)

    def test_normal_struct_keeps_fields(self):
        self.assertTrue(Normal.has_field("a"))
        self.assertTrue(Normal.has_field("b"))
        self.assertTrue(Normal.has_field("c"))
        self.assertEqual(Normal.get_size_bytes(), 24)

    def test_normal_struct_offsets_match_c(self):
        self.assertEqual(normal_offsets(), 0 + 8 * 16 + 16 * 256)

    def test_normal_struct_abi_roundtrip(self):
        self.assertEqual(normal_roundtrip(1, 2, 3), 6)

    def test_nested_struct_keeps_fields(self):
        self.assertTrue(Outer.has_field("in_"))
        self.assertTrue(Outer.has_field("arr"))
        self.assertEqual(Outer.get_size_bytes(), 24)

    def test_nested_struct_offsets_match_c(self):
        self.assertEqual(outer_offsets(), 4 + 12 * 16)

    def test_nested_struct_abi_roundtrip(self):
        self.assertEqual(outer_roundtrip(), 21)

    def test_enum_field_struct_keeps_fields(self):
        WithEnum = _layout_mod.WithEnum
        self.assertTrue(WithEnum.has_field("e"))
        self.assertTrue(WithEnum.has_field("x"))
        self.assertEqual(WithEnum.get_size_bytes(), 8)

    def test_opaque_inner_with_different_alignment_degrades_outer(self):
        # Bits is opaque storage with alignment 1 (array[u8, 4]) while C
        # aligns it to 4; OuterBits field offsets would shift, so the
        # outer record degrades as well.
        OuterBits = _layout_mod.OuterBits
        self.assertFalse(OuterBits.has_field("bits"))
        self.assertTrue(OuterBits.has_field("_storage"))
        self.assertEqual(OuterBits.get_size_bytes(), 12)


@unittest.skipUnless(_BACKEND_AVAILABLE, "clang backend or cc not available")
class TestCimportNamespaces(unittest.TestCase):
    """C tag namespace vs ordinary identifier namespace."""

    def test_typedef_keeps_plain_name_tag_renamed(self):
        # typedef int Foo; wins the plain name; struct Foo -> struct_Foo.
        from pythoc import i32 as _i32
        self.assertIs(_ns_mod.Foo, _i32)
        self.assertTrue(struct_Foo.has_field("v"))

    def test_typedef_of_own_tag_is_not_a_collision(self):
        # typedef struct Bar Bar; names the same type: plain name kept.
        self.assertTrue(hasattr(_ns_mod, "Bar"))
        self.assertFalse(hasattr(_ns_mod, "struct_Bar"))
        self.assertTrue(_ns_mod.Bar.has_field("v"))

    def test_enum_tag_renamed_on_typedef_collision(self):
        self.assertTrue(hasattr(_ns_mod, "enum_Color"))
        self.assertEqual(_ns_mod.COLOR_RED, 0)
        self.assertEqual(_ns_mod.enum_Color.COLOR_GREEN, 1)

    def test_function_keeps_plain_name_tag_renamed(self):
        self.assertTrue(hasattr(_ns_mod, "struct_Dup"))
        self.assertTrue(struct_Foo.has_field("v"))
        self.assertEqual(dup_call(41), 42)

    def test_renamed_tag_usable_in_signature(self):
        self.assertEqual(foo_roundtrip(42), 42)


@unittest.skipUnless(_BACKEND_AVAILABLE, "clang backend or cc not available")
class TestCimportAsmLabel(unittest.TestCase):
    """__asm__ labels keep the C binding name but link the labeled symbol."""

    def test_function_binding_name_and_symbol(self):
        self.assertTrue(hasattr(_asm_mod, "asm_fn"))
        self.assertFalse(hasattr(_asm_mod, "alt_asm_fn"))
        self.assertEqual(asm_fn.c_name, "alt_asm_fn")

    def test_var_binding_name_and_symbol(self):
        self.assertEqual(asm_var.c_name, "alt_asm_var")

    def test_asm_label_calls_resolve(self):
        self.assertEqual(call_asm_fn(), 77)
        self.assertEqual(read_asm_var(), 55)


@unittest.skipUnless(_BACKEND_AVAILABLE, "clang backend or cc not available")
class TestCimportCrossImportIdentity(unittest.TestCase):
    """The same header imported twice yields interchangeable types."""

    def test_double_import_returns_fresh_modules(self):
        self.assertIsNot(_twice_a, _twice_b)
        self.assertIsNot(_twice_a.Point, _twice_b.Point)

    def test_struct_from_import_a_flows_into_binding_from_import_b(self):
        self.assertEqual(cross_import_sum(30, 12), 42)


@unittest.skipUnless(_BACKEND_AVAILABLE, "clang backend or cc not available")
class TestCimportNoProto(unittest.TestCase):
    """K&R no-proto declarations bind as zero-arg functions."""

    def test_no_proto_binds_zero_arg(self):
        # The generated binding is `def knr_noargs() -> i32` with no params.
        self.assertEqual(knr_noargs.param_types, [])
        sig = inspect.signature(knr_noargs.func)
        self.assertEqual(list(sig.parameters), [])

    def test_no_proto_callable(self):
        self.assertEqual(call_knr(), 42)


if __name__ == "__main__":
    unittest.main()
