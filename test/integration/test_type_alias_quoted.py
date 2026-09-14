#!/usr/bin/env python3
"""
Quoted type-name resolution and typedef-style aliases.

A quoted name in type position (``ptr["A"]``) resolves in three layers:
1. the visible module namespace -- a plain ``A = SomeType`` assignment
   acts as a typedef-style alias, no registration needed.  Because
   compilation is lazy, "visible" means defined anywhere at module level
   before the first native call, including after the function definition;
2. the session forward-ref registry (cross-module / generated code);
3. otherwise the string stays a lazy forward reference, resolved at IR
   materialization -- or materialized as an opaque incomplete type behind
   a pointer, which C explicitly allows.

Note: @compile wrappers are defined at module level because pythoc requires
all @compile definitions to precede the first native call from this module.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc import i8, i32, i64, ptr, array, compile, void, func
from pythoc.logger import set_raise_on_error

# Test mode: compile errors raise exceptions instead of sys.exit(1).
set_raise_on_error(True)

try:
    from type_alias_quoted_lib import (
        LibPoint, LibPointAlias, lib_sum_point,
    )
    import type_alias_quoted_lib as _lib  # noqa: F401  (registers names)
except ImportError:
    from test.integration.type_alias_quoted_lib import (
        LibPoint, LibPointAlias, lib_sum_point,
    )
    import test.integration.type_alias_quoted_lib as _lib  # noqa: F401
# RegOnlyPoint is deliberately NOT imported by name: the cross-TU registry
# test reaches it only through the session forward-ref registry.


# ============================================================
# Same-module aliases (plain assignment, typedef-style)
# ============================================================

AliasI64 = i64


@compile
def alias_scalar_read(p: ptr["AliasI64"]) -> i64:
    """Quoted scalar alias in a parameter annotation."""
    return p[0] + i64(1)


@compile
def call_alias_scalar() -> i64:
    x: i64 = 41
    return alias_scalar_read(ptr(x))  # 42


@compile
class Point:
    x: i32
    y: i32


PointAlias = Point


@compile
def alias_struct_sum(p: ptr["PointAlias"]) -> i32:
    """Quoted struct alias: field access through the alias name."""
    return p.x + p.y


@compile
def call_alias_struct() -> i32:
    pt: Point
    pt.x = 30
    pt.y = 12
    return alias_struct_sum(ptr(pt))  # 42


@compile
def alias_array_sum() -> i32:
    """Quoted alias as array element type in a local annotation."""
    arr: array["PointAlias", 2]
    arr[0].x = 1
    arr[0].y = 2
    arr[1].x = 3
    arr[1].y = 4
    return arr[0].x + arr[0].y + arr[1].x + arr[1].y  # 10


ChainA = i64
ChainB = ChainA


@compile
def alias_chain_read(p: ptr["ChainB"]) -> i64:
    """Alias-to-alias: B = A collapses to the same object."""
    return p[0] * i64(2)


@compile
def call_alias_chain() -> i64:
    x: i64 = 21
    return alias_chain_read(ptr(x))  # 42


@compile(suffix="late_alias")
def late_alias_read(p: ptr["LateAlias"]) -> i32:
    """The alias is assigned after this function is defined.  Compilation
    is lazy, so by the first call the whole module has executed and the
    name resolves through the module's live globals."""
    return p[0] + i32(5)


LateAlias = i32


@compile(suffix="late_alias")
def call_late_alias() -> i32:
    v: i32 = 37
    return late_alias_read(ptr(v))  # 42


# ============================================================
# A = ptr["B"] -- aliases whose target is itself a quoted name
# ============================================================

EagerPtr = ptr["Point"]  # Point is visible above: binds eagerly


@compile
def eager_ptr_alias(p: EagerPtr) -> i32:
    return p[0].y - p[0].x


@compile
def call_eager_ptr_alias() -> i32:
    pt: Point
    pt.x = 2
    pt.y = 44
    return eager_ptr_alias(ptr(pt))  # 42


LazyRegPtr = ptr["LazyRegPoint"]  # LazyRegPoint is defined below


@compile
def lazy_reg_read(p: LazyRegPtr) -> i32:
    return p[0].v * i32(2)


@compile
class LazyRegPoint:
    v: i32


@compile
def call_lazy_reg() -> i32:
    lr: LazyRegPoint
    lr.v = 21
    return lazy_reg_read(ptr(lr))  # 42


LazyAliasPtr = ptr["LaterPlainAlias"]  # target is a plain alias defined below


@compile(suffix="lazy_alias_ptr")
def lazy_alias_ptr_read(p: LazyAliasPtr) -> i64:
    """Alias to a quoted name whose target is itself a plain alias defined
    later in the module; resolved at compile time via live globals."""
    return p[0] + i64(2)


LaterPlainAlias = i64


@compile(suffix="lazy_alias_ptr")
def call_lazy_alias_ptr() -> i64:
    x: i64 = 40
    return lazy_alias_ptr_read(ptr(x))  # 42


# A func-type alias whose component is a quoted forward reference to a class
# defined below -- the _Py_iteritemfunc shape: the string must survive to
# materialization and resolve through the registry (the class marks itself
# at decoration).
FuncCbAlias = func["FuncCbResult", i64, i64]


@compile
class FuncCbResult:
    v: i64


@compile(suffix="func_quoted")
def func_quoted_param(cb: FuncCbAlias) -> i64:
    return i64(42)


@compile
class NestNode:
    v: i64
    # Field type quotes the alias BEFORE the alias assignment below exists:
    # the pointee stays a lazy string at class-decoration time.
    left: ptr["NestNodeAlias"]


NestNodeAlias = NestNode


@compile(suffix="nested_ptr_alias")
def nested_ptr_walk(root: ptr[ptr["NestNodeAlias"]]) -> i64:
    """ptr[ptr["Alias"]] with the alias bound to a class whose own field
    carries the lazy string: getptr of that field is ptr[ptr[<string>]]
    and must coerce to the parameter's ptr[ptr[<class>]]."""
    if root[0] != ptr[void](0):
        root = ptr(root[0].left)
    return i64(7)


@compile(suffix="nested_ptr_alias")
def call_nested_ptr_walk() -> i64:
    n: NestNode
    n.v = 35
    n.left = ptr[void](0)
    r: ptr[NestNode] = ptr(n)
    return nested_ptr_walk(ptr(r)) + i64(n.v)  # 7 + 35 = 42


# ============================================================
# Registry re-registration between decoration and flush
# ============================================================

@compile
class GateAliasPlaceholder:
    _pad: i8


# A placeholder occupies the registry entry first (generated-code pattern:
# a typedef-forwarding module registers its placeholder before the module
# with the real layout executes).
from pythoc import mark_type_defined as _mark_type_defined
_mark_type_defined("GateAliasTarget", GateAliasPlaceholder)


@compile(suffix="gate_retime")
def gate_retime_fn(op: ptr[void]) -> ptr["GateAliasTarget"]:
    # Written while the registry holds the placeholder: the annotation must
    # stay a lazy string and resolve at compile time, NOT bake the
    # placeholder in at decoration time -- the registry entry is retargeted
    # to the real class below, and the annotation and the body cast must
    # agree on the final answer.
    return ptr["GateAliasTarget"](op)


@compile
class GateAliasReal:
    v: i64


# The module-level name is rebound and the registry entry retargeted to the
# real class (the defining module's tail mark).  Only now does the name
# become visible in this module's namespace.
GateAliasTarget = GateAliasReal
_mark_type_defined("GateAliasTarget", GateAliasReal)


@compile(suffix="gate_retime")
def call_gate_retime() -> i64:
    r: GateAliasReal
    r.v = 42
    out: ptr["GateAliasTarget"] = gate_retime_fn(ptr[void](ptr(r)))
    return out.v  # 42; field access proves the real layout won


# ============================================================
# Opaque / incomplete types behind aliases
# ============================================================

OpaqueHandle = ptr["AliasTestOpaqueTag"]  # tag intentionally never defined


@compile
def opaque_alias_deref(p: OpaqueHandle) -> i64:
    """Opaque pointer alias: usable through casts, like C's struct S*."""
    q: ptr[i64] = ptr[i64](ptr[void](p))
    return q[0] + i64(1)


@compile
def call_opaque_alias() -> i64:
    x: i64 = 41
    h: OpaqueHandle = OpaqueHandle(ptr[void](ptr(x)))
    return opaque_alias_deref(h)  # 42


OpaqueA = ptr["AliasTestSharedTag"]
OpaqueB = ptr["AliasTestSharedTag"]


@compile
def opaque_shared_take_a(p: OpaqueA) -> i64:
    q: ptr[i64] = ptr[i64](ptr[void](p))
    return q[0] + i64(2)


@compile
def call_opaque_shared() -> i64:
    """Two aliases to the same incomplete tag are interchangeable."""
    x: i64 = 40
    b: OpaqueB = OpaqueB(ptr[void](ptr(x)))
    return opaque_shared_take_a(b)  # 42


# ============================================================
# Cross-TU: types reached from another module
# ============================================================

@compile
def cross_visible_sum(p: ptr["LibPointAlias"]) -> i32:
    """LibPointAlias is imported into this module: visible-namespace path."""
    return p.x * i32(10) + p.y


@compile
def call_cross_visible() -> i32:
    pt: LibPoint
    pt.x = 4
    pt.y = 2
    return cross_visible_sum(ptr(pt))  # 42


@compile
def cross_registry_field() -> i32:
    """RegOnlyPoint is never imported by name; the quoted reference resolves
    through the session registry populated by the lib module's execution."""
    arr: array["RegOnlyPoint", 1]
    arr[0].v = 40
    return arr[0].v + i32(2)  # 42


@compile
def call_lib_lazy_alias() -> i32:
    """The lib module's LibPointPtr = ptr["LibPoint"] is the parameter type
    of lib_sum_point; calling it from here exercises the alias's eager
    binding in the lib module."""
    pt: LibPoint
    pt.x = 39
    pt.y = 3
    return lib_sum_point(ptr(pt))  # 42


# ============================================================
# Error cases (each in its own suffix group so a failing compile does
# not poison this module's default object file)
# ============================================================

@compile(suffix="bad_array_missing")
def bad_array_missing() -> i32:
    arr: array["AliasTestTotallyMissing", 2]
    return i32(0)


AliasTestNotAType = 42


@compile(suffix="bad_nontype_deref")
def bad_nontype_deref() -> i32:
    x: i32 = 1
    p: ptr["AliasTestNotAType"] = ptr["AliasTestNotAType"](ptr[void](ptr(x)))
    return p[0]


# ============================================================
# Test cases
# ============================================================

class TestSameModuleAlias(unittest.TestCase):
    def test_scalar_alias(self):
        self.assertEqual(call_alias_scalar(), 42)

    def test_struct_alias(self):
        self.assertEqual(call_alias_struct(), 42)

    def test_array_of_alias(self):
        self.assertEqual(alias_array_sum(), 10)

    def test_alias_chain(self):
        self.assertEqual(call_alias_chain(), 42)

    def test_late_alias(self):
        # Alias assigned after the function definition: resolves at compile
        # time through the module's live globals (lazy compilation).
        self.assertEqual(call_late_alias(), 42)


class TestPtrAliasCornerCases(unittest.TestCase):
    def test_eager_ptr_alias(self):
        self.assertEqual(call_eager_ptr_alias(), 42)

    def test_lazy_registry_target(self):
        self.assertEqual(call_lazy_reg(), 42)

    def test_lazy_plain_alias_target(self):
        self.assertEqual(call_lazy_alias_ptr(), 42)

    def test_nested_ptr_alias_compat(self):
        self.assertEqual(call_nested_ptr_walk(), 42)

    def test_registry_retarget_between_decoration_and_flush(self):
        self.assertEqual(call_gate_retime(), 42)

    def test_func_quoted_component(self):
        # Compiling a function whose parameter is a func[...] type with a
        # quoted forward-ref component must materialize through the registry
        # (pre-fix this failed the flush with "Unknown function type
        # component").  Compile-only: ctypes cannot synthesize a null
        # function-pointer argument for a direct call.
        from pythoc.decorators.compile import flush_all_pending_outputs
        flush_all_pending_outputs()


class TestOpaqueAlias(unittest.TestCase):
    def test_opaque_deref_through_cast(self):
        self.assertEqual(call_opaque_alias(), 42)

    def test_shared_opaque_tag_aliases(self):
        self.assertEqual(call_opaque_shared(), 42)


class TestCrossTU(unittest.TestCase):
    def test_visible_imported_alias(self):
        self.assertEqual(call_cross_visible(), 42)

    def test_registry_only_type(self):
        self.assertEqual(cross_registry_field(), 42)

    def test_lib_own_lazy_alias(self):
        self.assertEqual(call_lib_lazy_alias(), 42)


class TestAliasErrors(unittest.TestCase):
    def test_missing_array_element_type(self):
        with self.assertRaises(Exception) as ctx:
            bad_array_missing()
        self.assertIn("AliasTestTotallyMissing", str(ctx.exception))

    def test_nontype_binding_not_adopted(self):
        # A quoted name bound to a non-type is not adopted as an alias;
        # the pointer stays incomplete and dereferencing it is an error.
        with self.assertRaises(Exception) as ctx:
            bad_nontype_deref()
        self.assertIn("AliasTestNotAType", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
