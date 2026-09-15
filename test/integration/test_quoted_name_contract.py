#!/usr/bin/env python3
"""
Quoted type-name resolution: the behavioral contract, pinned by tests.

A quoted name in type position (``ptr["A"]``, ``array["A", 4]``,
``func[..., "A"]``, ``static["A"]``, ``ob_base: "A"``) resolves at compile
time through these layers, in order:

  R1. the compiling module's visible namespace -- a plain ``A = SomeType``
      assignment is a typedef-style alias (identity binding: ``A is
      SomeType``), no registration needed;
  R2. the session forward-ref registry (``mark_type_defined``), the
      cross-module channel;
  R3. otherwise the string stays a lazy forward reference: behind a pointer
      it materializes as an opaque incomplete type (C: ``struct S *`` with
      no visible definition); anywhere a layout is required, resolution
      failure is a loud error.

Refinements:

  R4.  "Visible" means the decoration-time snapshot; names added to the
       module AFTER decoration still resolve, via the module's live globals
       (compilation is lazy: by the first call the whole module has run).
  R5.  A name visible at definition time binds eagerly and STAYS bound:
       rebounding the module-level name later does not retarget already
       decorated annotations.
  R6.  The registry is mutable over module-execution time: entries
       retargeted between decoration and flush resolve to the FINAL
       registration, consistently for annotations and body casts.
  R7.  The visible namespace wins over the registry when both bind a name.
  R8.  A name bound to a non-type object is not adopted as an alias.
  R9.  Resolution recurses into nested pointers (``ptr[ptr["A"]]``).
  R10. Quoted components of func[...] types resolve the same way.
  R11. Type qualifiers with quoted inner types (``static["Box"]``) resolve
       the same way.
  R12. Value positions never defer: an unresolvable type in a value position
       (array element, by-value field/return) is a compile-time error.

Where each rule is pinned:

  R1  test_scalar_alias / test_struct_alias / test_alias_chain
      (test_type_alias_quoted.py)
  R2  test_registry_only_type (test_type_alias_quoted.py)
  R3  test_opaque_deref_through_cast / test_shared_opaque_tag_aliases
      (test_type_alias_quoted.py), test_incomplete_ptr
      (test_extended_builtins.py)
  R4  test_late_alias / test_lazy_plain_alias_target
      (test_type_alias_quoted.py)
  R5  test_def_time_binding_is_sticky (this file)
  R6  test_registry_retarget_between_decoration_and_flush
      (test_type_alias_quoted.py)
  R7  test_visible_namespace_beats_registry (this file)
  R8  test_nontype_binding_not_adopted (test_type_alias_quoted.py)
  R9  test_nested_ptr_alias_compat (test_type_alias_quoted.py)
  R10 test_func_quoted_component (test_type_alias_quoted.py)
  R11 test_quoted_static_member (test_extended_builtins.py)
  R12 test_missing_array_element_type (test_type_alias_quoted.py)

Alias identity (``A is B`` for ``A = B``) is pinned by
test_alias_identity below.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc import i32, i64, ptr, compile, void, mark_type_defined


# ============================================================
# R7: the visible namespace beats the registry
# ============================================================

@compile
class RegSideStruct:
    v: i64


# Registry entry claimed first (cross-module definer shape); the module-level
# alias below shadows it for THIS module's compiled code.
mark_type_defined("ContractShadow", RegSideStruct)
ContractShadow = i32


@compile(suffix="qc_shadow")
def shadow_read(p: ptr["ContractShadow"]) -> i64:
    return p[0] + i64(1)


@compile(suffix="qc_shadow")
def call_shadow_read() -> i64:
    x: i32 = 41
    return i64(shadow_read(ptr(x)))  # 42; i32 (namespace), not RegSideStruct


# ============================================================
# R5: a name visible at definition time binds eagerly and stays bound
# ============================================================

StickyAlias = i32


@compile(suffix="qc_sticky")
def sticky_read(p: ptr["StickyAlias"]) -> i32:
    return p[0]


# Rebound after the def: already-decorated annotations keep i32.
StickyAlias = i64


@compile(suffix="qc_sticky")
def call_sticky_read() -> i32:
    x: i32 = 42
    return sticky_read(ptr(x))  # 42 via ptr[i32]; a lazy re-resolve to i64
    # would reject this argument


class TestQuotedNameContract(unittest.TestCase):
    def test_visible_namespace_beats_registry(self):
        self.assertEqual(call_shadow_read(), 42)

    def test_def_time_binding_is_sticky(self):
        self.assertEqual(call_sticky_read(), 42)

    def test_alias_identity(self):
        # A plain assignment alias IS the target object; no wrapper, no copy.
        from pythoc import i64 as _i64
        AliasOfI64 = _i64
        self.assertIs(AliasOfI64, _i64)


if __name__ == "__main__":
    unittest.main()
