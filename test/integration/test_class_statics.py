#!/usr/bin/env python3
"""Integration tests for class-level static members.

A @compile (struct) or @union class may declare ``name: static[T] = seed``
in the class body. The member is not an instance field; it has static
storage duration (C++ static member semantics) and is accessed through the
class only: ``Cls.member``. Seeds must be link-time constants; an omitted
seed zero-initializes (C semantics), matching function-local statics.
"""
from __future__ import annotations

import os
import sys

from pythoc import (
    i32, i64, f64, ptr, array, compile, extern, union, static, const,
    volatile, seq,
)
from pythoc.libc.stdio import printf
from pythoc.logger import set_raise_on_error
from pythoc.build.output_manager import (
    flush_all_pending_outputs, clear_failed_group, get_output_manager,
)

# Enable exception raising for tests that expect to catch exceptions
set_raise_on_error(True)


# ---------------------------------------------------------------------------
# struct: basic read/write and persistence across calls
# ---------------------------------------------------------------------------

@compile
class Counter:
    count: static[i32] = 41


@compile
def bump() -> i32:
    Counter.count = Counter.count + 1
    return Counter.count


@compile
def test_class_static_basic() -> i32:
    if Counter.count != i32(41):
        printf("FAIL: initial value, got %d\n", Counter.count)
        return 1
    Counter.count = 100
    if Counter.count != i32(100):
        printf("FAIL: read-after-write, got %d\n", Counter.count)
        return 1
    printf("class static basic: ok\n")
    return 0


@compile
def test_class_static_persistence() -> i32:
    # bump() observes the value left behind by test_class_static_basic (100),
    # and both calls share one storage.
    first: i32 = bump()
    second: i32 = bump()
    if first != i32(101) or second != i32(102):
        printf("FAIL: persistence, got %d, %d\n", first, second)
        return 1
    printf("class static persistence: ok\n")
    return 0


# ---------------------------------------------------------------------------
# struct: omitted seed zero-initializes; aggregate (array) member
# ---------------------------------------------------------------------------

@compile
class Zeroed:
    n: static[i64]
    buf: static[array[i32, 4]]


@compile
def test_class_static_zero_init() -> i32:
    total: i64 = Zeroed.n
    for i in seq(4):
        total = total + Zeroed.buf[i]
    if total != i64(0):
        printf("FAIL: zero init, got %lld\n", total)
        return 1
    Zeroed.buf[2] = 7
    if Zeroed.buf[2] != i32(7):
        printf("FAIL: array element write, got %d\n", Zeroed.buf[2])
        return 1
    printf("class static zero init: ok\n")
    return 0


# ---------------------------------------------------------------------------
# struct: aggregate seed via sequence literal
# ---------------------------------------------------------------------------

@compile
class WithSeed:
    triple: static[array[i32, 3]] = (10, 20, 30)


@compile
def test_class_static_aggregate_seed() -> i32:
    total: i32 = WithSeed.triple[0] + WithSeed.triple[1] + WithSeed.triple[2]
    if total != i32(60):
        printf("FAIL: aggregate seed, got %d\n", total)
        return 1
    printf("class static aggregate seed: ok\n")
    return 0


# ---------------------------------------------------------------------------
# struct: several functions share one storage
# ---------------------------------------------------------------------------

@compile
class Shared:
    n: static[i32] = 0


@compile
def shared_inc() -> i32:
    Shared.n = Shared.n + 1
    return Shared.n


@compile
def shared_inc_twice() -> i32:
    shared_inc()
    return shared_inc()


@compile
def test_class_static_shared_storage() -> i32:
    shared_inc()
    v: i32 = shared_inc_twice()
    if Shared.n != i32(3) or v != i32(3):
        printf("FAIL: shared storage, got n=%d v=%d\n", Shared.n, v)
        return 1
    printf("class static shared storage: ok\n")
    return 0


# ---------------------------------------------------------------------------
# union: class static alongside union fields
# ---------------------------------------------------------------------------

@union
class Tagged:
    as_i: i32
    as_f: f64
    hits: static[i32] = 3


@compile
def test_union_class_static() -> i32:
    if Tagged.hits != i32(3):
        printf("FAIL: union static seed, got %d\n", Tagged.hits)
        return 1
    Tagged.hits = Tagged.hits + 2
    if Tagged.hits != i32(5):
        printf("FAIL: union static write, got %d\n", Tagged.hits)
        return 1
    printf("union class static: ok\n")
    return 0


# ---------------------------------------------------------------------------
# generic factory: each instantiation owns an independent static
# ---------------------------------------------------------------------------

def make_counted(elem_type):
    @compile(suffix=elem_type)
    class _Counted:
        value: elem_type
        created: static[i32] = 0
    return _Counted


CountedI = make_counted(i32)
CountedF = make_counted(f64)


@compile
def test_generic_class_static() -> i32:
    CountedI.created = 10
    CountedF.created = 20
    if CountedI.created != i32(10) or CountedF.created != i32(20):
        printf("FAIL: generic statics interfere: i=%d f=%d\n",
               CountedI.created, CountedF.created)
        return 1
    printf("generic class static: ok\n")
    return 0


# ---------------------------------------------------------------------------
# external-linkage class static (C-like exported object)
# ---------------------------------------------------------------------------

@compile(linkage='external')
class ExtCounter:
    n: static[i32] = 7


@compile
def test_class_static_external_linkage() -> i32:
    if ExtCounter.n != i32(7):
        printf("FAIL: external static seed, got %d\n", ExtCounter.n)
        return 1
    ExtCounter.n = ExtCounter.n + 1
    if ExtCounter.n != i32(8):
        printf("FAIL: external static write, got %d\n", ExtCounter.n)
        return 1
    printf("class static external linkage: ok\n")
    return 0


# ---------------------------------------------------------------------------
# internal-linkage class static (explicit opt-in, module-private storage)
# ---------------------------------------------------------------------------

@compile(linkage='internal')
class LocalCounter:
    # volatile defeats constant folding: an internal global that folds away
    # legitimately disappears from the optimised IR, leaving nothing to
    # inspect for the linkage test below.
    n: static[volatile[i32]] = 5


@compile
def read_local_counter() -> i32:
    LocalCounter.n = LocalCounter.n + 1
    return LocalCounter.n


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

@compile
def main() -> i32:
    printf("=== Class-Level Static Member Tests ===\n")
    if test_class_static_basic() != i32(0):
        return 1
    if test_class_static_persistence() != i32(0):
        return 1
    if test_class_static_zero_init() != i32(0):
        return 1
    if test_class_static_aggregate_seed() != i32(0):
        return 1
    if test_class_static_shared_storage() != i32(0):
        return 1
    if test_union_class_static() != i32(0):
        return 1
    if test_generic_class_static() != i32(0):
        return 1
    if test_class_static_external_linkage() != i32(0):
        return 1
    if read_local_counter() != i32(6):
        printf("FAIL: internal static read, got %d\n", read_local_counter())
        return 1
    printf("=== All Class Static Tests Complete ===\n")
    return 0


# ---------------------------------------------------------------------------
# error tests (Python side)
# ---------------------------------------------------------------------------

def test_error_static_const_member() -> bool:
    """Writing a static[const[T]] member raises RuntimeError."""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_class_static_const')
    try:
        @compile(suffix="bad_class_static_const")
        class BadConstMember:
            x: static[const[i32]] = 1

        @compile(suffix="bad_class_static_const")
        def should_fail() -> i32:
            BadConstMember.x = 2  # ERROR: Cannot modify static const member
            return 0

        flush_all_pending_outputs()
        print("FAIL test_error_static_const_member - should have raised RuntimeError")
        return False
    except RuntimeError as e:
        if "const" in str(e).lower():
            print(f"OK test_error_static_const_member passed: {e}")
            return True
        print(f"FAIL test_error_static_const_member - wrong error: {e}")
        return False
    finally:
        clear_failed_group(group_key)


def _global_line(ir_text: str, class_name: str, member: str):
    import re
    name = f"{class_name}.{member}"
    # llvmlite prints IR identifiers unquoted when they only contain
    # [a-zA-Z$._0-9-], so accept both @Name.member and @"Name.member".
    pattern = re.compile(rf'@"?{re.escape(name)}"?\s*=')
    for line in ir_text.splitlines():
        if pattern.search(line):
            return line.strip()
    return None


def test_class_static_linkage_ir() -> bool:
    """Default class statics are weak_odr (one shared entity); 'internal'
    opts into module-private storage; linkage='external' is exported."""
    @extern
    class ExternSlot:
        x: static[i32]

    @compile(suffix="read_extern_slot")
    def read_extern_slot() -> i32:
        return ExternSlot.x

    flush_all_pending_outputs()
    default_ir = get_output_manager().get_ir_text(bump._group_key)
    default_line = _global_line(default_ir, "Counter", "count")
    if default_line is None or "weak_odr" not in default_line:
        print(f"FAIL test_class_static_linkage_ir: default not weak_odr: {default_line}")
        return False

    internal_line = _global_line(default_ir, "LocalCounter", "n")
    if internal_line is None or "internal" not in internal_line:
        print(f"FAIL test_class_static_linkage_ir: explicit internal missing: {internal_line}")
        return False

    ext_ir = get_output_manager().get_ir_text(test_class_static_external_linkage._group_key)
    ext_line = _global_line(ext_ir, "ExtCounter", "n")
    if ext_line is None or "internal" in ext_line:
        print(f"FAIL test_class_static_linkage_ir: external still internal: {ext_line}")
        return False

    decl_ir = get_output_manager().get_ir_text(read_extern_slot._group_key)
    decl_line = _global_line(decl_ir, "ExternSlot", "x")
    if decl_line is None or "external" not in decl_line:
        print(f"FAIL test_class_static_linkage_ir: extern decl missing: {decl_line}")
        return False
    # A declaration ends at the type ("@x = external global i32"); a
    # definition appends an initializer ("... global i32 41").
    if not decl_line.rstrip().endswith(" i32"):
        print(f"FAIL test_class_static_linkage_ir: extern decl has initializer: {decl_line}")
        return False

    print("OK test_class_static_linkage_ir passed")
    return True


def run_error_tests() -> bool:
    all_passed = True
    print("\n=== Class Static Error Tests ===\n")
    if not test_error_static_const_member():
        all_passed = False
    if not test_class_static_linkage_ir():
        all_passed = False
    print("\n=== Error Tests Complete ===\n")
    return all_passed


if __name__ == "__main__":
    rc = main()
    all_passed = run_error_tests()
    sys.exit(0 if (rc == 0 and all_passed) else 1)
