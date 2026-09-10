# -*- coding: utf-8 -*-
"""
End-to-end tests for function pointers through cimport.

C function pointers map to pythoc func[...] values (a C function pointer is
a single indirection, and pythoc's func type is already a function-pointer
value), so they are first-class: passable to C, storable in structs,
returnable from C, and directly callable from @compile code.

Covers:
- fn-ptr typedef in a header binds as a func[...] type
- C function taking a fn-ptr parameter, called with an @compile callback
- struct with a fn-ptr field: field set to an @compile function, invoked
  by C through the struct
- C function returning a fn-ptr, with the result called from @compile
- fn-ptr as out-param (C writes a fn-ptr into caller-provided memory)
- reentrancy: a callback that itself calls back into cimport'd C functions
  while C is on the stack

Note: @compile wrappers are defined at module level because pythoc requires
all @compile definitions to precede the first native call from this module.
"""
from __future__ import annotations

import os
import unittest

from pythoc import compile, func, i32, ptr


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
    os.path.dirname(__file__), '..', '..', 'build', 'test', 'cimport_fnptr'))
os.makedirs(_fixture_dir, exist_ok=True)


def _write_fixture(name: str, content: str) -> str:
    path = os.path.join(_fixture_dir, name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path


if _BACKEND_AVAILABLE:
    from pythoc.cimport import cimport

    _header = _write_fixture('fnptr.h', '''
typedef int (*binop_t)(int, int);
typedef int (*unary_t)(int);

struct FnBox {
    unary_t fn;
    int bias;
};

int apply_twice(binop_t fn, int x);
int apply_n(unary_t fn, int x, int n);
binop_t pick_op(int which);
void fill_op(binop_t *out);
int call_box(struct FnBox *box, int x);
int c_add(int a, int b);
int c_mul(int a, int b);
int c_double(int x);
''')
    _source = _write_fixture('fnptr.c', '''
#include "fnptr.h"

int c_add(int a, int b) { return a + b; }
int c_mul(int a, int b) { return a * b; }
int c_double(int x) { return 2 * x; }

int apply_twice(binop_t fn, int x) { return fn(fn(x, x), fn(x, x)); }

int apply_n(unary_t fn, int x, int n) {
    int v = x;
    for (int i = 0; i < n; i++)
        v = fn(v);
    return v;
}

binop_t pick_op(int which) { return which ? c_mul : c_add; }

void fill_op(binop_t *out) { *out = c_add; }

int call_box(struct FnBox *box, int x) { return box->fn(x) + box->bias; }
''')
    _mod = cimport(_header, sources=[_source],
                   compile_sources=True, include_dirs=[_fixture_dir])
    binop_t = _mod.binop_t
    FnBox = _mod.FnBox
    apply_twice = _mod.apply_twice
    apply_n = _mod.apply_n
    pick_op = _mod.pick_op
    fill_op = _mod.fill_op
    call_box = _mod.call_box
    c_add = _mod.c_add
    c_mul = _mod.c_mul
    c_double = _mod.c_double

    # --- @compile callbacks ---
    @compile
    def pc_add(a: i32, b: i32) -> i32:
        return a + b

    @compile
    def pc_max(a: i32, b: i32) -> i32:
        if a > b:
            return a
        return b

    @compile
    def pc_inc(x: i32) -> i32:
        return x + 1

    @compile
    def pc_sq_plus1(x: i32) -> i32:
        return x * x + 1

    @compile
    def pc_reentrant_combine(a: i32, b: i32) -> i32:
        # Callback invoked from C that calls back into cimport'd C
        # functions while C is on the stack.
        return c_mul(c_add(a, b), c_double(1))

    # --- scenarios ---
    @compile
    def cb_as_param(x: i32) -> i32:
        return apply_twice(pc_add, x)

    @compile
    def cb_as_param_other(x: i32) -> i32:
        return apply_twice(pc_max, x)

    @compile
    def cb_applied_n_times(x: i32, n: i32) -> i32:
        return apply_n(pc_inc, x, n)

    @compile
    def reentrant_callback(x: i32) -> i32:
        return apply_twice(pc_reentrant_combine, x)

    @compile
    def returned_fnptr_called(which: i32, a: i32, b: i32) -> i32:
        op: binop_t = pick_op(which)
        return op(a, b)

    @compile
    def returned_fnptr_passed_back(which: i32, x: i32) -> i32:
        op: binop_t = pick_op(which)
        return apply_twice(op, x)

    @compile
    def out_param_fnptr(a: i32, b: i32) -> i32:
        op: binop_t
        fill_op(ptr(op))
        return op(a, b)

    @compile
    def struct_field_callback(x: i32) -> i32:
        box: FnBox
        box.fn = pc_sq_plus1
        box.bias = 7
        return call_box(ptr(box), x)

    @compile
    def struct_field_callback_swap(x: i32) -> i32:
        box: FnBox
        box.fn = pc_inc
        box.bias = 0
        first: i32 = call_box(ptr(box), x)
        box.fn = pc_sq_plus1
        second: i32 = call_box(ptr(box), x)
        return first * 1000 + second


@unittest.skipUnless(_BACKEND_AVAILABLE, "clang backend or cc not available")
class TestCimportFnptrBinding(unittest.TestCase):
    """The fn-ptr typedef binds as a callable func[...] type."""

    def test_typedef_is_func_type(self):
        self.assertTrue(issubclass(binop_t, func))
        self.assertEqual(binop_t.get_name(), "func[i32, i32, i32]")

    def test_struct_field_accessible(self):
        self.assertTrue(FnBox.has_field("fn"))
        self.assertTrue(FnBox.has_field("bias"))
        # { fn ptr, i32 } padded to 8-byte alignment
        self.assertEqual(FnBox.get_size_bytes(), 16)


@unittest.skipUnless(_BACKEND_AVAILABLE, "clang backend or cc not available")
class TestCimportFnptrParam(unittest.TestCase):
    """C functions taking fn-ptr parameters, fed @compile callbacks."""

    def test_apply_twice_add(self):
        # (x+x) + (x+x)
        self.assertEqual(cb_as_param(3), 12)
        self.assertEqual(cb_as_param(-2), -8)

    def test_apply_twice_max(self):
        # max(max(x,x), max(x,x)) == x
        self.assertEqual(cb_as_param_other(9), 9)

    def test_apply_n_times(self):
        self.assertEqual(cb_applied_n_times(10, 5), 15)
        self.assertEqual(cb_applied_n_times(0, 3), 3)

    def test_reentrant_callback(self):
        # combine(a,b) = (a+b)*2; apply_twice(combine, x):
        # inner = combine(x,x) = 4x; result = combine(4x,4x) = 16x
        self.assertEqual(reentrant_callback(1), 16)
        self.assertEqual(reentrant_callback(3), 48)


@unittest.skipUnless(_BACKEND_AVAILABLE, "clang backend or cc not available")
class TestCimportFnptrReturnAndOut(unittest.TestCase):
    """Fn-ptrs produced by C: return values and out-params."""

    def test_returned_fnptr_called_from_compile(self):
        self.assertEqual(returned_fnptr_called(0, 4, 5), 9)
        self.assertEqual(returned_fnptr_called(1, 4, 5), 20)

    def test_returned_fnptr_passed_back_to_c(self):
        # pick c_mul; apply_twice(c_mul, 4) = (4*4)*(4*4)
        self.assertEqual(returned_fnptr_passed_back(1, 4), 256)
        # pick c_add; apply_twice(c_add, 4) = (4+4)+(4+4)
        self.assertEqual(returned_fnptr_passed_back(0, 4), 16)

    def test_out_param_fnptr(self):
        # C writes c_add into caller memory; called from @compile.
        self.assertEqual(out_param_fnptr(10, 20), 30)
        self.assertEqual(out_param_fnptr(-1, 1), 0)


@unittest.skipUnless(_BACKEND_AVAILABLE, "clang backend or cc not available")
class TestCimportFnptrStructField(unittest.TestCase):
    """Structs carrying fn-ptr fields."""

    def test_call_through_struct(self):
        # sq_plus1(5) + 7 = 33
        self.assertEqual(struct_field_callback(5), 33)

    def test_swap_field_between_calls(self):
        # inc(9) = 10, sq_plus1(9) = 82
        self.assertEqual(struct_field_callback_swap(9), 10082)


if __name__ == "__main__":
    unittest.main()
