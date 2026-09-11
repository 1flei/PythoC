# -*- coding: utf-8 -*-
"""
Composite "realistic mini-library" test for cimport.

One cohesive C library (one header + two .c sources) exercising every
cimport feature family in a single module, driven by an @compile
"application" that uses all of it together.  This catches feature
interaction bugs that the unit-style tests miss.

The fixture library is a tiny vector store:
- opaque handle type (typedef struct vs_store vs_store)
- create/destroy functions, extern global allocation counter
- numeric macros (VS_VERSION / VS_MAX_DIM) used in @compile arithmetic
- enum with explicit + negative values for error codes
- static inline helpers in the header (dimension check, default capacity,
  vec3 constructor / dot / cross)
- a bitfield struct inside the store, exposed only through accessors
- a function-pointer hook (on-resize callback set from @compile)
- a variadic log function backed by an internal buffer
- struct-by-value vec3 (dot/cross as static inline, normalize in a .c
  taking and returning by value)

Note: @compile wrappers are defined at module level because pythoc requires
all @compile definitions to precede the first native call from this module.
"""
from __future__ import annotations

import os
import sys
import unittest

from pythoc import compile, i32, f64, ptr, nullptr, void


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
# Module-level fixtures: the mini-library + @compile application
# =============================================================================

_fixture_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'build', 'test', 'cimport_project'))
os.makedirs(_fixture_dir, exist_ok=True)


def _write_fixture(name: str, content: str) -> str:
    path = os.path.join(_fixture_dir, name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path


if _BACKEND_AVAILABLE:
    from pythoc.cimport import cimport

    _header = _write_fixture('vs.h', '''
#ifndef VS_H
#define VS_H

#define VS_VERSION 2
#define VS_MAX_DIM 4

enum VsErr { VS_OK = 0, VS_ERR_DIM = -1, VS_ERR_RANGE = -2 };

struct vec3 { double x; double y; double z; };

typedef struct vs_store vs_store;

typedef void (*vs_resize_cb)(int new_cap);

extern int vs_total_allocs;
extern int vs_cb_count;

static inline int vs_dim_ok(int dim) { return dim > 0 && dim <= VS_MAX_DIM; }
static inline int vs_default_cap(void) { return 2; }

static inline struct vec3 vs_vec3(double x, double y, double z) {
    struct vec3 v = {x, y, z};
    return v;
}
static inline double vs_dot(struct vec3 a, struct vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
static inline struct vec3 vs_cross(struct vec3 a, struct vec3 b) {
    struct vec3 r = {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
    return r;
}

vs_store *vs_create(int dim);
void vs_destroy(vs_store *s);
int vs_count(vs_store *s);
enum VsErr vs_push(vs_store *s, struct vec3 v);
enum VsErr vs_get_into(vs_store *s, int i, struct vec3 *out);
struct vec3 vs_normalize(struct vec3 v);
void vs_set_resize_cb(vs_store *s, vs_resize_cb cb);
int vs_is_verbose(vs_store *s);
void vs_set_verbose(vs_store *s, int v);
int vs_err_code(enum VsErr e);
void vs_log(const char *fmt, ...);
const char *vs_last_log(void);

#endif
''')

    _core = _write_fixture('vs_core.c', '''
#include <stdlib.h>
#include "vs.h"

int vs_total_allocs = 0;

struct vs_store {
    int dim;
    int count;
    int cap;
    struct vec3 *data;
    vs_resize_cb cb;
    struct {
        unsigned dim_ok : 3;
        unsigned verbose : 1;
    } flags;
};

vs_store *vs_create(int dim) {
    if (!vs_dim_ok(dim))
        return NULL;
    vs_store *s = (vs_store *)malloc(sizeof(vs_store));
    if (!s)
        return NULL;
    vs_total_allocs++;
    s->dim = dim;
    s->count = 0;
    s->cap = vs_default_cap();
    s->data = (struct vec3 *)malloc(sizeof(struct vec3) * s->cap);
    vs_total_allocs++;
    s->cb = NULL;
    s->flags.dim_ok = 1;
    s->flags.verbose = 0;
    return s;
}

void vs_destroy(vs_store *s) {
    if (!s)
        return;
    free(s->data);
    free(s);
}

int vs_count(vs_store *s) { return s->count; }

enum VsErr vs_push(vs_store *s, struct vec3 v) {
    if (!s)
        return VS_ERR_DIM;
    if (s->count == s->cap) {
        int new_cap = s->cap * 2;
        s->data = (struct vec3 *)realloc(s->data, sizeof(struct vec3) * new_cap);
        vs_total_allocs++;
        s->cap = new_cap;
        if (s->cb)
            s->cb(new_cap);
    }
    s->data[s->count++] = v;
    return VS_OK;
}

enum VsErr vs_get_into(vs_store *s, int i, struct vec3 *out) {
    if (!s || !out)
        return VS_ERR_DIM;
    if (i < 0 || i >= s->count)
        return VS_ERR_RANGE;
    *out = s->data[i];
    return VS_OK;
}

void vs_set_resize_cb(vs_store *s, vs_resize_cb cb) { s->cb = cb; }

int vs_is_verbose(vs_store *s) { return s->flags.verbose; }

void vs_set_verbose(vs_store *s, int v) { s->flags.verbose = v ? 1 : 0; }
''')

    _util = _write_fixture('vs_util.c', '''
#include <stdarg.h>
#include <stdio.h>
#include <math.h>
#include "vs.h"

int vs_cb_count = 0;

static char vs_log_buf[256];

int vs_err_code(enum VsErr e) { return (int)e; }

struct vec3 vs_normalize(struct vec3 v) {
    double len = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    struct vec3 r = {v.x / len, v.y / len, v.z / len};
    return r;
}

void vs_log(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(vs_log_buf, sizeof(vs_log_buf), fmt, ap);
    va_end(ap);
}

const char *vs_last_log(void) { return vs_log_buf; }
''')

    _vs = cimport(_header, sources=[_core, _util],
                  compile_sources=True, include_dirs=[_fixture_dir])
    # includes=True: system string.h delegates to _string.h on newer
    # macOS SDKs.
    _string = cimport('string.h', lib='c', includes=True)

    vec3 = _vs.vec3
    vs_store = _vs.vs_store
    vs_create = _vs.vs_create
    vs_destroy = _vs.vs_destroy
    vs_count = _vs.vs_count
    vs_push = _vs.vs_push
    vs_get_into = _vs.vs_get_into
    vs_normalize = _vs.vs_normalize
    vs_set_resize_cb = _vs.vs_set_resize_cb
    vs_is_verbose = _vs.vs_is_verbose
    vs_set_verbose = _vs.vs_set_verbose
    vs_err_code = _vs.vs_err_code
    vs_log = _vs.vs_log
    vs_last_log = _vs.vs_last_log
    vs_dim_ok = _vs.vs_dim_ok
    vs_default_cap = _vs.vs_default_cap
    vs_vec3 = _vs.vs_vec3
    vs_dot = _vs.vs_dot
    vs_cross = _vs.vs_cross
    vs_total_allocs = _vs.vs_total_allocs
    vs_cb_count = _vs.vs_cb_count
    VS_VERSION = _vs.VS_VERSION
    VS_MAX_DIM = _vs.VS_MAX_DIM
    strcmp = _string.strcmp

    # --- resize hook set from @compile: calls back into the library ---
    @compile
    def on_resize(new_cap: i32) -> void:
        vs_cb_count += 1
        vs_log("cap=%d", new_cap)

    # --- the application ---
    @compile
    def app_macro_arithmetic() -> i32:
        return VS_VERSION * 10 + VS_MAX_DIM

    @compile
    def app_inline_helpers() -> i32:
        if vs_dim_ok(3) != 1:
            return 1
        if vs_dim_ok(5) != 0:
            return 2
        if vs_dim_ok(0) != 0:
            return 3
        if vs_default_cap() != 2:
            return 4
        return 0

    @compile
    def app_vec3_inline_math() -> i32:
        a: vec3 = vs_vec3(1.0, 2.0, 3.0)
        b: vec3 = vs_vec3(4.0, 5.0, 6.0)
        d: f64 = vs_dot(a, b)
        if d != 32.0:
            return 1
        ux: vec3 = vs_vec3(1.0, 0.0, 0.0)
        uy: vec3 = vs_vec3(0.0, 1.0, 0.0)
        c: vec3 = vs_cross(ux, uy)
        if c.x != 0.0 or c.y != 0.0 or c.z != 1.0:
            return 2
        return 0

    @compile
    def app_normalize_byval() -> i32:
        v: vec3 = vs_vec3(3.0, 4.0, 0.0)
        n: vec3 = vs_normalize(v)
        len2: f64 = vs_dot(n, n)
        if len2 < 0.99 or len2 > 1.01:
            return 1
        if n.z != 0.0:
            return 2
        return 0

    @compile
    def app_store_lifecycle() -> i32:
        allocs_before: i32 = vs_total_allocs
        s: ptr[vs_store] = vs_create(3)
        if s == nullptr:
            return 1
        # create allocates the handle + the data buffer
        if vs_total_allocs != allocs_before + 2:
            return 2
        vs_set_verbose(s, 1)
        if vs_is_verbose(s) != 1:
            return 3
        vs_set_verbose(s, 0)
        if vs_is_verbose(s) != 0:
            return 4
        cb_before: i32 = vs_cb_count
        vs_set_resize_cb(s, on_resize)
        # default capacity is 2, so the third push grows and fires the hook
        e: i32 = 0
        e += vs_err_code(vs_push(s, vs_vec3(1.0, 0.0, 0.0)))
        e += vs_err_code(vs_push(s, vs_vec3(0.0, 1.0, 0.0)))
        e += vs_err_code(vs_push(s, vs_vec3(0.0, 0.0, 1.0)))
        if e != 0:
            return 5
        if vs_count(s) != 3:
            return 6
        if vs_cb_count != cb_before + 1:
            return 7
        # the hook logged through the variadic vs_log
        if strcmp(vs_last_log(), "cap=4") != 0:
            return 8
        # growth reallocated once
        if vs_total_allocs != allocs_before + 3:
            return 9
        out: vec3
        if vs_err_code(vs_get_into(s, 2, ptr(out))) != 0:
            return 10
        if out.x != 0.0 or out.y != 0.0 or out.z != 1.0:
            return 11
        vs_destroy(s)
        return 0

    @compile
    def app_error_codes() -> i32:
        # out-of-range dim: opaque handle comes back null
        bad: ptr[vs_store] = vs_create(VS_MAX_DIM + 1)
        if bad != nullptr:
            return 1
        s: ptr[vs_store] = vs_create(3)
        if s == nullptr:
            return 2
        out: vec3
        code: i32 = vs_err_code(vs_get_into(s, 99, ptr(out)))
        vs_destroy(s)
        if code != -2:
            return 3
        # push through a null store reports the dim error enum
        if vs_err_code(vs_push(nullptr, vs_vec3(1.0, 2.0, 3.0))) != -1:
            return 4
        return 0

    @compile
    def app_full_roundtrip() -> i32:
        # push orthogonal vectors, read them back, verify geometry
        s: ptr[vs_store] = vs_create(3)
        if s == nullptr:
            return 1
        vs_push(s, vs_vec3(1.0, 2.0, 3.0))
        vs_push(s, vs_vec3(4.0, 5.0, 6.0))
        a: vec3
        b: vec3
        vs_get_into(s, 0, ptr(a))
        vs_get_into(s, 1, ptr(b))
        d: f64 = vs_dot(a, b)
        if d != 32.0:
            return 2
        c: vec3 = vs_cross(a, b)
        n: vec3 = vs_normalize(c)
        len2: f64 = vs_dot(n, n)
        if len2 < 0.99 or len2 > 1.01:
            return 3
        vs_destroy(s)
        return 0


@unittest.skipUnless(_BACKEND_AVAILABLE, "clang backend or cc not available")
class TestCimportProjectFeatures(unittest.TestCase):
    """Individual feature families inside the composite library."""

    def test_macros_in_compile_arithmetic(self):
        self.assertEqual(app_macro_arithmetic(), 24)

    def test_static_inline_helpers(self):
        self.assertEqual(app_inline_helpers(), 0)

    def test_struct_by_value_inline_math(self):
        self.assertEqual(app_vec3_inline_math(), 0)

    def test_struct_by_value_from_c_source(self):
        self.assertEqual(app_normalize_byval(), 0)

    def test_error_enum_codes(self):
        self.assertEqual(app_error_codes(), 0)


@unittest.skipUnless(_BACKEND_AVAILABLE, "clang backend or cc not available")
class TestCimportProjectIntegration(unittest.TestCase):
    """Everything together: handle, globals, bitfields, callback, varargs."""

    def test_store_lifecycle(self):
        self.assertEqual(app_store_lifecycle(), 0)

    def test_full_geometry_roundtrip(self):
        self.assertEqual(app_full_roundtrip(), 0)

    @unittest.skipIf(sys.platform == 'win32',
                     "Python-side access to process-global symbols (lib='') "
                     "is not supported on Windows")
    def test_globals_persist_across_calls(self):
        before = _vs.vs_total_allocs.value
        self.assertEqual(app_store_lifecycle(), 0)
        # one lifecycle run performs exactly three allocations
        self.assertEqual(_vs.vs_total_allocs.value, before + 3)


if __name__ == "__main__":
    unittest.main()
