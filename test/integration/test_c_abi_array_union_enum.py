#!/usr/bin/env python3
"""
C ABI tests for array, union, and enum types.

Tests all 4 cases for each type:
1. pythoc type as C function param
2. pythoc function returns type to C
3. C type as pythoc function param
4. C function returns type to pythoc

Array: fixed-size arrays passed by value (small) or pointer (large)
Union: same size as largest member, coerced like struct
Enum: struct{tag_t, union[payloads...]} - tag + payload union
"""

import unittest
import os
import sys
import ctypes
import tempfile
import subprocess

# =============================================================================
# C Source Code
# =============================================================================

C_SOURCE = """
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// ============================================================================
// Array types - fixed size arrays
// ============================================================================

// Small arrays (<=16 bytes) - may be passed in registers
typedef struct { int32_t data[2]; } Array2_i32;   // 8 bytes
typedef struct { int64_t data[2]; } Array2_i64;   // 16 bytes

// Large arrays (>16 bytes) - passed by pointer
typedef struct { int32_t data[4]; } Array4_i32;   // 16 bytes
typedef struct { int64_t data[4]; } Array4_i64;   // 32 bytes
typedef struct { int32_t data[8]; } Array8_i32;   // 32 bytes

// ============================================================================
// Union types - size of largest member
// ============================================================================

// Small union (8 bytes) - coerced to i64
typedef union {
    int32_t int_val;
    float float_val;
} Union_i32_f32;

// Medium union (16 bytes) - coerced to (i64, i64)
typedef union {
    int64_t long_val;
    double double_val;
    struct { int32_t a; int32_t b; } pair;
} Union_i64_f64;

// Large union (>16 bytes) - sret/byval
typedef union {
    int64_t arr[4];
    struct { int64_t a; int64_t b; int64_t c; int64_t d; } quad;
} Union_Large32;

// ============================================================================
// Enum types - struct{tag, union[payloads]}
// ============================================================================

// Simple enum (no payload) - just tag
typedef struct {
    int8_t tag;
} SimpleEnum;

// Enum with payload - tag + union
// Result = Ok(i32) | Err(i32)
typedef struct {
    int32_t tag;
    union {
        int32_t ok_val;
        int32_t err_val;
    } payload;
} Result_i32;

// Enum with mixed payloads - tag + union of different sizes
// Value = Int(i32) | Float(f64) | None
typedef struct {
    int32_t tag;
    union {
        int32_t int_val;
        double float_val;
    } payload;
} Value_Enum;

// Large enum with struct payload
// Expression = Const(i32) | Add(i64, i64) | Mul(i64, i64)
typedef struct {
    int32_t tag;
    int32_t _pad;  // alignment padding
    union {
        int32_t const_val;
        struct { int64_t a; int64_t b; } binop;
    } payload;
} Expression_Enum;

// ============================================================================
// C functions for Array
// ============================================================================

// Return arrays
Array2_i32 c_return_array2_i32(void) {
    Array2_i32 arr = {{10, 20}};
    return arr;
}

Array4_i32 c_return_array4_i32(void) {
    Array4_i32 arr = {{1, 2, 3, 4}};
    return arr;
}

Array4_i64 c_return_array4_i64(void) {
    Array4_i64 arr = {{100, 200, 300, 400}};
    return arr;
}

// Take array params
int32_t c_sum_array2_i32(Array2_i32 arr) {
    return arr.data[0] + arr.data[1];
}

int32_t c_sum_array4_i32(Array4_i32 arr) {
    return arr.data[0] + arr.data[1] + arr.data[2] + arr.data[3];
}

int64_t c_sum_array4_i64(Array4_i64 arr) {
    return arr.data[0] + arr.data[1] + arr.data[2] + arr.data[3];
}

// Take and return arrays
Array2_i32 c_double_array2_i32(Array2_i32 arr) {
    Array2_i32 result = {{arr.data[0] * 2, arr.data[1] * 2}};
    return result;
}

Array4_i64 c_increment_array4_i64(Array4_i64 arr) {
    Array4_i64 result = {{arr.data[0] + 1, arr.data[1] + 1, arr.data[2] + 1, arr.data[3] + 1}};
    return result;
}

// ============================================================================
// C functions for Union
// ============================================================================

// Return unions
Union_i32_f32 c_return_union_int(void) {
    Union_i32_f32 u;
    u.int_val = 42;
    return u;
}

Union_i64_f64 c_return_union_double(void) {
    Union_i64_f64 u;
    u.double_val = 3.14159;
    return u;
}

Union_Large32 c_return_union_large(void) {
    Union_Large32 u;
    u.arr[0] = 10;
    u.arr[1] = 20;
    u.arr[2] = 30;
    u.arr[3] = 40;
    return u;
}

// Take union params
int32_t c_get_union_int(Union_i32_f32 u) {
    return u.int_val;
}

double c_get_union_double(Union_i64_f64 u) {
    return u.double_val;
}

int64_t c_sum_union_large(Union_Large32 u) {
    return u.arr[0] + u.arr[1] + u.arr[2] + u.arr[3];
}

// Take and return unions
Union_i32_f32 c_double_union_int(Union_i32_f32 u) {
    Union_i32_f32 result;
    result.int_val = u.int_val * 2;
    return result;
}

Union_Large32 c_increment_union_large(Union_Large32 u) {
    Union_Large32 result;
    result.arr[0] = u.arr[0] + 1;
    result.arr[1] = u.arr[1] + 1;
    result.arr[2] = u.arr[2] + 1;
    result.arr[3] = u.arr[3] + 1;
    return result;
}

// ============================================================================
// C functions for Enum
// ============================================================================

// Return enums
SimpleEnum c_return_simple_enum(int8_t tag) {
    SimpleEnum e = {tag};
    return e;
}

Result_i32 c_return_result_ok(int32_t val) {
    Result_i32 r;
    r.tag = 0;  // Ok
    r.payload.ok_val = val;
    return r;
}

Result_i32 c_return_result_err(int32_t code) {
    Result_i32 r;
    r.tag = 1;  // Err
    r.payload.err_val = code;
    return r;
}

Value_Enum c_return_value_int(int32_t val) {
    Value_Enum v;
    v.tag = 0;  // Int
    v.payload.int_val = val;
    return v;
}

Value_Enum c_return_value_float(double val) {
    Value_Enum v;
    v.tag = 1;  // Float
    v.payload.float_val = val;
    return v;
}

Expression_Enum c_return_expr_const(int32_t val) {
    Expression_Enum e;
    e.tag = 0;  // Const
    e.payload.const_val = val;
    return e;
}

Expression_Enum c_return_expr_add(int64_t a, int64_t b) {
    Expression_Enum e;
    e.tag = 1;  // Add
    e.payload.binop.a = a;
    e.payload.binop.b = b;
    return e;
}

// Take enum params
int8_t c_get_simple_enum_tag(SimpleEnum e) {
    return e.tag;
}

int32_t c_get_result_value(Result_i32 r) {
    if (r.tag == 0) {
        return r.payload.ok_val;
    } else {
        return -r.payload.err_val;  // Return negative for error
    }
}

int64_t c_eval_expression(Expression_Enum e) {
    switch (e.tag) {
        case 0: return e.payload.const_val;
        case 1: return e.payload.binop.a + e.payload.binop.b;
        case 2: return e.payload.binop.a * e.payload.binop.b;
        default: return 0;
    }
}

// Take and return enums
Result_i32 c_double_result(Result_i32 r) {
    Result_i32 result;
    result.tag = r.tag;
    if (r.tag == 0) {
        result.payload.ok_val = r.payload.ok_val * 2;
    } else {
        result.payload.err_val = r.payload.err_val * 2;
    }
    return result;
}

Expression_Enum c_negate_expr(Expression_Enum e) {
    Expression_Enum result;
    result.tag = e.tag;
    if (e.tag == 0) {
        result.payload.const_val = -e.payload.const_val;
    } else {
        result.payload.binop.a = -e.payload.binop.a;
        result.payload.binop.b = -e.payload.binop.b;
    }
    return result;
}

// ============================================================================
// Function pointer callers for C calling pythoc
// ============================================================================

// Array function pointers
typedef Array2_i32 (*fn_return_array2_i32_t)(void);
typedef Array4_i64 (*fn_return_array4_i64_t)(void);
typedef int32_t (*fn_sum_array2_i32_t)(Array2_i32);
typedef int64_t (*fn_sum_array4_i64_t)(Array4_i64);
typedef Array2_i32 (*fn_double_array2_i32_t)(Array2_i32);
typedef Array4_i64 (*fn_increment_array4_i64_t)(Array4_i64);

int32_t c_call_fn_return_array2_i32(fn_return_array2_i32_t fn) {
    Array2_i32 arr = fn();
    return arr.data[0] + arr.data[1];
}

int64_t c_call_fn_return_array4_i64(fn_return_array4_i64_t fn) {
    Array4_i64 arr = fn();
    return arr.data[0] + arr.data[1] + arr.data[2] + arr.data[3];
}

int32_t c_call_fn_sum_array2_i32(fn_sum_array2_i32_t fn) {
    Array2_i32 arr = {{25, 75}};
    return fn(arr);
}

int64_t c_call_fn_sum_array4_i64(fn_sum_array4_i64_t fn) {
    Array4_i64 arr = {{100, 200, 300, 400}};
    return fn(arr);
}

int32_t c_call_fn_double_array2_i32(fn_double_array2_i32_t fn) {
    Array2_i32 arr = {{10, 20}};
    Array2_i32 result = fn(arr);
    return result.data[0] + result.data[1];
}

int64_t c_call_fn_increment_array4_i64(fn_increment_array4_i64_t fn) {
    Array4_i64 arr = {{10, 20, 30, 40}};
    Array4_i64 result = fn(arr);
    return result.data[0] + result.data[1] + result.data[2] + result.data[3];
}

// Union function pointers
typedef Union_i32_f32 (*fn_return_union_i32_f32_t)(void);
typedef Union_Large32 (*fn_return_union_large_t)(void);
typedef int32_t (*fn_get_union_int_t)(Union_i32_f32);
typedef int64_t (*fn_sum_union_large_t)(Union_Large32);
typedef Union_i32_f32 (*fn_double_union_int_t)(Union_i32_f32);
typedef Union_Large32 (*fn_increment_union_large_t)(Union_Large32);

int32_t c_call_fn_return_union_i32_f32(fn_return_union_i32_f32_t fn) {
    Union_i32_f32 u = fn();
    return u.int_val;
}

int64_t c_call_fn_return_union_large(fn_return_union_large_t fn) {
    Union_Large32 u = fn();
    return u.arr[0] + u.arr[1] + u.arr[2] + u.arr[3];
}

int32_t c_call_fn_get_union_int(fn_get_union_int_t fn) {
    Union_i32_f32 u;
    u.int_val = 123;
    return fn(u);
}

int64_t c_call_fn_sum_union_large(fn_sum_union_large_t fn) {
    Union_Large32 u;
    u.arr[0] = 10;
    u.arr[1] = 20;
    u.arr[2] = 30;
    u.arr[3] = 40;
    return fn(u);
}

int32_t c_call_fn_double_union_int(fn_double_union_int_t fn) {
    Union_i32_f32 u;
    u.int_val = 50;
    Union_i32_f32 result = fn(u);
    return result.int_val;
}

int64_t c_call_fn_increment_union_large(fn_increment_union_large_t fn) {
    Union_Large32 u;
    u.arr[0] = 10;
    u.arr[1] = 20;
    u.arr[2] = 30;
    u.arr[3] = 40;
    Union_Large32 result = fn(u);
    return result.arr[0] + result.arr[1] + result.arr[2] + result.arr[3];
}

// Enum function pointers
typedef Result_i32 (*fn_return_result_t)(void);
typedef Expression_Enum (*fn_return_expr_t)(void);
typedef int32_t (*fn_get_result_t)(Result_i32);
typedef int64_t (*fn_eval_expr_t)(Expression_Enum);
typedef Result_i32 (*fn_double_result_t)(Result_i32);
typedef Expression_Enum (*fn_negate_expr_t)(Expression_Enum);

int32_t c_call_fn_return_result(fn_return_result_t fn) {
    Result_i32 r = fn();
    return r.tag == 0 ? r.payload.ok_val : -r.payload.err_val;
}

int64_t c_call_fn_return_expr(fn_return_expr_t fn) {
    Expression_Enum e = fn();
    switch (e.tag) {
        case 0: return e.payload.const_val;
        case 1: return e.payload.binop.a + e.payload.binop.b;
        case 2: return e.payload.binop.a * e.payload.binop.b;
        default: return 0;
    }
}

int32_t c_call_fn_get_result(fn_get_result_t fn) {
    Result_i32 r;
    r.tag = 0;
    r.payload.ok_val = 77;
    return fn(r);
}

int64_t c_call_fn_eval_expr(fn_eval_expr_t fn) {
    Expression_Enum e;
    e.tag = 1;  // Add
    e.payload.binop.a = 100;
    e.payload.binop.b = 200;
    return fn(e);
}

int32_t c_call_fn_double_result(fn_double_result_t fn) {
    Result_i32 r;
    r.tag = 0;
    r.payload.ok_val = 25;
    Result_i32 result = fn(r);
    return result.payload.ok_val;
}

int64_t c_call_fn_negate_expr(fn_negate_expr_t fn) {
    Expression_Enum e;
    e.tag = 1;  // Add
    e.payload.binop.a = 10;
    e.payload.binop.b = 20;
    Expression_Enum result = fn(e);
    return result.payload.binop.a + result.payload.binop.b;
}
"""

# =============================================================================
# Compile C library
# =============================================================================

_c_lib = None
_c_lib_path = None

def _compile_c_library():
    """Compile C library and return path"""
    global _c_lib, _c_lib_path
    
    if _c_lib_path is not None:
        return _c_lib_path
    
    c_file = os.path.join(tempfile.gettempdir(), 'test_c_abi_aue.c')
    so_file = os.path.join(tempfile.gettempdir(), 'libtest_c_abi_aue.so')
    
    with open(c_file, 'w') as f:
        f.write(C_SOURCE)
    
    result = subprocess.run(
        ['gcc', '-shared', '-fPIC', '-O2', '-o', so_file, c_file],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"C compilation failed: {result.stderr}")
    
    _c_lib_path = so_file
    return so_file

def get_c_library():
    """Get loaded C library"""
    global _c_lib
    if _c_lib is None:
        _compile_c_library()
        _c_lib = ctypes.CDLL(_c_lib_path)
    return _c_lib

# Compile C library NOW before @extern declarations
C_LIB_PATH = _compile_c_library()


# =============================================================================
# pythoc type definitions
# =============================================================================

from pythoc import (
    compile, struct, union, enum, extern,
    i8, i16, i32, i64,
    u8, u16, u32, u64,
    f32, f64,
    array,
    ptr, func, void
)

# -----------------------------------------------------------------------------
# Array types (wrapped in struct for C ABI compatibility)
# -----------------------------------------------------------------------------

@struct
class Array2_i32:
    data: array[i32, 2]

@struct
class Array4_i32:
    data: array[i32, 4]

@struct
class Array4_i64:
    data: array[i64, 4]

# -----------------------------------------------------------------------------
# Union types
# -----------------------------------------------------------------------------

@union
class Union_i32_f32:
    int_val: i32
    float_val: f32

@union
class Union_i64_f64:
    long_val: i64
    double_val: f64

@struct
class QuadI64:
    a: i64
    b: i64
    c: i64
    d: i64

@union
class Union_Large32:
    arr: array[i64, 4]
    quad: QuadI64

# -----------------------------------------------------------------------------
# Enum types
# -----------------------------------------------------------------------------

@enum(i8)
class SimpleEnum:
    A: None
    B: None
    C: None

@enum(i32)
class Result_i32:
    Ok: i32
    Err: i32

@enum(i32)
class Value_Enum:
    Int: i32
    Float: f64
    NoneVal: None

@struct
class BinOp:
    a: i64
    b: i64

@enum(i32)
class Expression_Enum:
    Const: i32
    Add: BinOp
    Mul: BinOp


# =============================================================================
# @extern declarations for C functions
# =============================================================================

# Array functions
@extern(lib=C_LIB_PATH)
def c_return_array2_i32() -> Array2_i32: ...

@extern(lib=C_LIB_PATH)
def c_return_array4_i32() -> Array4_i32: ...

@extern(lib=C_LIB_PATH)
def c_return_array4_i64() -> Array4_i64: ...

@extern(lib=C_LIB_PATH)
def c_sum_array2_i32(arr: Array2_i32) -> i32: ...

@extern(lib=C_LIB_PATH)
def c_sum_array4_i32(arr: Array4_i32) -> i32: ...

@extern(lib=C_LIB_PATH)
def c_sum_array4_i64(arr: Array4_i64) -> i64: ...

@extern(lib=C_LIB_PATH)
def c_double_array2_i32(arr: Array2_i32) -> Array2_i32: ...

@extern(lib=C_LIB_PATH)
def c_increment_array4_i64(arr: Array4_i64) -> Array4_i64: ...

# Union functions
@extern(lib=C_LIB_PATH)
def c_return_union_int() -> Union_i32_f32: ...

@extern(lib=C_LIB_PATH)
def c_return_union_double() -> Union_i64_f64: ...

@extern(lib=C_LIB_PATH)
def c_return_union_large() -> Union_Large32: ...

@extern(lib=C_LIB_PATH)
def c_get_union_int(u: Union_i32_f32) -> i32: ...

@extern(lib=C_LIB_PATH)
def c_get_union_double(u: Union_i64_f64) -> f64: ...

@extern(lib=C_LIB_PATH)
def c_sum_union_large(u: Union_Large32) -> i64: ...

@extern(lib=C_LIB_PATH)
def c_double_union_int(u: Union_i32_f32) -> Union_i32_f32: ...

@extern(lib=C_LIB_PATH)
def c_increment_union_large(u: Union_Large32) -> Union_Large32: ...

# Enum functions
@extern(lib=C_LIB_PATH)
def c_return_simple_enum(tag: i8) -> SimpleEnum: ...

@extern(lib=C_LIB_PATH)
def c_return_result_ok(val: i32) -> Result_i32: ...

@extern(lib=C_LIB_PATH)
def c_return_result_err(code: i32) -> Result_i32: ...

@extern(lib=C_LIB_PATH)
def c_return_value_int(val: i32) -> Value_Enum: ...

@extern(lib=C_LIB_PATH)
def c_return_value_float(val: f64) -> Value_Enum: ...

@extern(lib=C_LIB_PATH)
def c_return_expr_const(val: i32) -> Expression_Enum: ...

@extern(lib=C_LIB_PATH)
def c_return_expr_add(a: i64, b: i64) -> Expression_Enum: ...

@extern(lib=C_LIB_PATH)
def c_get_simple_enum_tag(e: SimpleEnum) -> i8: ...

@extern(lib=C_LIB_PATH)
def c_get_result_value(r: Result_i32) -> i32: ...

@extern(lib=C_LIB_PATH)
def c_eval_expression(e: Expression_Enum) -> i64: ...

@extern(lib=C_LIB_PATH)
def c_double_result(r: Result_i32) -> Result_i32: ...

@extern(lib=C_LIB_PATH)
def c_negate_expr(e: Expression_Enum) -> Expression_Enum: ...

# Function pointer callers
@extern(lib=C_LIB_PATH)
def c_call_fn_return_array2_i32(fn: ptr[func[Array2_i32]]) -> i32: ...

@extern(lib=C_LIB_PATH)
def c_call_fn_return_array4_i64(fn: ptr[func[Array4_i64]]) -> i64: ...

@extern(lib=C_LIB_PATH)
def c_call_fn_sum_array2_i32(fn: ptr[func[Array2_i32, i32]]) -> i32: ...

@extern(lib=C_LIB_PATH)
def c_call_fn_sum_array4_i64(fn: ptr[func[Array4_i64, i64]]) -> i64: ...

@extern(lib=C_LIB_PATH)
def c_call_fn_double_array2_i32(fn: ptr[func[Array2_i32, Array2_i32]]) -> i32: ...

@extern(lib=C_LIB_PATH)
def c_call_fn_increment_array4_i64(fn: ptr[func[Array4_i64, Array4_i64]]) -> i64: ...

@extern(lib=C_LIB_PATH)
def c_call_fn_return_union_i32_f32(fn: ptr[func[Union_i32_f32]]) -> i32: ...

@extern(lib=C_LIB_PATH)
def c_call_fn_return_union_large(fn: ptr[func[Union_Large32]]) -> i64: ...

@extern(lib=C_LIB_PATH)
def c_call_fn_get_union_int(fn: ptr[func[Union_i32_f32, i32]]) -> i32: ...

@extern(lib=C_LIB_PATH)
def c_call_fn_sum_union_large(fn: ptr[func[Union_Large32, i64]]) -> i64: ...

@extern(lib=C_LIB_PATH)
def c_call_fn_double_union_int(fn: ptr[func[Union_i32_f32, Union_i32_f32]]) -> i32: ...

@extern(lib=C_LIB_PATH)
def c_call_fn_increment_union_large(fn: ptr[func[Union_Large32, Union_Large32]]) -> i64: ...

@extern(lib=C_LIB_PATH)
def c_call_fn_return_result(fn: ptr[func[Result_i32]]) -> i32: ...

@extern(lib=C_LIB_PATH)
def c_call_fn_return_expr(fn: ptr[func[Expression_Enum]]) -> i64: ...

@extern(lib=C_LIB_PATH)
def c_call_fn_get_result(fn: ptr[func[Result_i32, i32]]) -> i32: ...

@extern(lib=C_LIB_PATH)
def c_call_fn_eval_expr(fn: ptr[func[Expression_Enum, i64]]) -> i64: ...

@extern(lib=C_LIB_PATH)
def c_call_fn_double_result(fn: ptr[func[Result_i32, Result_i32]]) -> i32: ...

@extern(lib=C_LIB_PATH)
def c_call_fn_negate_expr(fn: ptr[func[Expression_Enum, Expression_Enum]]) -> i64: ...


# =============================================================================
# pythoc functions (to be called from C)
# =============================================================================

# Array functions
@compile
def pc_return_array2_i32() -> Array2_i32:
    arr: Array2_i32 = Array2_i32()
    arr.data[0] = 100
    arr.data[1] = 200
    return arr

@compile
def pc_return_array4_i64() -> Array4_i64:
    arr: Array4_i64 = Array4_i64()
    arr.data[0] = 10
    arr.data[1] = 20
    arr.data[2] = 30
    arr.data[3] = 40
    return arr

@compile
def pc_sum_array2_i32(arr: Array2_i32) -> i32:
    return arr.data[0] + arr.data[1]

@compile
def pc_sum_array4_i64(arr: Array4_i64) -> i64:
    return arr.data[0] + arr.data[1] + arr.data[2] + arr.data[3]

@compile
def pc_double_array2_i32(arr: Array2_i32) -> Array2_i32:
    result: Array2_i32 = Array2_i32()
    result.data[0] = arr.data[0] * 2
    result.data[1] = arr.data[1] * 2
    return result

@compile
def pc_increment_array4_i64(arr: Array4_i64) -> Array4_i64:
    result: Array4_i64 = Array4_i64()
    result.data[0] = arr.data[0] + 1
    result.data[1] = arr.data[1] + 1
    result.data[2] = arr.data[2] + 1
    result.data[3] = arr.data[3] + 1
    return result

# Union functions
@compile
def pc_return_union_i32_f32() -> Union_i32_f32:
    u: Union_i32_f32 = Union_i32_f32()
    u.int_val = 42
    return u

@compile
def pc_return_union_large() -> Union_Large32:
    u: Union_Large32 = Union_Large32()
    u.arr[0] = 10
    u.arr[1] = 20
    u.arr[2] = 30
    u.arr[3] = 40
    return u

@compile
def pc_get_union_int(u: Union_i32_f32) -> i32:
    return u.int_val

@compile
def pc_sum_union_large(u: Union_Large32) -> i64:
    return u.arr[0] + u.arr[1] + u.arr[2] + u.arr[3]

@compile
def pc_double_union_int(u: Union_i32_f32) -> Union_i32_f32:
    result: Union_i32_f32 = Union_i32_f32()
    result.int_val = u.int_val * 2
    return result

@compile
def pc_increment_union_large(u: Union_Large32) -> Union_Large32:
    result: Union_Large32 = Union_Large32()
    result.arr[0] = u.arr[0] + 1
    result.arr[1] = u.arr[1] + 1
    result.arr[2] = u.arr[2] + 1
    result.arr[3] = u.arr[3] + 1
    return result

# Enum functions
@compile
def pc_return_result_ok() -> Result_i32:
    return Result_i32(Result_i32.Ok, 42)

@compile
def pc_return_expr_add() -> Expression_Enum:
    binop: BinOp = BinOp()
    binop.a = 100
    binop.b = 200
    return Expression_Enum(Expression_Enum.Add, binop)

@compile
def pc_get_result_value(r: Result_i32) -> i32:
    match r:
        case (Result_i32.Ok, val):
            return val
        case (Result_i32.Err, val):
            return 0 - val

@compile
def pc_eval_expression(e: Expression_Enum) -> i64:
    match e:
        case (Expression_Enum.Const, val):
            return val
        case (Expression_Enum.Add, binop):
            return binop.a + binop.b
        case (Expression_Enum.Mul, binop):
            return binop.a * binop.b

@compile
def pc_double_result(r: Result_i32) -> Result_i32:
    match r:
        case (Result_i32.Ok, val):
            return Result_i32(Result_i32.Ok, val * 2)
        case (Result_i32.Err, val):
            return Result_i32(Result_i32.Err, val * 2)

@compile
def pc_negate_expr(e: Expression_Enum) -> Expression_Enum:
    match e:
        case (Expression_Enum.Const, val):
            return Expression_Enum(Expression_Enum.Const, 0 - val)
        case (Expression_Enum.Add, binop):
            neg_binop: BinOp = BinOp()
            neg_binop.a = 0 - binop.a
            neg_binop.b = 0 - binop.b
            return Expression_Enum(Expression_Enum.Add, neg_binop)
        case (Expression_Enum.Mul, binop):
            neg_binop2: BinOp = BinOp()
            neg_binop2.a = 0 - binop.a
            neg_binop2.b = 0 - binop.b
            return Expression_Enum(Expression_Enum.Mul, neg_binop2)


# =============================================================================
# pythoc functions that call C functions
# =============================================================================

# Case 4: C function returns type to pythoc
@compile
def test_c_return_array2_i32() -> i32:
    arr: Array2_i32 = c_return_array2_i32()
    return arr.data[0] + arr.data[1]  # 10 + 20 = 30

@compile
def test_c_return_array4_i64() -> i64:
    arr: Array4_i64 = c_return_array4_i64()
    return arr.data[0] + arr.data[1] + arr.data[2] + arr.data[3]  # 100+200+300+400 = 1000

@compile
def test_c_return_union_int() -> i32:
    u: Union_i32_f32 = c_return_union_int()
    return u.int_val  # 42

@compile
def test_c_return_union_large() -> i64:
    u: Union_Large32 = c_return_union_large()
    return u.arr[0] + u.arr[1] + u.arr[2] + u.arr[3]  # 10+20+30+40 = 100

@compile
def test_c_return_result_ok() -> i32:
    r: Result_i32 = c_return_result_ok(42)
    match r:
        case (Result_i32.Ok, val):
            return val
        case _:
            return -1

@compile
def test_c_return_expr_add() -> i64:
    e: Expression_Enum = c_return_expr_add(100, 200)
    match e:
        case (Expression_Enum.Add, binop):
            return binop.a + binop.b
        case _:
            return -1

# Case 1: pythoc type as C function param
@compile
def test_c_sum_array2_i32() -> i32:
    arr: Array2_i32 = Array2_i32()
    arr.data[0] = 50
    arr.data[1] = 75
    return c_sum_array2_i32(arr)  # 125

@compile
def test_c_sum_array4_i64() -> i64:
    arr: Array4_i64 = Array4_i64()
    arr.data[0] = 10
    arr.data[1] = 20
    arr.data[2] = 30
    arr.data[3] = 40
    return c_sum_array4_i64(arr)  # 100

@compile
def test_c_get_union_int() -> i32:
    u: Union_i32_f32 = Union_i32_f32()
    u.int_val = 123
    return c_get_union_int(u)  # 123

@compile
def test_c_sum_union_large() -> i64:
    u: Union_Large32 = Union_Large32()
    u.arr[0] = 10
    u.arr[1] = 20
    u.arr[2] = 30
    u.arr[3] = 40
    return c_sum_union_large(u)  # 100

@compile
def test_c_get_result_value() -> i32:
    r: Result_i32 = Result_i32(Result_i32.Ok, 77)
    return c_get_result_value(r)  # 77

@compile
def test_c_eval_expression() -> i64:
    binop: BinOp = BinOp()
    binop.a = 100
    binop.b = 200
    e: Expression_Enum = Expression_Enum(Expression_Enum.Add, binop)
    return c_eval_expression(e)  # 300

# Case 1+4 combined: take and return
@compile
def test_c_double_array2_i32() -> i32:
    arr: Array2_i32 = Array2_i32()
    arr.data[0] = 10
    arr.data[1] = 20
    result: Array2_i32 = c_double_array2_i32(arr)
    return result.data[0] + result.data[1]  # 20 + 40 = 60

@compile
def test_c_increment_array4_i64() -> i64:
    arr: Array4_i64 = Array4_i64()
    arr.data[0] = 10
    arr.data[1] = 20
    arr.data[2] = 30
    arr.data[3] = 40
    result: Array4_i64 = c_increment_array4_i64(arr)
    return result.data[0] + result.data[1] + result.data[2] + result.data[3]  # 11+21+31+41 = 104

@compile
def test_c_double_union_int() -> i32:
    u: Union_i32_f32 = Union_i32_f32()
    u.int_val = 25
    result: Union_i32_f32 = c_double_union_int(u)
    return result.int_val  # 50

@compile
def test_c_increment_union_large() -> i64:
    u: Union_Large32 = Union_Large32()
    u.arr[0] = 10
    u.arr[1] = 20
    u.arr[2] = 30
    u.arr[3] = 40
    result: Union_Large32 = c_increment_union_large(u)
    return result.arr[0] + result.arr[1] + result.arr[2] + result.arr[3]  # 11+21+31+41 = 104

@compile
def test_c_double_result() -> i32:
    r: Result_i32 = Result_i32(Result_i32.Ok, 25)
    result: Result_i32 = c_double_result(r)
    match result:
        case (Result_i32.Ok, val):
            return val
        case _:
            return -1

@compile
def test_c_negate_expr() -> i64:
    binop: BinOp = BinOp()
    binop.a = 10
    binop.b = 20
    e: Expression_Enum = Expression_Enum(Expression_Enum.Add, binop)
    result: Expression_Enum = c_negate_expr(e)
    match result:
        case (Expression_Enum.Add, result_binop):
            return result_binop.a + result_binop.b
        case _:
            return 0


# =============================================================================
# Test Cases
# =============================================================================

class TestPythocCallsC_Array(unittest.TestCase):
    """Test pythoc calling C functions with array types"""
    
    @classmethod
    def setUpClass(cls):
        get_c_library()
    
    def test_c_return_array2_i32(self):
        """Case 4: C returns array to pythoc"""
        self.assertEqual(test_c_return_array2_i32(), 30)
    
    def test_c_return_array4_i64(self):
        """Case 4: C returns large array to pythoc"""
        self.assertEqual(test_c_return_array4_i64(), 1000)
    
    def test_c_sum_array2_i32(self):
        """Case 1: pythoc array as C param"""
        self.assertEqual(test_c_sum_array2_i32(), 125)
    
    def test_c_sum_array4_i64(self):
        """Case 1: pythoc large array as C param"""
        self.assertEqual(test_c_sum_array4_i64(), 100)
    
    def test_c_double_array2_i32(self):
        """Case 1+4: pythoc array param, C returns array"""
        self.assertEqual(test_c_double_array2_i32(), 60)
    
    def test_c_increment_array4_i64(self):
        """Case 1+4: pythoc large array param, C returns array"""
        self.assertEqual(test_c_increment_array4_i64(), 104)


class TestPythocCallsC_Union(unittest.TestCase):
    """Test pythoc calling C functions with union types"""
    
    @classmethod
    def setUpClass(cls):
        get_c_library()
    
    def test_c_return_union_int(self):
        """Case 4: C returns union to pythoc"""
        self.assertEqual(test_c_return_union_int(), 42)
    
    def test_c_return_union_large(self):
        """Case 4: C returns large union to pythoc"""
        self.assertEqual(test_c_return_union_large(), 100)
    
    def test_c_get_union_int(self):
        """Case 1: pythoc union as C param"""
        self.assertEqual(test_c_get_union_int(), 123)
    
    def test_c_sum_union_large(self):
        """Case 1: pythoc large union as C param"""
        self.assertEqual(test_c_sum_union_large(), 100)
    
    def test_c_double_union_int(self):
        """Case 1+4: pythoc union param, C returns union"""
        self.assertEqual(test_c_double_union_int(), 50)
    
    def test_c_increment_union_large(self):
        """Case 1+4: pythoc large union param, C returns union"""
        self.assertEqual(test_c_increment_union_large(), 104)


class TestPythocCallsC_Enum(unittest.TestCase):
    """Test pythoc calling C functions with enum types"""
    
    @classmethod
    def setUpClass(cls):
        get_c_library()
    
    def test_c_return_result_ok(self):
        """Case 4: C returns enum to pythoc"""
        self.assertEqual(test_c_return_result_ok(), 42)
    
    def test_c_return_expr_add(self):
        """Case 4: C returns large enum to pythoc"""
        self.assertEqual(test_c_return_expr_add(), 300)
    
    def test_c_get_result_value(self):
        """Case 1: pythoc enum as C param"""
        self.assertEqual(test_c_get_result_value(), 77)
    
    def test_c_eval_expression(self):
        """Case 1: pythoc large enum as C param"""
        self.assertEqual(test_c_eval_expression(), 300)
    
    def test_c_double_result(self):
        """Case 1+4: pythoc enum param, C returns enum"""
        self.assertEqual(test_c_double_result(), 50)
    
    def test_c_negate_expr(self):
        """Case 1+4: pythoc large enum param, C returns enum"""
        self.assertEqual(test_c_negate_expr(), -30)


class TestCCallsPythoc_Array(unittest.TestCase):
    """Test C calling pythoc functions with array types"""
    
    @classmethod
    def setUpClass(cls):
        cls.c_lib = get_c_library()
    
    def _get_pythoc_func_ptr(self, pc_func):
        """Get ctypes function pointer for a pythoc function"""
        from pythoc.native_executor import get_multi_so_executor
        from pythoc.build.output_manager import flush_all_pending_outputs
        executor = get_multi_so_executor()
        flush_all_pending_outputs()
        
        so_file = pc_func._so_file
        if so_file not in executor.loaded_libs:
            lib = ctypes.CDLL(so_file)
            executor.loaded_libs[so_file] = lib
        
        lib = executor.loaded_libs[so_file]
        func_name = pc_func._actual_func_name
        native_func = getattr(lib, func_name)
        return ctypes.cast(native_func, ctypes.c_void_p).value
    
    def test_c_calls_pc_return_array2_i32(self):
        """Case 2: pythoc returns array to C"""
        fn_ptr = self._get_pythoc_func_ptr(pc_return_array2_i32)
        self.c_lib.c_call_fn_return_array2_i32.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_fn_return_array2_i32.restype = ctypes.c_int32
        result = self.c_lib.c_call_fn_return_array2_i32(fn_ptr)
        self.assertEqual(result, 300)  # 100 + 200
    
    def test_c_calls_pc_return_array4_i64(self):
        """Case 2: pythoc returns large array to C"""
        fn_ptr = self._get_pythoc_func_ptr(pc_return_array4_i64)
        self.c_lib.c_call_fn_return_array4_i64.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_fn_return_array4_i64.restype = ctypes.c_int64
        result = self.c_lib.c_call_fn_return_array4_i64(fn_ptr)
        self.assertEqual(result, 100)  # 10+20+30+40
    
    def test_c_calls_pc_sum_array2_i32(self):
        """Case 3: C array as pythoc param"""
        fn_ptr = self._get_pythoc_func_ptr(pc_sum_array2_i32)
        self.c_lib.c_call_fn_sum_array2_i32.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_fn_sum_array2_i32.restype = ctypes.c_int32
        result = self.c_lib.c_call_fn_sum_array2_i32(fn_ptr)
        self.assertEqual(result, 100)  # 25 + 75
    
    def test_c_calls_pc_sum_array4_i64(self):
        """Case 3: C large array as pythoc param"""
        fn_ptr = self._get_pythoc_func_ptr(pc_sum_array4_i64)
        self.c_lib.c_call_fn_sum_array4_i64.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_fn_sum_array4_i64.restype = ctypes.c_int64
        result = self.c_lib.c_call_fn_sum_array4_i64(fn_ptr)
        self.assertEqual(result, 1000)  # 100+200+300+400
    
    def test_c_calls_pc_double_array2_i32(self):
        """Case 2+3: C array param, pythoc returns array"""
        fn_ptr = self._get_pythoc_func_ptr(pc_double_array2_i32)
        self.c_lib.c_call_fn_double_array2_i32.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_fn_double_array2_i32.restype = ctypes.c_int32
        result = self.c_lib.c_call_fn_double_array2_i32(fn_ptr)
        self.assertEqual(result, 60)  # (10*2) + (20*2)
    
    def test_c_calls_pc_increment_array4_i64(self):
        """Case 2+3: C large array param, pythoc returns array"""
        fn_ptr = self._get_pythoc_func_ptr(pc_increment_array4_i64)
        self.c_lib.c_call_fn_increment_array4_i64.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_fn_increment_array4_i64.restype = ctypes.c_int64
        result = self.c_lib.c_call_fn_increment_array4_i64(fn_ptr)
        self.assertEqual(result, 104)  # 11+21+31+41


class TestCCallsPythoc_Union(unittest.TestCase):
    """Test C calling pythoc functions with union types"""
    
    @classmethod
    def setUpClass(cls):
        cls.c_lib = get_c_library()
    
    def _get_pythoc_func_ptr(self, pc_func):
        """Get ctypes function pointer for a pythoc function"""
        from pythoc.native_executor import get_multi_so_executor
        from pythoc.build.output_manager import flush_all_pending_outputs
        executor = get_multi_so_executor()
        flush_all_pending_outputs()
        
        so_file = pc_func._so_file
        if so_file not in executor.loaded_libs:
            lib = ctypes.CDLL(so_file)
            executor.loaded_libs[so_file] = lib
        
        lib = executor.loaded_libs[so_file]
        func_name = pc_func._actual_func_name
        native_func = getattr(lib, func_name)
        return ctypes.cast(native_func, ctypes.c_void_p).value
    
    def test_c_calls_pc_return_union_i32_f32(self):
        """Case 2: pythoc returns union to C"""
        fn_ptr = self._get_pythoc_func_ptr(pc_return_union_i32_f32)
        self.c_lib.c_call_fn_return_union_i32_f32.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_fn_return_union_i32_f32.restype = ctypes.c_int32
        result = self.c_lib.c_call_fn_return_union_i32_f32(fn_ptr)
        self.assertEqual(result, 42)
    
    def test_c_calls_pc_return_union_large(self):
        """Case 2: pythoc returns large union to C"""
        fn_ptr = self._get_pythoc_func_ptr(pc_return_union_large)
        self.c_lib.c_call_fn_return_union_large.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_fn_return_union_large.restype = ctypes.c_int64
        result = self.c_lib.c_call_fn_return_union_large(fn_ptr)
        self.assertEqual(result, 100)  # 10+20+30+40
    
    def test_c_calls_pc_get_union_int(self):
        """Case 3: C union as pythoc param"""
        fn_ptr = self._get_pythoc_func_ptr(pc_get_union_int)
        self.c_lib.c_call_fn_get_union_int.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_fn_get_union_int.restype = ctypes.c_int32
        result = self.c_lib.c_call_fn_get_union_int(fn_ptr)
        self.assertEqual(result, 123)
    
    def test_c_calls_pc_sum_union_large(self):
        """Case 3: C large union as pythoc param"""
        fn_ptr = self._get_pythoc_func_ptr(pc_sum_union_large)
        self.c_lib.c_call_fn_sum_union_large.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_fn_sum_union_large.restype = ctypes.c_int64
        result = self.c_lib.c_call_fn_sum_union_large(fn_ptr)
        self.assertEqual(result, 100)  # 10+20+30+40
    
    def test_c_calls_pc_double_union_int(self):
        """Case 2+3: C union param, pythoc returns union"""
        fn_ptr = self._get_pythoc_func_ptr(pc_double_union_int)
        self.c_lib.c_call_fn_double_union_int.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_fn_double_union_int.restype = ctypes.c_int32
        result = self.c_lib.c_call_fn_double_union_int(fn_ptr)
        self.assertEqual(result, 100)  # 50 * 2
    
    def test_c_calls_pc_increment_union_large(self):
        """Case 2+3: C large union param, pythoc returns union"""
        fn_ptr = self._get_pythoc_func_ptr(pc_increment_union_large)
        self.c_lib.c_call_fn_increment_union_large.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_fn_increment_union_large.restype = ctypes.c_int64
        result = self.c_lib.c_call_fn_increment_union_large(fn_ptr)
        self.assertEqual(result, 104)  # 11+21+31+41


class TestCCallsPythoc_Enum(unittest.TestCase):
    """Test C calling pythoc functions with enum types"""
    
    @classmethod
    def setUpClass(cls):
        cls.c_lib = get_c_library()
    
    def _get_pythoc_func_ptr(self, pc_func):
        """Get ctypes function pointer for a pythoc function"""
        from pythoc.native_executor import get_multi_so_executor
        from pythoc.build.output_manager import flush_all_pending_outputs
        executor = get_multi_so_executor()
        flush_all_pending_outputs()
        
        so_file = pc_func._so_file
        if so_file not in executor.loaded_libs:
            lib = ctypes.CDLL(so_file)
            executor.loaded_libs[so_file] = lib
        
        lib = executor.loaded_libs[so_file]
        func_name = pc_func._actual_func_name
        native_func = getattr(lib, func_name)
        return ctypes.cast(native_func, ctypes.c_void_p).value
    
    def test_c_calls_pc_return_result(self):
        """Case 2: pythoc returns enum to C"""
        fn_ptr = self._get_pythoc_func_ptr(pc_return_result_ok)
        self.c_lib.c_call_fn_return_result.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_fn_return_result.restype = ctypes.c_int32
        result = self.c_lib.c_call_fn_return_result(fn_ptr)
        self.assertEqual(result, 42)
    
    def test_c_calls_pc_return_expr(self):
        """Case 2: pythoc returns large enum to C"""
        fn_ptr = self._get_pythoc_func_ptr(pc_return_expr_add)
        self.c_lib.c_call_fn_return_expr.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_fn_return_expr.restype = ctypes.c_int64
        result = self.c_lib.c_call_fn_return_expr(fn_ptr)
        self.assertEqual(result, 300)  # 100 + 200
    
    def test_c_calls_pc_get_result(self):
        """Case 3: C enum as pythoc param"""
        fn_ptr = self._get_pythoc_func_ptr(pc_get_result_value)
        self.c_lib.c_call_fn_get_result.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_fn_get_result.restype = ctypes.c_int32
        result = self.c_lib.c_call_fn_get_result(fn_ptr)
        self.assertEqual(result, 77)
    
    def test_c_calls_pc_eval_expr(self):
        """Case 3: C large enum as pythoc param"""
        fn_ptr = self._get_pythoc_func_ptr(pc_eval_expression)
        self.c_lib.c_call_fn_eval_expr.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_fn_eval_expr.restype = ctypes.c_int64
        result = self.c_lib.c_call_fn_eval_expr(fn_ptr)
        self.assertEqual(result, 300)  # 100 + 200
    
    def test_c_calls_pc_double_result(self):
        """Case 2+3: C enum param, pythoc returns enum"""
        fn_ptr = self._get_pythoc_func_ptr(pc_double_result)
        self.c_lib.c_call_fn_double_result.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_fn_double_result.restype = ctypes.c_int32
        result = self.c_lib.c_call_fn_double_result(fn_ptr)
        self.assertEqual(result, 50)  # 25 * 2
    
    def test_c_calls_pc_negate_expr(self):
        """Case 2+3: C large enum param, pythoc returns enum"""
        fn_ptr = self._get_pythoc_func_ptr(pc_negate_expr)
        self.c_lib.c_call_fn_negate_expr.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_fn_negate_expr.restype = ctypes.c_int64
        result = self.c_lib.c_call_fn_negate_expr(fn_ptr)
        self.assertEqual(result, -30)  # -10 + -20


if __name__ == '__main__':
    unittest.main()
