#!/usr/bin/env python3
"""
Comprehensive C ABI tests for struct return and parameter passing.

This test ACTUALLY tests C interop by:
1. Compiling C code to a shared library
2. Using @extern to call C functions from pythoc
3. Getting pythoc function pointers and calling them from C

Covers different struct sizes that trigger different ABI handling:
- Small structs (<=8 bytes): coerced to i64
- Medium structs (<=16 bytes): coerced to (i64, i64)
- Large structs (>16 bytes): sret (indirect return via pointer)
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

// Struct definitions matching pythoc
typedef struct { int8_t a; } Small1;
typedef struct { int16_t a; } Small2;
typedef struct { int32_t a; } Small4;
typedef struct { int32_t a; int32_t b; } Small8;
typedef struct { int64_t a; } Small8_i64;
typedef struct { double a; } Small8_f64;

typedef struct { int32_t a; int32_t b; int32_t c; } Medium12;
typedef struct { int64_t a; int64_t b; } Medium16;
typedef struct { double a; double b; } Medium16_f64;

typedef struct { int64_t a; int64_t b; int64_t c; } Large24;
typedef struct { int64_t a; int64_t b; int64_t c; int64_t d; } Large32;
typedef struct { int64_t a; int64_t b; int64_t c; int64_t d; int64_t e; int64_t f; } Large48;
typedef struct { int64_t a; int64_t b; int64_t c; int64_t d; int64_t e; int64_t f; int64_t g; int64_t h; } Large64;

// ============ C functions that return structs ============

Small1 c_return_small1(void) {
    Small1 s = {42};
    return s;
}

Small4 c_return_small4(void) {
    Small4 s = {12345};
    return s;
}

Small8 c_return_small8(void) {
    Small8 s = {100, 200};
    return s;
}

Medium12 c_return_medium12(void) {
    Medium12 s = {10, 20, 30};
    return s;
}

Medium16 c_return_medium16(void) {
    Medium16 s = {1000, 2000};
    return s;
}

Large24 c_return_large24(void) {
    Large24 s = {111, 222, 333};
    return s;
}

Large32 c_return_large32(void) {
    Large32 s = {1, 2, 3, 4};
    return s;
}

Large48 c_return_large48(void) {
    Large48 s = {10, 20, 30, 40, 50, 60};
    return s;
}

Large64 c_return_large64(void) {
    Large64 s = {1, 2, 3, 4, 5, 6, 7, 8};
    return s;
}

// ============ C functions that take struct parameters ============

int32_t c_sum_small8(Small8 s) {
    return s.a + s.b;
}

int64_t c_sum_medium16(Medium16 s) {
    return s.a + s.b;
}

int64_t c_sum_large24(Large24 s) {
    return s.a + s.b + s.c;
}

int64_t c_sum_large48(Large48 s) {
    return s.a + s.b + s.c + s.d + s.e + s.f;
}

int64_t c_sum_large64(Large64 s) {
    return s.a + s.b + s.c + s.d + s.e + s.f + s.g + s.h;
}

// ============ C functions that take and return structs ============

Small8 c_double_small8(Small8 s) {
    Small8 r = {s.a * 2, s.b * 2};
    return r;
}

Large24 c_increment_large24(Large24 s) {
    Large24 r = {s.a + 1, s.b + 1, s.c + 1};
    return r;
}

Large48 c_increment_large48(Large48 s) {
    Large48 r = {s.a + 1, s.b + 1, s.c + 1, s.d + 1, s.e + 1, s.f + 1};
    return r;
}

Large64 c_increment_large64(Large64 s) {
    Large64 r = {s.a + 1, s.b + 1, s.c + 1, s.d + 1, s.e + 1, s.f + 1, s.g + 1, s.h + 1};
    return r;
}

// ============ C functions that call pythoc via function pointers ============

// Function pointer types
typedef Small8 (*fn_return_small8_t)(void);
typedef Large24 (*fn_return_large24_t)(void);
typedef int32_t (*fn_sum_small8_t)(Small8);
typedef int64_t (*fn_sum_large24_t)(Large24);
typedef Small8 (*fn_double_small8_t)(Small8);
typedef Large24 (*fn_increment_large24_t)(Large24);

// C functions that call pythoc functions via function pointers
int32_t c_call_fn_return_small8(fn_return_small8_t fn) {
    Small8 s = fn();
    return s.a + s.b;
}

int64_t c_call_fn_return_large24(fn_return_large24_t fn) {
    Large24 s = fn();
    return s.a + s.b + s.c;
}

int32_t c_call_fn_sum_small8(fn_sum_small8_t fn) {
    Small8 s = {25, 75};
    return fn(s);
}

int64_t c_call_fn_sum_large24(fn_sum_large24_t fn) {
    Large24 s = {100, 200, 300};
    return fn(s);
}

int32_t c_call_fn_double_small8(fn_double_small8_t fn) {
    Small8 s = {10, 20};
    Small8 r = fn(s);
    return r.a + r.b;
}

int64_t c_call_fn_increment_large24(fn_increment_large24_t fn) {
    Large24 s = {100, 200, 300};
    Large24 r = fn(s);
    return r.a + r.b + r.c;
}

// Large48 and Large64 function pointer types and callers
typedef Large48 (*fn_return_large48_t)(void);
typedef Large64 (*fn_return_large64_t)(void);
typedef int64_t (*fn_sum_large48_t)(Large48);
typedef int64_t (*fn_sum_large64_t)(Large64);
typedef Large48 (*fn_increment_large48_t)(Large48);
typedef Large64 (*fn_increment_large64_t)(Large64);

int64_t c_call_fn_return_large48(fn_return_large48_t fn) {
    Large48 s = fn();
    return s.a + s.b + s.c + s.d + s.e + s.f;
}

int64_t c_call_fn_return_large64(fn_return_large64_t fn) {
    Large64 s = fn();
    return s.a + s.b + s.c + s.d + s.e + s.f + s.g + s.h;
}

int64_t c_call_fn_sum_large48(fn_sum_large48_t fn) {
    Large48 s = {10, 20, 30, 40, 50, 60};
    return fn(s);
}

int64_t c_call_fn_sum_large64(fn_sum_large64_t fn) {
    Large64 s = {1, 2, 3, 4, 5, 6, 7, 8};
    return fn(s);
}

int64_t c_call_fn_increment_large48(fn_increment_large48_t fn) {
    Large48 s = {10, 20, 30, 40, 50, 60};
    Large48 r = fn(s);
    return r.a + r.b + r.c + r.d + r.e + r.f;
}

int64_t c_call_fn_increment_large64(fn_increment_large64_t fn) {
    Large64 s = {1, 2, 3, 4, 5, 6, 7, 8};
    Large64 r = fn(s);
    return r.a + r.b + r.c + r.d + r.e + r.f + r.g + r.h;
}
"""

# =============================================================================
# Compile C library (must happen before @extern declarations)
# =============================================================================

_c_lib = None
_c_lib_path = None

def _compile_c_library():
    """Compile C library and return path"""
    global _c_lib, _c_lib_path
    
    if _c_lib_path is not None:
        return _c_lib_path
    
    # Write C source to temp file
    c_file = os.path.join(tempfile.gettempdir(), 'test_c_abi.c')
    so_file = os.path.join(tempfile.gettempdir(), 'libtest_c_abi.so')
    
    with open(c_file, 'w') as f:
        f.write(C_SOURCE)
    
    # Compile to shared library
    result = subprocess.run(
        ['gcc', '-shared', '-fPIC', '-O2', '-o', so_file, c_file],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"C compilation failed: {result.stderr}")
    
    _c_lib_path = so_file
    return so_file

def get_c_library():
    """Get loaded C library (compile if needed)"""
    global _c_lib
    if _c_lib is None:
        _compile_c_library()
        _c_lib = ctypes.CDLL(_c_lib_path)
    return _c_lib

# Compile C library NOW before @extern declarations
C_LIB_PATH = _compile_c_library()


# =============================================================================
# pythoc struct definitions
# =============================================================================

from pythoc import (
    compile, struct, extern,
    i8, i16, i32, i64,
    u8, u16, u32, u64,
    f32, f64,
    bool as pc_bool,
    ptr, func, void
)

# Small structs (<=8 bytes) - i64 coercion
@struct
class Small1:
    a: i8

@struct
class Small4:
    a: i32

@struct
class Small8:
    a: i32
    b: i32

# Medium structs (9-16 bytes) - (i64, i64) coercion
@struct
class Medium12:
    a: i32
    b: i32
    c: i32

@struct
class Medium16:
    a: i64
    b: i64

# Large structs (>16 bytes) - sret
@struct
class Large24:
    a: i64
    b: i64
    c: i64

@struct
class Large32:
    a: i64
    b: i64
    c: i64
    d: i64

@struct
class Large48:
    a: i64
    b: i64
    c: i64
    d: i64
    e: i64
    f: i64

@struct
class Large64:
    a: i64
    b: i64
    c: i64
    d: i64
    e: i64
    f: i64
    g: i64
    h: i64


# =============================================================================
# Part 1: @extern declarations for C functions
# =============================================================================

# C functions returning structs
@extern(lib=C_LIB_PATH)
def c_return_small1() -> Small1: ...

@extern(lib=C_LIB_PATH)
def c_return_small4() -> Small4: ...

@extern(lib=C_LIB_PATH)
def c_return_small8() -> Small8: ...

@extern(lib=C_LIB_PATH)
def c_return_medium12() -> Medium12: ...

@extern(lib=C_LIB_PATH)
def c_return_medium16() -> Medium16: ...

@extern(lib=C_LIB_PATH)
def c_return_large24() -> Large24: ...

@extern(lib=C_LIB_PATH)
def c_return_large32() -> Large32: ...

@extern(lib=C_LIB_PATH)
def c_return_large48() -> Large48: ...

@extern(lib=C_LIB_PATH)
def c_return_large64() -> Large64: ...

# C functions taking struct parameters
@extern(lib=C_LIB_PATH)
def c_sum_small8(s: Small8) -> i32: ...

@extern(lib=C_LIB_PATH)
def c_sum_medium16(s: Medium16) -> i64: ...

@extern(lib=C_LIB_PATH)
def c_sum_large24(s: Large24) -> i64: ...

@extern(lib=C_LIB_PATH)
def c_sum_large48(s: Large48) -> i64: ...

@extern(lib=C_LIB_PATH)
def c_sum_large64(s: Large64) -> i64: ...

# C functions taking and returning structs
@extern(lib=C_LIB_PATH)
def c_double_small8(s: Small8) -> Small8: ...

@extern(lib=C_LIB_PATH)
def c_increment_large24(s: Large24) -> Large24: ...

@extern(lib=C_LIB_PATH)
def c_increment_large48(s: Large48) -> Large48: ...

@extern(lib=C_LIB_PATH)
def c_increment_large64(s: Large64) -> Large64: ...

# C functions that call function pointers
@extern(lib=C_LIB_PATH)
def c_call_fn_return_small8(fn: ptr[func[Small8]]) -> i32: ...

@extern(lib=C_LIB_PATH)
def c_call_fn_return_large24(fn: ptr[func[Large24]]) -> i64: ...

@extern(lib=C_LIB_PATH)
def c_call_fn_return_large48(fn: ptr[func[Large48]]) -> i64: ...

@extern(lib=C_LIB_PATH)
def c_call_fn_return_large64(fn: ptr[func[Large64]]) -> i64: ...

@extern(lib=C_LIB_PATH)
def c_call_fn_sum_small8(fn: ptr[func[Small8, i32]]) -> i32: ...

@extern(lib=C_LIB_PATH)
def c_call_fn_sum_large24(fn: ptr[func[Large24, i64]]) -> i64: ...

@extern(lib=C_LIB_PATH)
def c_call_fn_sum_large48(fn: ptr[func[Large48, i64]]) -> i64: ...

@extern(lib=C_LIB_PATH)
def c_call_fn_sum_large64(fn: ptr[func[Large64, i64]]) -> i64: ...

@extern(lib=C_LIB_PATH)
def c_call_fn_double_small8(fn: ptr[func[Small8, Small8]]) -> i32: ...

@extern(lib=C_LIB_PATH)
def c_call_fn_increment_large24(fn: ptr[func[Large24, Large24]]) -> i64: ...

@extern(lib=C_LIB_PATH)
def c_call_fn_increment_large48(fn: ptr[func[Large48, Large48]]) -> i64: ...

@extern(lib=C_LIB_PATH)
def c_call_fn_increment_large64(fn: ptr[func[Large64, Large64]]) -> i64: ...


# =============================================================================
# Part 2: pythoc functions (to be called from C)
# =============================================================================

@compile
def pc_return_small8() -> Small8:
    """pythoc function returning 8-byte struct"""
    s: Small8 = Small8()
    s.a = 100
    s.b = 200
    return s

@compile
def pc_return_large24() -> Large24:
    """pythoc function returning 24-byte struct"""
    s: Large24 = Large24()
    s.a = 111
    s.b = 222
    s.c = 333
    return s

@compile
def pc_return_large48() -> Large48:
    """pythoc function returning 48-byte struct"""
    s: Large48 = Large48()
    s.a = 10
    s.b = 20
    s.c = 30
    s.d = 40
    s.e = 50
    s.f = 60
    return s

@compile
def pc_return_large64() -> Large64:
    """pythoc function returning 64-byte struct"""
    s: Large64 = Large64()
    s.a = 1
    s.b = 2
    s.c = 3
    s.d = 4
    s.e = 5
    s.f = 6
    s.g = 7
    s.h = 8
    return s

@compile
def pc_sum_small8(s: Small8) -> i32:
    """pythoc function taking 8-byte struct"""
    return s.a + s.b

@compile
def pc_sum_large24(s: Large24) -> i64:
    """pythoc function taking 24-byte struct"""
    return s.a + s.b + s.c

@compile
def pc_sum_large48(s: Large48) -> i64:
    """pythoc function taking 48-byte struct"""
    return s.a + s.b + s.c + s.d + s.e + s.f

@compile
def pc_sum_large64(s: Large64) -> i64:
    """pythoc function taking 64-byte struct"""
    return s.a + s.b + s.c + s.d + s.e + s.f + s.g + s.h

@compile
def pc_double_small8(s: Small8) -> Small8:
    """pythoc function taking and returning 8-byte struct"""
    r: Small8 = Small8()
    r.a = s.a * 2
    r.b = s.b * 2
    return r

@compile
def pc_increment_large24(s: Large24) -> Large24:
    """pythoc function taking and returning 24-byte struct"""
    r: Large24 = Large24()
    r.a = s.a + 1
    r.b = s.b + 1
    r.c = s.c + 1
    return r

@compile
def pc_increment_large48(s: Large48) -> Large48:
    """pythoc function taking and returning 48-byte struct"""
    r: Large48 = Large48()
    r.a = s.a + 1
    r.b = s.b + 1
    r.c = s.c + 1
    r.d = s.d + 1
    r.e = s.e + 1
    r.f = s.f + 1
    return r

@compile
def pc_increment_large64(s: Large64) -> Large64:
    """pythoc function taking and returning 64-byte struct"""
    r: Large64 = Large64()
    r.a = s.a + 1
    r.b = s.b + 1
    r.c = s.c + 1
    r.d = s.d + 1
    r.e = s.e + 1
    r.f = s.f + 1
    r.g = s.g + 1
    r.h = s.h + 1
    return r


# =============================================================================
# Part 3: pythoc functions that call C functions
# =============================================================================

@compile
def test_call_c_return_small8() -> i32:
    """Call C function returning 8-byte struct"""
    s: Small8 = c_return_small8()
    return s.a + s.b  # Expected: 100 + 200 = 300

@compile
def test_call_c_return_medium16() -> i64:
    """Call C function returning 16-byte struct"""
    s: Medium16 = c_return_medium16()
    return s.a + s.b  # Expected: 1000 + 2000 = 3000

@compile
def test_call_c_return_large24() -> i64:
    """Call C function returning 24-byte struct"""
    s: Large24 = c_return_large24()
    return s.a + s.b + s.c  # Expected: 111 + 222 + 333 = 666

@compile
def test_call_c_return_large48() -> i64:
    """Call C function returning 48-byte struct"""
    s: Large48 = c_return_large48()
    return s.a + s.b + s.c + s.d + s.e + s.f  # Expected: 10+20+30+40+50+60 = 210

@compile
def test_call_c_return_large64() -> i64:
    """Call C function returning 64-byte struct"""
    s: Large64 = c_return_large64()
    return s.a + s.b + s.c + s.d + s.e + s.f + s.g + s.h  # Expected: 1+2+3+4+5+6+7+8 = 36

@compile
def test_call_c_sum_small8() -> i32:
    """Call C function with 8-byte struct param"""
    s: Small8 = Small8()
    s.a = 50
    s.b = 75
    return c_sum_small8(s)  # Expected: 125

@compile
def test_call_c_sum_large24() -> i64:
    """Call C function with 24-byte struct param"""
    s: Large24 = Large24()
    s.a = 100
    s.b = 200
    s.c = 300
    return c_sum_large24(s)  # Expected: 600

@compile
def test_call_c_sum_large48() -> i64:
    """Call C function with 48-byte struct param"""
    s: Large48 = Large48()
    s.a = 10
    s.b = 20
    s.c = 30
    s.d = 40
    s.e = 50
    s.f = 60
    return c_sum_large48(s)  # Expected: 210

@compile
def test_call_c_sum_large64() -> i64:
    """Call C function with 64-byte struct param"""
    s: Large64 = Large64()
    s.a = 1
    s.b = 2
    s.c = 3
    s.d = 4
    s.e = 5
    s.f = 6
    s.g = 7
    s.h = 8
    return c_sum_large64(s)  # Expected: 36

@compile
def test_call_c_double_small8() -> i32:
    """Call C function that takes and returns 8-byte struct"""
    s: Small8 = Small8()
    s.a = 10
    s.b = 20
    r: Small8 = c_double_small8(s)
    return r.a + r.b  # Expected: 20 + 40 = 60

@compile
def test_call_c_increment_large24() -> i64:
    """Call C function that takes and returns 24-byte struct"""
    s: Large24 = Large24()
    s.a = 100
    s.b = 200
    s.c = 300
    r: Large24 = c_increment_large24(s)
    return r.a + r.b + r.c  # Expected: 101 + 201 + 301 = 603

@compile
def test_call_c_increment_large48() -> i64:
    """Call C function that takes and returns 48-byte struct"""
    s: Large48 = Large48()
    s.a = 10
    s.b = 20
    s.c = 30
    s.d = 40
    s.e = 50
    s.f = 60
    r: Large48 = c_increment_large48(s)
    return r.a + r.b + r.c + r.d + r.e + r.f  # Expected: 11+21+31+41+51+61 = 216

@compile
def test_call_c_increment_large64() -> i64:
    """Call C function that takes and returns 64-byte struct"""
    s: Large64 = Large64()
    s.a = 1
    s.b = 2
    s.c = 3
    s.d = 4
    s.e = 5
    s.f = 6
    s.g = 7
    s.h = 8
    r: Large64 = c_increment_large64(s)
    return r.a + r.b + r.c + r.d + r.e + r.f + r.g + r.h  # Expected: 2+3+4+5+6+7+8+9 = 44


# =============================================================================
# Test Cases
# =============================================================================

class TestPythocCallsC(unittest.TestCase):
    """Test pythoc calling C functions"""
    
    @classmethod
    def setUpClass(cls):
        """Ensure C library is compiled"""
        get_c_library()
    
    def test_c_return_small8(self):
        """pythoc calls C function returning 8-byte struct"""
        result = test_call_c_return_small8()
        self.assertEqual(result, 300)
    
    def test_c_return_medium16(self):
        """pythoc calls C function returning 16-byte struct"""
        result = test_call_c_return_medium16()
        self.assertEqual(result, 3000)
    
    def test_c_return_large24(self):
        """pythoc calls C function returning 24-byte struct (sret)"""
        result = test_call_c_return_large24()
        self.assertEqual(result, 666)
    
    def test_c_return_large48(self):
        """pythoc calls C function returning 48-byte struct (sret)"""
        result = test_call_c_return_large48()
        self.assertEqual(result, 210)
    
    def test_c_return_large64(self):
        """pythoc calls C function returning 64-byte struct (sret)"""
        result = test_call_c_return_large64()
        self.assertEqual(result, 36)
    
    def test_c_sum_small8(self):
        """pythoc calls C function with 8-byte struct param"""
        result = test_call_c_sum_small8()
        self.assertEqual(result, 125)
    
    def test_c_sum_large24(self):
        """pythoc calls C function with 24-byte struct param (byval)"""
        result = test_call_c_sum_large24()
        self.assertEqual(result, 600)
    
    def test_c_sum_large48(self):
        """pythoc calls C function with 48-byte struct param (byval)"""
        result = test_call_c_sum_large48()
        self.assertEqual(result, 210)
    
    def test_c_sum_large64(self):
        """pythoc calls C function with 64-byte struct param (byval)"""
        result = test_call_c_sum_large64()
        self.assertEqual(result, 36)
    
    def test_c_double_small8(self):
        """pythoc calls C function taking and returning 8-byte struct"""
        result = test_call_c_double_small8()
        self.assertEqual(result, 60)
    
    def test_c_increment_large24(self):
        """pythoc calls C function taking and returning 24-byte struct"""
        result = test_call_c_increment_large24()
        self.assertEqual(result, 603)
    
    def test_c_increment_large48(self):
        """pythoc calls C function taking and returning 48-byte struct"""
        result = test_call_c_increment_large48()
        self.assertEqual(result, 216)
    
    def test_c_increment_large64(self):
        """pythoc calls C function taking and returning 64-byte struct"""
        result = test_call_c_increment_large64()
        self.assertEqual(result, 44)


class TestCCallsPythoc(unittest.TestCase):
    """Test C calling pythoc functions via function pointers"""
    
    @classmethod
    def setUpClass(cls):
        """Ensure C library is compiled and get function pointers"""
        cls.c_lib = get_c_library()
    
    def _get_pythoc_func_ptr(self, pc_func):
        """Get ctypes function pointer for a pythoc function"""
        # Use the normal execution path to ensure .so is compiled and loaded
        from pythoc.native_executor import get_multi_so_executor
        executor = get_multi_so_executor()
        
        # This will compile .so if needed and load the library properly
        executor.execute_function(pc_func)
        
        # Now get the function address from the loaded library
        so_file = pc_func._so_file
        lib = executor.loaded_libs[so_file]
        
        # Get function from library
        func_name = pc_func._actual_func_name
        native_func = getattr(lib, func_name)
        
        # Return the raw address
        if hasattr(native_func, '_address'):
            return native_func._address
        else:
            return ctypes.cast(native_func, ctypes.c_void_p).value
    
    def test_c_calls_pc_return_small8(self):
        """C calls pythoc function returning 8-byte struct"""
        # Get pythoc function pointer
        fn_ptr = self._get_pythoc_func_ptr(pc_return_small8)
        
        # Set up C function
        self.c_lib.c_call_fn_return_small8.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_fn_return_small8.restype = ctypes.c_int32
        
        # Call C function with pythoc function pointer
        result = self.c_lib.c_call_fn_return_small8(fn_ptr)
        self.assertEqual(result, 300)  # 100 + 200
    
    def test_c_calls_pc_return_large24(self):
        """C calls pythoc function returning 24-byte struct (sret)"""
        fn_ptr = self._get_pythoc_func_ptr(pc_return_large24)
        
        self.c_lib.c_call_fn_return_large24.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_fn_return_large24.restype = ctypes.c_int64
        
        result = self.c_lib.c_call_fn_return_large24(fn_ptr)
        self.assertEqual(result, 666)  # 111 + 222 + 333
    
    def test_c_calls_pc_return_large48(self):
        """C calls pythoc function returning 48-byte struct (sret)"""
        fn_ptr = self._get_pythoc_func_ptr(pc_return_large48)
        
        self.c_lib.c_call_fn_return_large48.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_fn_return_large48.restype = ctypes.c_int64
        
        result = self.c_lib.c_call_fn_return_large48(fn_ptr)
        self.assertEqual(result, 210)  # 10+20+30+40+50+60
    
    def test_c_calls_pc_return_large64(self):
        """C calls pythoc function returning 64-byte struct (sret)"""
        fn_ptr = self._get_pythoc_func_ptr(pc_return_large64)
        
        self.c_lib.c_call_fn_return_large64.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_fn_return_large64.restype = ctypes.c_int64
        
        result = self.c_lib.c_call_fn_return_large64(fn_ptr)
        self.assertEqual(result, 36)  # 1+2+3+4+5+6+7+8
    
    def test_c_calls_pc_sum_small8(self):
        """C calls pythoc function with 8-byte struct param"""
        fn_ptr = self._get_pythoc_func_ptr(pc_sum_small8)
        
        self.c_lib.c_call_fn_sum_small8.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_fn_sum_small8.restype = ctypes.c_int32
        
        result = self.c_lib.c_call_fn_sum_small8(fn_ptr)
        self.assertEqual(result, 100)  # 25 + 75
    
    def test_c_calls_pc_sum_large24(self):
        """C calls pythoc function with 24-byte struct param"""
        fn_ptr = self._get_pythoc_func_ptr(pc_sum_large24)
        
        self.c_lib.c_call_fn_sum_large24.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_fn_sum_large24.restype = ctypes.c_int64
        
        result = self.c_lib.c_call_fn_sum_large24(fn_ptr)
        self.assertEqual(result, 600)  # 100 + 200 + 300
    
    def test_c_calls_pc_sum_large48(self):
        """C calls pythoc function with 48-byte struct param"""
        fn_ptr = self._get_pythoc_func_ptr(pc_sum_large48)
        
        self.c_lib.c_call_fn_sum_large48.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_fn_sum_large48.restype = ctypes.c_int64
        
        result = self.c_lib.c_call_fn_sum_large48(fn_ptr)
        self.assertEqual(result, 210)  # 10+20+30+40+50+60
    
    def test_c_calls_pc_sum_large64(self):
        """C calls pythoc function with 64-byte struct param"""
        fn_ptr = self._get_pythoc_func_ptr(pc_sum_large64)
        
        self.c_lib.c_call_fn_sum_large64.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_fn_sum_large64.restype = ctypes.c_int64
        
        result = self.c_lib.c_call_fn_sum_large64(fn_ptr)
        self.assertEqual(result, 36)  # 1+2+3+4+5+6+7+8
    
    def test_c_calls_pc_double_small8(self):
        """C calls pythoc function taking and returning 8-byte struct"""
        fn_ptr = self._get_pythoc_func_ptr(pc_double_small8)
        
        self.c_lib.c_call_fn_double_small8.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_fn_double_small8.restype = ctypes.c_int32
        
        result = self.c_lib.c_call_fn_double_small8(fn_ptr)
        self.assertEqual(result, 60)  # (10*2) + (20*2) = 60
    
    def test_c_calls_pc_increment_large24(self):
        """C calls pythoc function taking and returning 24-byte struct"""
        fn_ptr = self._get_pythoc_func_ptr(pc_increment_large24)
        
        self.c_lib.c_call_fn_increment_large24.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_fn_increment_large24.restype = ctypes.c_int64
        
        result = self.c_lib.c_call_fn_increment_large24(fn_ptr)
        self.assertEqual(result, 603)  # (100+1) + (200+1) + (300+1) = 603
    
    def test_c_calls_pc_increment_large48(self):
        """C calls pythoc function taking and returning 48-byte struct"""
        fn_ptr = self._get_pythoc_func_ptr(pc_increment_large48)
        
        self.c_lib.c_call_fn_increment_large48.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_fn_increment_large48.restype = ctypes.c_int64
        
        result = self.c_lib.c_call_fn_increment_large48(fn_ptr)
        self.assertEqual(result, 216)  # (10+1)+(20+1)+(30+1)+(40+1)+(50+1)+(60+1) = 216
    
    def test_c_calls_pc_increment_large64(self):
        """C calls pythoc function taking and returning 64-byte struct"""
        fn_ptr = self._get_pythoc_func_ptr(pc_increment_large64)
        
        self.c_lib.c_call_fn_increment_large64.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_fn_increment_large64.restype = ctypes.c_int64
        
        result = self.c_lib.c_call_fn_increment_large64(fn_ptr)
        self.assertEqual(result, 44)  # (1+1)+(2+1)+...+(8+1) = 44


if __name__ == '__main__':
    unittest.main()
