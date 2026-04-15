"""
Integration test for C ABI varargs with actual runtime execution.

Tests:
1. pythoc @compile function calls another pythoc varargs function (IR-level call)
2. C code calls pythoc varargs function via function pointer
3. Multiple types: i32, i64, f64, ptr[i8], mixed

This validates that the LLVM va_arg instruction (patched into llvmlite)
generates correct platform-specific ABI code for all common types.
"""

import os
import sys
import unittest
import ctypes
import tempfile
import subprocess

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

from pythoc import (
    compile, i32, i64, f64, u64, ptr, i8, void,
    va_start, va_arg, va_end,
    flush_all_pending_outputs,
)
from pythoc.libc.stdio import printf


# ============================================================================
# Part 1: pythoc varargs functions (callee)
# ============================================================================

@compile
def va_sum_three_i32(count: i32, *args) -> i32:
    """Sum three i32 varargs."""
    ap = va_start()
    a: i32 = va_arg(ap, i32)
    b: i32 = va_arg(ap, i32)
    c: i32 = va_arg(ap, i32)
    va_end(ap)
    return a + b + c


@compile
def va_first_i32(dummy: i32, *args) -> i32:
    """Read one i32 vararg."""
    ap = va_start()
    val: i32 = va_arg(ap, i32)
    va_end(ap)
    return val


@compile
def va_sum_two_i64(dummy: i32, *args) -> i64:
    """Sum two i64 varargs."""
    ap = va_start()
    a: i64 = va_arg(ap, i64)
    b: i64 = va_arg(ap, i64)
    va_end(ap)
    return a + b


@compile
def va_sum_two_f64(dummy: i32, *args) -> f64:
    """Sum two f64 varargs."""
    ap = va_start()
    a: f64 = va_arg(ap, f64)
    b: f64 = va_arg(ap, f64)
    va_end(ap)
    return a + b


@compile
def va_mixed_i32_f64(dummy: i32, *args) -> f64:
    """Read i32 then f64, return their sum as f64."""
    ap = va_start()
    a: i32 = va_arg(ap, i32)
    b: f64 = va_arg(ap, f64)
    va_end(ap)
    return f64(a) + b


@compile
def va_read_ptr(dummy: i32, *args) -> ptr[i8]:
    """Read a ptr[i8] vararg."""
    ap = va_start()
    p: ptr[i8] = va_arg(ap, ptr[i8])
    va_end(ap)
    return p


@compile
def va_many_i32(dummy: i32, *args) -> i32:
    """Sum 8 i32 varargs (exceeds register window on x86_64)."""
    ap = va_start()
    s: i32 = va_arg(ap, i32)
    s = s + va_arg(ap, i32)
    s = s + va_arg(ap, i32)
    s = s + va_arg(ap, i32)
    s = s + va_arg(ap, i32)
    s = s + va_arg(ap, i32)
    s = s + va_arg(ap, i32)
    s = s + va_arg(ap, i32)
    va_end(ap)
    return s


# ============================================================================
# Part 2: pythoc caller wrappers (compile-time IR-level calls)
# ============================================================================

@compile
def test_call_sum_three_i32() -> i32:
    return va_sum_three_i32(i32(3), i32(10), i32(20), i32(30))


@compile
def test_call_first_i32() -> i32:
    return va_first_i32(i32(0), i32(42))


@compile
def test_call_sum_two_i64() -> i64:
    return va_sum_two_i64(i32(0), i64(100000), i64(200000))


@compile
def test_call_sum_two_f64() -> f64:
    return va_sum_two_f64(i32(0), f64(1.5), f64(2.5))


@compile
def test_call_mixed_i32_f64() -> f64:
    return va_mixed_i32_f64(i32(0), i32(10), f64(3.14))


@compile
def test_call_many_i32() -> i32:
    return va_many_i32(
        i32(0),
        i32(1), i32(2), i32(3), i32(4),
        i32(5), i32(6), i32(7), i32(8),
    )


# ============================================================================
# Part 3: C source that calls pythoc varargs functions
# ============================================================================

C_SOURCE = r"""
#include <stdint.h>
#include <stdarg.h>
#include <math.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

typedef int32_t (*sum3_i32_fn)(int32_t, ...);
typedef int32_t (*first_i32_fn)(int32_t, ...);
typedef int64_t (*sum2_i64_fn)(int32_t, ...);
typedef double  (*sum2_f64_fn)(int32_t, ...);
typedef double  (*mixed_fn)(int32_t, ...);
typedef int32_t (*many_i32_fn)(int32_t, ...);

EXPORT int32_t c_call_sum3_i32(sum3_i32_fn fn) {
    return fn(3, 10, 20, 30);
}

EXPORT int32_t c_call_first_i32(first_i32_fn fn) {
    return fn(0, 42);
}

EXPORT int64_t c_call_sum2_i64(sum2_i64_fn fn) {
    return fn(0, (int64_t)100000, (int64_t)200000);
}

EXPORT double c_call_sum2_f64(sum2_f64_fn fn) {
    return fn(0, 1.5, 2.5);
}

EXPORT double c_call_mixed(mixed_fn fn) {
    return fn(0, 10, 3.14);
}

EXPORT int32_t c_call_many_i32(many_i32_fn fn) {
    return fn(0, 1, 2, 3, 4, 5, 6, 7, 8);
}

/* C sanity check */
EXPORT int32_t c_sum3(int32_t count, ...) {
    va_list ap;
    va_start(ap, count);
    int32_t a = va_arg(ap, int32_t);
    int32_t b = va_arg(ap, int32_t);
    int32_t c = va_arg(ap, int32_t);
    va_end(ap);
    return a + b + c;
}
"""

_c_lib = None
_c_dll_path = None


def _compile_c_library():
    global _c_lib, _c_dll_path
    if _c_dll_path is not None:
        return _c_dll_path

    from pythoc.utils.cc_utils import compile_c_to_object
    from pythoc.utils.link_utils import (
        get_shared_lib_extension, find_available_linker, get_platform_link_flags,
    )

    build_dir = os.path.join(tempfile.gettempdir(), "pythoc_test_va_arg")
    os.makedirs(build_dir, exist_ok=True)

    c_file = os.path.join(build_dir, "test_va_helper.c")
    with open(c_file, "w") as f:
        f.write(C_SOURCE)

    obj_file = os.path.join(build_dir, "test_va_helper.o")
    compile_c_to_object(c_file, output_path=obj_file)

    ext = get_shared_lib_extension()
    dll_file = os.path.join(build_dir, f"test_va_helper{ext}")

    linker = find_available_linker()
    linker_cmd = linker.split()
    platform_flags = get_platform_link_flags(shared=True, linker=linker)
    cmd = (
        linker_cmd + platform_flags
        + [os.path.abspath(obj_file), "-o", os.path.abspath(dll_file)]
    )
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise RuntimeError(f"C library linking failed:\n{result.stderr}\ncmd={cmd}")

    _c_dll_path = dll_file
    return dll_file


def _get_c_library():
    global _c_lib
    if _c_lib is None:
        _compile_c_library()
        _c_lib = ctypes.CDLL(_c_dll_path)
    return _c_lib


def _get_pythoc_func_ptr(pc_func):
    from pythoc.native_executor import get_multi_so_executor
    executor = get_multi_so_executor()
    executor.execute_function(pc_func)
    so_file = pc_func._so_file
    lib = executor.loaded_libs[so_file]
    func_name = pc_func._actual_func_name
    native_func = getattr(lib, func_name)
    if hasattr(native_func, "_address"):
        return native_func._address
    return ctypes.cast(native_func, ctypes.c_void_p).value


# ============================================================================
# Tests: pythoc @compile -> pythoc varargs (IR-level call, no ctypes involved)
# ============================================================================

class TestVaArgPythocCallsPythoc(unittest.TestCase):
    """pythoc @compile wrapper calls another @compile varargs function."""

    def test_sum_three_i32(self):
        self.assertEqual(test_call_sum_three_i32(), 60)

    def test_first_i32(self):
        self.assertEqual(test_call_first_i32(), 42)

    def test_sum_two_i64(self):
        self.assertEqual(test_call_sum_two_i64(), 300000)

    def test_sum_two_f64(self):
        self.assertAlmostEqual(test_call_sum_two_f64(), 4.0, places=10)

    def test_mixed_i32_f64(self):
        self.assertAlmostEqual(test_call_mixed_i32_f64(), 13.14, places=10)

    def test_many_i32_exceeds_registers(self):
        """8 i32 varargs: first 5 in registers, rest on stack (x86_64 SysV)."""
        self.assertEqual(test_call_many_i32(), 36)  # 1+2+3+4+5+6+7+8


# ============================================================================
# Tests: C -> pythoc varargs (via function pointer)
# ============================================================================

class TestVaArgCCallsPythoc(unittest.TestCase):
    """C code calls pythoc varargs functions via function pointer."""

    @classmethod
    def setUpClass(cls):
        cls.c_lib = _get_c_library()

    def test_c_sanity(self):
        self.c_lib.c_sum3.argtypes = [ctypes.c_int32]
        self.c_lib.c_sum3.restype = ctypes.c_int32
        self.assertEqual(self.c_lib.c_sum3(3, 10, 20, 30), 60)

    def test_c_calls_sum3_i32(self):
        fn_ptr = _get_pythoc_func_ptr(va_sum_three_i32)
        self.c_lib.c_call_sum3_i32.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_sum3_i32.restype = ctypes.c_int32
        self.assertEqual(self.c_lib.c_call_sum3_i32(fn_ptr), 60)

    def test_c_calls_first_i32(self):
        fn_ptr = _get_pythoc_func_ptr(va_first_i32)
        self.c_lib.c_call_first_i32.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_first_i32.restype = ctypes.c_int32
        self.assertEqual(self.c_lib.c_call_first_i32(fn_ptr), 42)

    def test_c_calls_sum2_i64(self):
        fn_ptr = _get_pythoc_func_ptr(va_sum_two_i64)
        self.c_lib.c_call_sum2_i64.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_sum2_i64.restype = ctypes.c_int64
        self.assertEqual(self.c_lib.c_call_sum2_i64(fn_ptr), 300000)

    def test_c_calls_sum2_f64(self):
        fn_ptr = _get_pythoc_func_ptr(va_sum_two_f64)
        self.c_lib.c_call_sum2_f64.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_sum2_f64.restype = ctypes.c_double
        self.assertAlmostEqual(self.c_lib.c_call_sum2_f64(fn_ptr), 4.0, places=10)

    def test_c_calls_mixed(self):
        fn_ptr = _get_pythoc_func_ptr(va_mixed_i32_f64)
        self.c_lib.c_call_mixed.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_mixed.restype = ctypes.c_double
        self.assertAlmostEqual(self.c_lib.c_call_mixed(fn_ptr), 13.14, places=10)

    def test_c_calls_many_i32(self):
        fn_ptr = _get_pythoc_func_ptr(va_many_i32)
        self.c_lib.c_call_many_i32.argtypes = [ctypes.c_void_p]
        self.c_lib.c_call_many_i32.restype = ctypes.c_int32
        self.assertEqual(self.c_lib.c_call_many_i32(fn_ptr), 36)


if __name__ == "__main__":
    print("=== C ABI varargs runtime tests ===\n")
    unittest.main(verbosity=2)
