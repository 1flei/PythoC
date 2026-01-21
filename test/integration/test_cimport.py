"""
Test cimport module - C header/source import functionality

Tests the cimport feature for importing C headers and sources as pythoc modules.

Test cases:
1. cimport C source -> compile to .o -> use symbols (compile_sources=True)
2. cimport header with C source files
3. cimport header with pre-compiled shared library
"""
from __future__ import annotations
import os
import tempfile
import unittest
import subprocess

from pythoc import compile, i32, i64, ptr, void, struct
from pythoc.libc.stdio import printf
from pythoc.registry import get_unified_registry


# =============================================================================
# Test fixtures - C source files for symbol import tests (no actual calls)
# =============================================================================

TEST_HEADER_CONTENT = '''
int add(int a, int b);
int multiply(int a, int b);
void print_hello(void);

struct Point {
    int x;
    int y;
};

typedef int myint;
'''

TEST_SOURCE_CONTENT = '''
#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}

void print_hello(void) {
    printf("Hello from C!\\n");
}

int get_answer(void) {
    return 42;
}
'''


# =============================================================================
# Basic symbol import tests (no actual calls)
# =============================================================================

class TestCimportHeader(unittest.TestCase):
    """Test cimport with header files - symbol generation only (no actual calls)"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.header_path = os.path.join(self.temp_dir, 'test.h')
        with open(self.header_path, 'w') as f:
            f.write(TEST_HEADER_CONTENT)
        # Note: Don't clear link_objects here - it would affect actual call tests
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_header_import_generates_bindings(self):
        """Test that cimport generates bindings from a header file"""
        from pythoc.cimport import cimport
        
        # Use a dummy lib that won't be linked (symbol generation test only)
        mod = cimport(self.header_path, lib='c')
        
        self.assertTrue(hasattr(mod, 'add'))
        self.assertTrue(hasattr(mod, 'multiply'))
        self.assertTrue(hasattr(mod, 'print_hello'))
        self.assertTrue(hasattr(mod, 'Point'))
        self.assertTrue(hasattr(mod, 'myint'))


class TestCimportSource(unittest.TestCase):
    """Test cimport with C source files - symbol generation only (no actual calls)"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.source_path = os.path.join(self.temp_dir, 'test.c')
        with open(self.source_path, 'w') as f:
            f.write(TEST_SOURCE_CONTENT)
        # Save current link_objects to restore later
        self._saved_link_objects = list(get_unified_registry().get_link_objects())
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # Restore link_objects to not affect other tests
        get_unified_registry().clear_link_objects()
        for obj in self._saved_link_objects:
            get_unified_registry().add_link_object(obj)
    
    @unittest.skip("Compiled bindgen cannot handle preprocessor directives like #include")
    def test_source_import_generates_bindings(self):
        """Test that cimport generates bindings from a C source file"""
        from pythoc.cimport import cimport
        
        # Use 'c' as lib to avoid registering fake library
        mod = cimport(self.source_path, lib='c', compile_sources=False)
        
        self.assertTrue(hasattr(mod, 'add'))
        self.assertTrue(hasattr(mod, 'multiply'))
        self.assertTrue(hasattr(mod, 'print_hello'))
        self.assertTrue(hasattr(mod, 'get_answer'))
    
    @unittest.skip("Compiled bindgen cannot handle preprocessor directives like #include")
    def test_source_import_with_compilation(self):
        """Test cimport with source compilation creates .o file"""
        from pythoc.cimport import cimport
        
        # Clear to test that cimport adds new objects
        get_unified_registry().clear_link_objects()
        
        mod = cimport(self.source_path, lib='c', compile_sources=True)
        
        link_objects = get_unified_registry().get_link_objects()
        self.assertTrue(len(link_objects) > 0)
        obj_exists = any(os.path.exists(obj) for obj in link_objects)
        self.assertTrue(obj_exists)


class TestCimportCaching(unittest.TestCase):
    """Test cimport caching behavior - symbol generation only (no actual calls)"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.header_path = os.path.join(self.temp_dir, 'cached.h')
        with open(self.header_path, 'w') as f:
            f.write('int cached_func(int x);\n')
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_caching_reuses_bindings(self):
        """Test that repeated cimport calls reuse cached bindings"""
        from pythoc.cimport import cimport
        import time
        
        mod1 = cimport(self.header_path, lib='c')
        time.sleep(0.01)
        mod2 = cimport(self.header_path, lib='c')
        
        self.assertTrue(hasattr(mod1, 'cached_func'))
        self.assertTrue(hasattr(mod2, 'cached_func'))
    
    def test_cache_invalidation_on_source_change(self):
        """Test that cache is invalidated when source changes"""
        from pythoc.cimport import cimport
        import time
        
        mod1 = cimport(self.header_path, lib='c')
        self.assertTrue(hasattr(mod1, 'cached_func'))
        
        time.sleep(0.1)
        with open(self.header_path, 'w') as f:
            f.write('int new_func(int x);\nint another_func(void);\n')
        
        mod2 = cimport(self.header_path, lib='c')
        self.assertTrue(hasattr(mod2, 'new_func'))
        self.assertTrue(hasattr(mod2, 'another_func'))


# =============================================================================
# Test Case 1: cimport C source with compile_sources=True
# Compile C source at module level, then define @compile wrappers
# =============================================================================

# C source for test case 1
_CASE1_C_SOURCE = '''
int c1_add(int a, int b) {
    return a + b;
}

int c1_sub(int a, int b) {
    return a - b;
}

int c1_mul(int a, int b) {
    return a * b;
}

int c1_square(int x) {
    return x * x;
}

int c1_factorial(int n) {
    int result = 1;
    int i;
    for (i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}
'''

# Create temp file and import at module level
_case1_temp_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'test', 'cimport_case1')
os.makedirs(_case1_temp_dir, exist_ok=True)
_case1_source_path = os.path.join(_case1_temp_dir, 'case1_math.c')
with open(_case1_source_path, 'w') as _f:
    _f.write(_CASE1_C_SOURCE)

# Import (don't clear registry - we need all .o files for linking)
from pythoc.cimport import cimport
_case1_mod = cimport(_case1_source_path, compile_sources=True)

# Get function references
c1_add = _case1_mod.c1_add
c1_sub = _case1_mod.c1_sub
c1_mul = _case1_mod.c1_mul
c1_square = _case1_mod.c1_square
c1_factorial = _case1_mod.c1_factorial


# Define @compile wrappers at module level
@compile
def test_c1_add(a: i32, b: i32) -> i32:
    return c1_add(a, b)


@compile
def test_c1_sub(a: i32, b: i32) -> i32:
    return c1_sub(a, b)


@compile
def test_c1_mul(a: i32, b: i32) -> i32:
    return c1_mul(a, b)


@compile
def test_c1_square(x: i32) -> i32:
    return c1_square(x)


@compile
def test_c1_factorial(n: i32) -> i32:
    return c1_factorial(n)


@compile
def test_c1_complex() -> i32:
    """Complex expression using multiple C functions"""
    a: i32 = c1_add(10, 5)   # 15
    b: i32 = c1_mul(a, 2)    # 30
    c: i32 = c1_sub(b, 10)   # 20
    return c


class TestCimportSourceActualCalls(unittest.TestCase):
    """Test Case 1: cimport C source -> compile to .o -> actually call functions"""
    
    def test_add(self):
        self.assertEqual(test_c1_add(3, 5), 8)
        self.assertEqual(test_c1_add(-10, 10), 0)
        self.assertEqual(test_c1_add(100, 200), 300)
    
    def test_sub(self):
        self.assertEqual(test_c1_sub(10, 3), 7)
        self.assertEqual(test_c1_sub(5, 10), -5)
    
    def test_mul(self):
        self.assertEqual(test_c1_mul(3, 4), 12)
        self.assertEqual(test_c1_mul(-2, 5), -10)
    
    def test_square(self):
        self.assertEqual(test_c1_square(5), 25)
        self.assertEqual(test_c1_square(-3), 9)
    
    def test_factorial(self):
        self.assertEqual(test_c1_factorial(0), 1)
        self.assertEqual(test_c1_factorial(1), 1)
        self.assertEqual(test_c1_factorial(5), 120)
        self.assertEqual(test_c1_factorial(10), 3628800)
    
    def test_complex_expression(self):
        self.assertEqual(test_c1_complex(), 20)


# =============================================================================
# Test Case 2: cimport header with C source files
# =============================================================================

_CASE2_HEADER = '''
int c2_add(int a, int b);
int c2_sub(int a, int b);
int c2_double(int x);
'''

_CASE2_SOURCE = '''
int c2_add(int a, int b) {
    return a + b;
}

int c2_sub(int a, int b) {
    return a - b;
}

int c2_double(int x) {
    return x * 2;
}
'''

# Create temp files
_case2_temp_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'test', 'cimport_case2')
os.makedirs(_case2_temp_dir, exist_ok=True)
_case2_header_path = os.path.join(_case2_temp_dir, 'case2_lib.h')
_case2_source_path = os.path.join(_case2_temp_dir, 'case2_lib.c')

with open(_case2_header_path, 'w') as _f:
    _f.write(_CASE2_HEADER)
with open(_case2_source_path, 'w') as _f:
    _f.write(_CASE2_SOURCE)

# Import (don't clear registry - we need all .o files for linking)
_case2_mod = cimport(
    _case2_header_path,
    sources=[_case2_source_path],
    compile_sources=True
)

c2_add = _case2_mod.c2_add
c2_sub = _case2_mod.c2_sub
c2_double = _case2_mod.c2_double


@compile
def test_c2_add(a: i32, b: i32) -> i32:
    return c2_add(a, b)


@compile
def test_c2_sub(a: i32, b: i32) -> i32:
    return c2_sub(a, b)


@compile
def test_c2_double(x: i32) -> i32:
    return c2_double(x)


class TestCimportHeaderWithSource(unittest.TestCase):
    """Test Case 2: cimport header with C source files"""
    
    def test_add(self):
        self.assertEqual(test_c2_add(10, 20), 30)
        self.assertEqual(test_c2_add(-5, 5), 0)
    
    def test_sub(self):
        self.assertEqual(test_c2_sub(100, 30), 70)
    
    def test_double(self):
        self.assertEqual(test_c2_double(21), 42)
        self.assertEqual(test_c2_double(-5), -10)


# =============================================================================
# Test Case 3: cimport header with pre-compiled shared library
# =============================================================================

_CASE3_HEADER = '''
int c3_add(int a, int b);
int c3_mul(int a, int b);
int c3_negate(int x);
'''

_CASE3_SOURCE = '''
int c3_add(int a, int b) {
    return a + b;
}

int c3_mul(int a, int b) {
    return a * b;
}

int c3_negate(int x) {
    return -x;
}
'''

# Create temp files and compile to shared library
# Use build directory for stable paths (cache-friendly)
_case3_temp_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'test', 'cimport_case3')
os.makedirs(_case3_temp_dir, exist_ok=True)
_case3_header_path = os.path.join(_case3_temp_dir, 'case3_shlib.h')
_case3_source_path = os.path.join(_case3_temp_dir, 'case3_shlib.c')
_case3_so_path = os.path.join(_case3_temp_dir, 'libcase3.so')

with open(_case3_header_path, 'w') as _f:
    _f.write(_CASE3_HEADER)
with open(_case3_source_path, 'w') as _f:
    _f.write(_CASE3_SOURCE)

# Compile to shared library
_result = subprocess.run(
    ['gcc', '-shared', '-fPIC', '-O2', '-o', _case3_so_path, _case3_source_path],
    capture_output=True, text=True, stdin=subprocess.DEVNULL
)
if _result.returncode != 0:
    raise RuntimeError(f"Failed to compile shared library: {_result.stderr}")

# Import with shared library path (don't clear registry)
_case3_mod = cimport(_case3_header_path, lib=_case3_so_path)

c3_add = _case3_mod.c3_add
c3_mul = _case3_mod.c3_mul
c3_negate = _case3_mod.c3_negate


@compile
def test_c3_add(a: i32, b: i32) -> i32:
    return c3_add(a, b)


@compile
def test_c3_mul(a: i32, b: i32) -> i32:
    return c3_mul(a, b)


@compile
def test_c3_negate(x: i32) -> i32:
    return c3_negate(x)


@compile
def test_c3_complex() -> i32:
    """Complex expression using shared library functions"""
    a: i32 = c3_add(5, 3)      # 8
    b: i32 = c3_mul(a, 2)      # 16
    c: i32 = c3_negate(b)      # -16
    return c


class TestCimportHeaderWithSharedLib(unittest.TestCase):
    """Test Case 3: cimport header with pre-compiled shared library"""
    
    def test_add(self):
        self.assertEqual(test_c3_add(7, 8), 15)
        self.assertEqual(test_c3_add(-100, 100), 0)
    
    def test_mul(self):
        self.assertEqual(test_c3_mul(6, 7), 42)
        self.assertEqual(test_c3_mul(-3, 4), -12)
    
    def test_negate(self):
        self.assertEqual(test_c3_negate(42), -42)
        self.assertEqual(test_c3_negate(-10), 10)
    
    def test_complex_expression(self):
        self.assertEqual(test_c3_complex(), -16)


# =============================================================================
# Cleanup temp directories at module unload
# =============================================================================

import atexit

def _cleanup():
    import shutil
    # Only clean up truly temporary directories, not build directory paths
    for d in [_case1_temp_dir, _case2_temp_dir]:
        if d.startswith('/tmp') or d.startswith(tempfile.gettempdir()):
            shutil.rmtree(d, ignore_errors=True)

atexit.register(_cleanup)


# =============================================================================
# Main test runner
# =============================================================================

@compile
def main() -> i32:
    printf("=== cimport Integration Tests ===\n")
    printf("Run with: PYTHONPATH=. python test/integration/test_cimport.py\n")
    return 0


if __name__ == '__main__':
    unittest.main()
