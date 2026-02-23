"""
Tests for linearize - metaprogramming linear resource wrappers.

Tests the linearize() utility for automatically generating linear-safe wrappers
around resource acquire/release function pairs.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc import compile, i32, i64, ptr, i8, struct
from pythoc.std.linearize import linearize
from pythoc.std.utility import move
from pythoc.libc.stdlib import malloc, free
from pythoc.libc.stdio import fopen, fclose

# Generate all wrapped functions upfront before any execution
lmalloc, lfree = linearize(malloc, free)
lfopen, lfclose = linearize(fopen, fclose)
MemoryHandle, lmalloc_struct, lfree_struct = linearize(malloc, free, struct_name="MemoryHandle")
FileHandle, lfopen_struct, lfclose_struct = linearize(fopen, fclose, struct_name="FileHandle")


@compile
def test_alloc():
    ptr_and_token = lmalloc(100)
    ptr_val = ptr_and_token[0]
    token = move(ptr_and_token[1])
    lfree(ptr_val, token)


@compile
def test_file():
    file_and_token = lfopen("build/test_linear_wrapper_tmp.txt", "w")
    file_ptr = file_and_token[0]
    token = move(file_and_token[1])
    lfclose(file_ptr, token)


@compile
def test_alloc_struct():
    ptr_val, prf = lmalloc_struct(200)
    lfree_struct(ptr_val, prf)


@compile
def test_file_struct():
    handle = lfopen_struct("build/test_linear_wrapper_tmp2.txt", "w")
    lfclose_struct(*handle)


@compile
def test_multi():
    # Allocate first block
    result1 = lmalloc(50)
    ptr1 = result1[0]
    token1 = move(result1[1])

    # Allocate second block
    result2 = lmalloc(100)
    ptr2 = result2[0]
    token2 = move(result2[1])

    # Free in reverse order
    lfree(ptr2, token2)
    lfree(ptr1, token1)


def test_malloc_free_wrapper():
    """Test linear_wrap with malloc/free"""
    print("Testing malloc/free wrapper...")
    
    # Debug: print generated code
    print(f"  lmalloc type: {type(lmalloc)}")
    print(f"  lfree type: {type(lfree)}")
    
    # Execute compiled function
    test_alloc()
    print("  malloc/free wrapper: PASS")


def test_fopen_fclose_wrapper():
    """Test linear_wrap with fopen/fclose"""
    print("Testing fopen/fclose wrapper...")
    
    print(f"  lfopen type: {type(lfopen)}")
    print(f"  lfclose type: {type(lfclose)}")
    
    # Execute compiled function
    test_file()
    print("  fopen/fclose wrapper: PASS")


def test_malloc_struct_wrapper():
    """Test linear_wrap with struct return"""
    print("Testing malloc/free struct wrapper...")
    
    # Execute compiled function
    test_alloc_struct()
    print("  malloc/free struct wrapper: PASS")


def test_file_struct_wrapper():
    """Test linear_wrap with struct return for fopen/fclose"""
    print("Testing fopen/fclose struct wrapper...")
    
    # Execute compiled function
    test_file_struct()
    print("  fopen/fclose struct wrapper: PASS")


def test_multiple_allocations():
    """Test multiple allocations with linear tokens"""
    print("Testing multiple allocations...")
    
    # Execute compiled function
    test_multi()
    print("  multiple allocations: PASS")


import unittest


class TestLinearWrapper(unittest.TestCase):
    """Test linear_wrapper metaprogramming utilities"""

    def test_malloc_free_wrapper(self):
        """Test linear_wrap with malloc/free"""
        test_alloc()

    def test_fopen_fclose_wrapper(self):
        """Test linear_wrap with fopen/fclose"""
        test_file()

    def test_malloc_struct_wrapper(self):
        """Test linear_wrap with struct return"""
        test_alloc_struct()

    def test_file_struct_wrapper(self):
        """Test linear_wrap with struct return for fopen/fclose"""
        test_file_struct()

    def test_multiple_allocations(self):
        """Test multiple allocations with linear tokens"""
        test_multi()


if __name__ == '__main__':
    unittest.main()
