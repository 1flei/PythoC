"""
Tests for linear_wrapper - metaprogramming linear resource wrappers

Tests the linear_wrap() and linear_wrap_struct() utilities for automatically
generating linear-safe wrappers around resource acquire/release function pairs.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc import compile, i32, i64, ptr, i8, struct
from pythoc.std.linear_wrapper import linear_wrap
from pythoc.std.utility import move
from pythoc.libc.stdlib import malloc, free
from pythoc.libc.stdio import fopen, fclose

# Generate all wrapped functions upfront before any execution
lmalloc, lfree = linear_wrap(malloc, free)
lfopen, lfclose = linear_wrap(fopen, fclose)
MemoryHandle, lmalloc_struct, lfree_struct = linear_wrap(malloc, free, struct_name="MemoryHandle")
FileHandle, lfopen_struct, lfclose_struct = linear_wrap(fopen, fclose, struct_name="FileHandle")

# Define all test functions upfront
@compile
def test_alloc():
    ptr_and_token = lmalloc(100)
    ptr_val = ptr_and_token[1]
    token = move(ptr_and_token[0])
    lfree(token, ptr_val)

@compile
def test_file():
    file_and_token = lfopen("/tmp/test_linear_wrapper.txt", "w")
    file_ptr = file_and_token[1]
    token = move(file_and_token[0])
    lfclose(token, file_ptr)

@compile
def test_alloc_struct():
    prf, ptr_val = lmalloc_struct(200)
    lfree_struct(prf, ptr_val)

@compile
def test_file_struct():
    handle = lfopen_struct("/tmp/test_linear_wrapper2.txt", "w")
    lfclose_struct(*handle)

@compile
def test_multi():
    # Allocate first block
    result1 = lmalloc(50)
    ptr1 = result1[1]
    token1 = move(result1[0])
    
    # Allocate second block
    result2 = lmalloc(100)
    ptr2 = result2[1]
    token2 = move(result2[0])
    
    # Free in reverse order
    lfree(token2, ptr2)
    lfree(token1, ptr1)


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
