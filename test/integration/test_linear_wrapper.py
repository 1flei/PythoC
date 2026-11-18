"""
Tests for linearize - metaprogramming linear resource wrappers.

Tests the linearize() utility for automatically generating linear-safe wrappers
around resource acquire/release function pairs.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc import compile, i32, i64, ptr, i8, struct, move
from pythoc.std.linearize import linearize
from pythoc.libc.stdlib import malloc, free
from pythoc.libc.stdio import fopen, fclose

# Generate all wrapped functions upfront before any execution
MemoryProof = linearize(malloc, free)
FileProof = linearize(fopen, fclose)
MemoryHandle = linearize(malloc, free, name="MemoryHandle")
FileHandle = linearize(fopen, fclose, name="FileHandle")


@compile
def test_alloc():
    ptr_and_token = MemoryProof.acquire(100)
    ptr_val = ptr_and_token[0]
    token = move(ptr_and_token[1])
    MemoryProof.release(ptr_val, token)


@compile
def test_file():
    file_and_token = FileProof.acquire("build/test_linear_wrapper_tmp.txt", "w")
    file_ptr = file_and_token[0]
    token = move(file_and_token[1])
    FileProof.release(file_ptr, token)


@compile
def test_alloc_struct():
    ptr_val, prf = MemoryHandle.acquire(200)
    MemoryHandle.release(ptr_val, prf)


@compile
def test_file_struct():
    handle = FileHandle.acquire("build/test_linear_wrapper_tmp2.txt", "w")
    FileHandle.release(*handle)


@compile
def test_multi():
    # Allocate first block
    result1 = MemoryProof.acquire(50)
    ptr1 = result1[0]
    token1 = move(result1[1])

    # Allocate second block
    result2 = MemoryProof.acquire(100)
    ptr2 = result2[0]
    token2 = move(result2[1])

    # Free in reverse order
    MemoryProof.release(ptr2, token2)
    MemoryProof.release(ptr1, token1)


def test_malloc_free_wrapper():
    """Test linear_wrap with malloc/free"""
    print("Testing malloc/free wrapper...")
    
    # Debug: print generated code
    print(f"  lmalloc type: {type(MemoryProof.acquire)}")
    print(f"  lfree type: {type(MemoryProof.release)}")
    
    # Execute compiled function
    test_alloc()
    print("  malloc/free wrapper: PASS")


def test_fopen_fclose_wrapper():
    """Test linear_wrap with fopen/fclose"""
    print("Testing fopen/fclose wrapper...")
    
    print(f"  lfopen type: {type(FileProof.acquire)}")
    print(f"  lfclose type: {type(FileProof.release)}")
    
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
