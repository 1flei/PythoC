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
    ptr_val = ptr_and_token[0]
    token = move(ptr_and_token[1])
    lfree(ptr_val, token)

@compile
def test_file():
    file_and_token = lfopen("/tmp/test_linear_wrapper.txt", "w")
    file_ptr = file_and_token[0]
    token = move(file_and_token[1])
    lfclose(file_ptr, token)

@compile
def test_alloc_struct():
    handle: MemoryHandle = lmalloc_struct(200)
    ptr_val = handle[0]
    token = move(handle[1])
    lfree_struct(ptr_val, token)

@compile
def test_file_struct():
    handle: FileHandle = lfopen_struct("/tmp/test_linear_wrapper2.txt", "w")
    file_ptr = handle[0]
    token = move(handle[1])
    lfclose_struct(file_ptr, token)

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


if __name__ == '__main__':
    print("Running linear_wrapper tests...\n")
    
    try:
        test_malloc_free_wrapper()
        test_fopen_fclose_wrapper()
        test_malloc_struct_wrapper()
        test_file_struct_wrapper()
        test_multiple_allocations()
        
        print("\nAll linear_wrapper tests passed!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
