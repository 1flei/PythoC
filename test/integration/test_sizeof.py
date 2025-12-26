#!/usr/bin/env python3
"""
Test sizeof in compiled PC functions
"""

from pythoc import i8, i16, i32, i64, u32, f64, ptr, compile, sizeof
from pythoc.libc.stdio import printf

@compile
class TestStruct:
    a: i32      # 4 bytes
    b: i64      # 8 bytes  
    c: i8       # 1 byte
    # Total with alignment: 4 + 4(pad) + 8 + 1 + 7(pad) = 24 bytes

@compile
class TestStruct2:
    a: TestStruct   # 24
    b: i8           # 1 bytes + 7(pad) = 8
    c: TestStruct   # 24
    d: i32          # 4 + 4(pad) = 8
    # Total with alignment: 24 + 8 + 24 + 8 = 64 bytes

@compile
class TestStruct3:
    a: TestStruct
    c: TestStruct2
    b: i8
    d: i32
    e: ptr[TestStruct2]

@compile
def test_sizeof_basic() -> i32:
    """Test sizeof with basic types"""
    size_i8: i32 = sizeof(i8)        # Should be 1
    size_i32: i32 = sizeof(i32)      # Should be 4
    size_i64: i32 = sizeof(i64)      # Should be 8
    size_f64: i32 = sizeof(f64)      # Should be 8
    
    return size_i8 + size_i32 + size_i64 + size_f64  # 1 + 4 + 8 + 8 = 21

@compile
def test_sizeof_pointers() -> i32:
    """Test sizeof with pointer types"""
    size_ptr_i32: i32 = sizeof(ptr[i32])    # Should be 8
    size_ptr_f64: i32 = sizeof(ptr[f64])    # Should be 8
    
    return size_ptr_i32 + size_ptr_f64  # 8 + 8 = 16

@compile  
def test_sizeof_struct() -> i32:
    """Test sizeof with struct types"""
    struct_size: i32 = sizeof(TestStruct)
    struct_size2: i32 = sizeof(TestStruct2)
    return struct_size + struct_size2

@compile  
def test_sizeof_struct3() -> i32:
    """Test sizeof with struct types"""
    struct_size: i32 = sizeof(TestStruct3)
    return struct_size

@compile
def main() -> i32:
    """Main function that calls all tests"""
    printf("Starting sizeof tests...")
    printf("test_sizeof_basic: %d\n", test_sizeof_basic())
    printf("test_sizeof_pointers: %d\n", test_sizeof_pointers())
    printf("test_sizeof_struct: %d\n", test_sizeof_struct())
    printf("test_sizeof_struct3: %d\n", test_sizeof_struct3())
    return 0

if __name__ == "__main__":
    main()
    
    # Verify results
    assert test_sizeof_basic() == 21, f"test_sizeof_basic expected 21, got {test_sizeof_basic()}"
    assert test_sizeof_pointers() == 16, f"test_sizeof_pointers expected 16, got {test_sizeof_pointers()}"
    
    print("All sizeof tests passed!")
