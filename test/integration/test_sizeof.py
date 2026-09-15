#!/usr/bin/env python3
"""
Test sizeof in compiled PC functions
"""

import unittest
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

@compile(suffix="sizeof_individual")
def sizeof_struct1() -> i32:
    """sizeof(TestStruct): 4 + 4(pad) + 8 + 1 + 7(pad)"""
    return sizeof(TestStruct)

@compile(suffix="sizeof_individual")
def sizeof_struct2() -> i32:
    """sizeof(TestStruct2): 24 + 8 + 24 + 8"""
    return sizeof(TestStruct2)

@compile(suffix="sizeof_individual")
def sizeof_struct3_only() -> i32:
    """sizeof(TestStruct3): 24 + 64 + 1 + 3(pad) + 4 + 8"""
    return sizeof(TestStruct3)

class TestSizeof(unittest.TestCase):
    def test_basic_types(self):
        self.assertEqual(test_sizeof_basic(), 21)

    def test_pointer_types(self):
        self.assertEqual(test_sizeof_pointers(), 16)

    def test_struct_sum(self):
        self.assertEqual(test_sizeof_struct(), 88)

    def test_struct3(self):
        self.assertEqual(test_sizeof_struct3(), 104)

    def test_individual_struct_sizes(self):
        self.assertEqual(sizeof_struct1(), 24)
        self.assertEqual(sizeof_struct2(), 64)
        self.assertEqual(sizeof_struct3_only(), 104)

if __name__ == "__main__":
    unittest.main()
