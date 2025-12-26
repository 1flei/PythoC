#!/usr/bin/env python3
"""
Test ptr functionality in compiled PC functions
"""

from __future__ import annotations
from pythoc import i8, i16, i32, i64, f64, ptr, compile, sizeof
from pythoc.libc.stdio import printf
from pythoc.libc.stdlib import malloc, free
from pythoc.libc.string import memset

@compile
class TestStruct:
    a: i32
    b: i64
    c: i8

def myprint(*args):
    print(*args)

@compile
def testprint() -> i32:
    # printf("Simple test\n")
    # Allocate array
    ps10 = ptr[TestStruct](malloc(10 * sizeof(TestStruct)))
    printf("After ptrcast: ps10 = %p\n", ps10)
    
    # Get pointer to second element
    print(TestStruct)
    myprint(TestStruct)
    print(ps10)
    myprint(ps10)
    return 0

if __name__ == "__main__":
    testprint()
    print("test_print passed!")
