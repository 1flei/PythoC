#!/usr/bin/env python3
"""
Test ptr functionality in compiled PC functions
"""

from __future__ import annotations
import sys
import os
import unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pythoc import i8, i16, i32, i64, f64, ptr, compile, sizeof
from pythoc.libc.stdio import printf
from pythoc.libc.stdlib import malloc, free
from pythoc.libc.string import memset

@compile
class TestStruct:
    a: i32
    b: i64
    c: i8

@compile
def test_ptr_struct() -> i32:
    """Test ptr with struct fields"""
    printf("Test ptr struct\n")
    s: TestStruct = TestStruct()
    s.a = 10
    s.b = 20
    s.c = 5
    total: i32 = s.a + i32(s.b) + i32(s.c)  # 35
    printf("S a=%d, b=%lld, c=%d\n", s.a, s.b, s.c)

    # Get pointers to struct fields
    ps = ptr(s)
    printf("PS a=%d, b=%lld, c=%d, p=%p\n", ps.a, ps.b, ps.c, ps)
    ps.a = 100
    total += s.a  # 100
    printf("S a=%d, b=%lld, c=%d\n", s.a, s.b, s.c)

    pps = ptr(ps)
    printf("PPS a=%d, b=%lld, c=%d, p=%p\n", pps.a, pps.b, pps.c, pps)
    pps.a = 200
    total += s.a  # 200
    printf("S a=%d, b=%lld, c=%d\n", s.a, s.b, s.c)

    ps2 = pps[0]
    total += ps2.a + i32(ps2.b) + i32(ps2.c)  # 225
    printf("PS2 a=%d, b=%lld, c=%d, p=%p\n", ps2.a, ps2.b, ps2.c, ps2)

    ps10 = ptr[TestStruct](malloc(10 * sizeof(TestStruct)))
    ps10[0] = s
    total += ps10.a + i32(ps10.b) + i32(ps10.c)  # 225
    ps10_1 = ps10 + 1
    ps10_1.a = 101
    ps10_1.b = 102
    ps10_1.c = 103
    total += ps10_1.a + i32(ps10_1.b) + i32(ps10_1.c)  # 306
    printf("PS10 a=%d, b=%lld, c=%d, p=%p\n", ps10.a, ps10.b, ps10.c, ps10)
    printf("PS10_1 a=%d, b=%lld, c=%d, p=%p\n", ps10_1.a, ps10_1.b, ps10_1.c, ps10_1)

    memset(ps10 + 2, 0, 8 * sizeof(TestStruct))
    i: i32 = 2
    while i < 10:
        ps10[i].a = i * 10
        ps10[i].b = i * 20
        total += ps10[i].a + i32(ps10[i].b) + i32(ps10[i].c)  # 30 * i
        printf("PS10 a=%d, b=%lld, c=%d, p=%p\n", ps10[i].a, ps10[i].b, ps10[i].c, ps10 + i)
        pi = ptr(ps10[i])
        printf("PS10 a=%d, b=%lld, c=%d, pi=%p\n", pi.a, pi.b, pi.c, pi)
        s : TestStruct = ps10[i]
        total += s.a + i32(s.b) + i32(s.c)  # 30 * i
        printf("PS10 a=%d, b=%lld, c=%d\n", s.a, s.b, s.c)
        i += 1

    free(ps10)
    return total

@compile
def simple_pointer_assign() -> i32:
    # printf("Simple test\n")
    # Allocate array
    ps10 = ptr[TestStruct](malloc(10 * sizeof(TestStruct)))
    printf("After ptrcast: ps10 = %p\n", ps10)
    
    # Get pointer to second element
    print(TestStruct)
    ps10_1 = ps10 + 1
    printf("sizeof(TestStruct)=%d\n", sizeof(TestStruct))
    printf("PS10 = %p, ps10_1 = %p, sizeof(TestStruct)=%d\n", ps10, ps10_1, sizeof(TestStruct))
    
    # # # Assign fields
    ps10_1.a = 201
    ps10_1.b = 202
    ps10_1.c = 203
    
    printf("ps10_1.a = %d\n", ps10_1.a)
    printf("ps10_1.b = %lld\n", ps10_1.b)
    printf("ps10_1.c = %d\n", i32(ps10_1.c))

    # c is i8: 203 wraps to -53 on store, so the sum is 201 + 202 - 53
    return ps10_1.a + i32(ps10_1.b) + i32(ps10_1.c)

@compile
def main() -> i32:
    test_ptr_struct()
    simple_pointer_assign()
    return 0

class TestMemberAccess(unittest.TestCase):
    def test_ptr_struct(self):
        # 35 + 100 + 200 + 225 + 225 + 306 + 2 * sum(30*i for i in 2..9)
        self.assertEqual(test_ptr_struct(), 3731)

    def test_simple_pointer_assign(self):
        self.assertEqual(simple_pointer_assign(), 350)

# run via native executor
if __name__ == "__main__":
    unittest.main()
