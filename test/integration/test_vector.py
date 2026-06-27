#!/usr/bin/env python3
import sys

from pythoc import i16, i32, i64, ptr, compile, seq
from pythoc.libc.stdio import printf
from pythoc.std.vector import Vector

IntVec = Vector(i32, 4)
I16Vec = Vector(i16, 1)
I64Vec = Vector(i64, 2)

def test_base(vect, push_size):
    @compile(suffix=(vect, push_size))
    def test_vector() -> i32:
        v: vect.type
        vp = ptr(v)
        vect.init(vp)

        for i in seq(push_size):
            vect.push_back(vp, i)

        sz: i32 = vect.size(vp)
        cap: i32 = vect.capacity(vp)
        printf("Vector: size=%d, capacity=%d\n", sz, cap)

        for j in seq(sz):
            val: i32 = vect.get(vp, j)
            printf("  v[%d] = %d\n", j, i32(val))

        vect.destroy(vp)
        return 0
    return test_vector


def test_boundary(vect, cap):
    """Push exactly inline_capacity elements.

    At size == inline_capacity the small-vector is still inline; a wrong
    inline/heap predicate would read the heap union member (garbage) here.
    Verify both the indexed accessor and the contiguous data() pointer.
    """
    @compile(suffix=(vect, cap, "boundary"))
    def run() -> i32:
        v: vect.type
        vp = ptr(v)
        vect.init(vp)

        for i in seq(cap):
            vect.push_back(vp, i)

        sz: i32 = i32(vect.size(vp))
        bad: i32 = 0
        j: i32 = 0
        while j < sz:
            if i32(vect.get(vp, j)) != j:
                bad = bad + 1
            j = j + 1

        d = vect.data(vp)
        if sz > 0:
            if i32(d[0]) != 0:
                bad = bad + 1
            if i32(d[sz - 1]) != (sz - 1):
                bad = bad + 1

        printf("boundary size=%d bad=%d\n", sz, bad)
        vect.destroy(vp)
        return bad
    return run

if __name__ == "__main__":
    fs = []
    for t in [IntVec, I16Vec, I64Vec]:
        fs += [test_base(t, 10)]

    for f in fs:
        f()

    # Regression: resting exactly at inline_capacity must stay inline.
    failures = 0
    for t, cap in [(IntVec, 4), (I16Vec, 1), (I64Vec, 2)]:
        failures += int(test_boundary(t, cap)())

    if failures:
        print("Vector boundary regression FAILED")
        sys.exit(1)

    print("All vector tests passed!")
