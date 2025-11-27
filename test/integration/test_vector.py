#!/usr/bin/env python3
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

fs = []
for t in [IntVec, I16Vec, I64Vec]:
    fs += [test_base(t, 10)]

for f in fs:
    f()
