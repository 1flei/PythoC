#!/usr/bin/env python3
import sys

from pythoc import i16, i32, i64, ptr, compile, seq
from pythoc.libc.stdio import printf
from pythoc.std.vector import Vector

IntVec = Vector(i32, 4)
I16Vec = Vector(i16, 1)
I64Vec = Vector(i64, 2)


# ---------------------------------------------------------------------------
# Basic push_back / get / size / capacity across the three vector types.
# ---------------------------------------------------------------------------

@compile(suffix=(IntVec, 10))
def test_intvec_base() -> i32:
    v: IntVec
    vp = ptr(v)
    IntVec.init(vp)

    for i in seq(10):
        IntVec.push_back(vp, i)

    sz: i32 = IntVec.size(vp)
    cap: i32 = IntVec.capacity(vp)
    printf("Vector: size=%d, capacity=%d\n", sz, cap)

    for j in seq(sz):
        val: i32 = IntVec.get(vp, j)
        printf("  v[%d] = %d\n", j, i32(val))

    IntVec.destroy(vp)
    return 0


@compile(suffix=(I16Vec, 10))
def test_i16vec_base() -> i32:
    v: I16Vec
    vp = ptr(v)
    I16Vec.init(vp)

    for i in seq(10):
        I16Vec.push_back(vp, i16(i))

    sz: i32 = I16Vec.size(vp)
    cap: i32 = I16Vec.capacity(vp)
    printf("Vector: size=%d, capacity=%d\n", sz, cap)

    for j in seq(sz):
        val: i16 = I16Vec.get(vp, j)
        printf("  v[%d] = %d\n", j, i32(val))

    I16Vec.destroy(vp)
    return 0


@compile(suffix=(I64Vec, 10))
def test_i64vec_base() -> i32:
    v: I64Vec
    vp = ptr(v)
    I64Vec.init(vp)

    for i in seq(10):
        I64Vec.push_back(vp, i64(i))

    sz: i32 = I64Vec.size(vp)
    cap: i32 = I64Vec.capacity(vp)
    printf("Vector: size=%d, capacity=%d\n", sz, cap)

    for j in seq(sz):
        val: i64 = I64Vec.get(vp, j)
        printf("  v[%d] = %lld\n", j, val)

    I64Vec.destroy(vp)
    return 0


# ---------------------------------------------------------------------------
# Boundary: push exactly inline_capacity elements.
#
# At size == inline_capacity the small-vector is still inline; a wrong
# inline/heap predicate would read the heap union member (garbage) here.
# ---------------------------------------------------------------------------

@compile(suffix=(IntVec, 4, "boundary"))
def test_intvec_boundary() -> i32:
    v: IntVec
    vp = ptr(v)
    IntVec.init(vp)

    for i in seq(4):
        IntVec.push_back(vp, i)

    sz: i32 = i32(IntVec.size(vp))
    bad: i32 = 0
    j: i32 = 0
    while j < sz:
        if IntVec.get(vp, j) != j:
            bad = bad + 1
        j = j + 1

    d = IntVec.data(vp)
    if sz > 0:
        if d[0] != 0:
            bad = bad + 1
        if d[sz - 1] != (sz - 1):
            bad = bad + 1

    printf("boundary size=%d bad=%d\n", sz, bad)
    IntVec.destroy(vp)
    return bad


@compile(suffix=(I16Vec, 1, "boundary"))
def test_i16vec_boundary() -> i32:
    v: I16Vec
    vp = ptr(v)
    I16Vec.init(vp)

    for i in seq(1):
        I16Vec.push_back(vp, i16(i))

    sz: i32 = i32(I16Vec.size(vp))
    bad: i32 = 0
    j: i32 = 0
    while j < sz:
        if i32(I16Vec.get(vp, j)) != j:
            bad = bad + 1
        j = j + 1

    d = I16Vec.data(vp)
    if sz > 0:
        if i32(d[0]) != 0:
            bad = bad + 1
        if i32(d[sz - 1]) != (sz - 1):
            bad = bad + 1

    printf("boundary size=%d bad=%d\n", sz, bad)
    I16Vec.destroy(vp)
    return bad


@compile(suffix=(I64Vec, 2, "boundary"))
def test_i64vec_boundary() -> i32:
    v: I64Vec
    vp = ptr(v)
    I64Vec.init(vp)

    for i in seq(2):
        I64Vec.push_back(vp, i64(i))

    sz: i32 = i32(I64Vec.size(vp))
    bad: i32 = 0
    j: i32 = 0
    while j < sz:
        if i64(I64Vec.get(vp, j)) != i64(j):
            bad = bad + 1
        j = j + 1

    d = I64Vec.data(vp)
    if sz > 0:
        if d[0] != 0:
            bad = bad + 1
        if d[sz - 1] != (sz - 1):
            bad = bad + 1

    printf("boundary size=%d bad=%d\n", sz, bad)
    I64Vec.destroy(vp)
    return bad


# ---------------------------------------------------------------------------
# Spill-then-pop regression.
#
# Spilling is one-way: after the vector spills to the heap, pop_back may bring
# the element count back at or below inline_capacity.  The vector must keep
# using the heap branch (correct values, no leak in destroy), and further
# push_back must keep appending on the heap.
# ---------------------------------------------------------------------------

@compile(suffix=(IntVec, "spill_pop"))
def test_intvec_spill_pop() -> i32:
    v: IntVec
    vp = ptr(v)
    IntVec.init(vp)

    for i in seq(6):
        IntVec.push_back(vp, i)

    # Mutate while spilled; the inline buffer keeps the stale pre-spill copy.
    IntVec.set(vp, 0, 99)

    IntVec.pop_back(vp)
    IntVec.pop_back(vp)
    IntVec.pop_back(vp)

    bad: i32 = 0
    sz: i32 = i32(IntVec.size(vp))
    if sz != 3:
        bad = bad + 1
    if IntVec.get(vp, 0) != 99:
        bad = bad + 1
    j: i32 = 1
    while j < sz:
        if IntVec.get(vp, j) != j:
            bad = bad + 1
        j = j + 1

    d = IntVec.data(vp)
    d[0] = 42
    if IntVec.get(vp, 0) != 42:
        bad = bad + 1

    IntVec.push_back(vp, 7)
    if i32(IntVec.size(vp)) != 4:
        bad = bad + 1
    if IntVec.get(vp, 0) != 42:
        bad = bad + 1
    if IntVec.get(vp, 3) != 7:
        bad = bad + 1

    printf("spill_pop size=%d cap=%d bad=%d\n",
           i32(IntVec.size(vp)), i32(IntVec.capacity(vp)), bad)
    IntVec.destroy(vp)
    return bad


@compile(suffix=(I64Vec, "spill_pop"))
def test_i64vec_spill_pop() -> i32:
    v: I64Vec
    vp = ptr(v)
    I64Vec.init(vp)

    for i in seq(5):
        I64Vec.push_back(vp, i64(i))

    for i in seq(4):
        I64Vec.pop_back(vp)

    bad: i32 = 0
    sz: i32 = i32(I64Vec.size(vp))
    if sz != 1:
        bad = bad + 1
    j: i32 = 0
    while j < sz:
        if i64(I64Vec.get(vp, j)) != i64(j):
            bad = bad + 1
        j = j + 1

    I64Vec.push_back(vp, i64(9))
    if i32(I64Vec.size(vp)) != 2:
        bad = bad + 1
    if i64(I64Vec.get(vp, 1)) != i64(9):
        bad = bad + 1

    printf("spill_pop i64 size=%d cap=%d bad=%d\n",
           i32(I64Vec.size(vp)), i32(I64Vec.capacity(vp)), bad)
    I64Vec.destroy(vp)
    return bad


if __name__ == "__main__":
    test_intvec_base()
    test_i16vec_base()
    test_i64vec_base()

    failures = 0
    failures += int(test_intvec_boundary())
    failures += int(test_i16vec_boundary())
    failures += int(test_i64vec_boundary())
    failures += int(test_intvec_spill_pop())
    failures += int(test_i64vec_spill_pop())

    if failures:
        print("Vector boundary regression FAILED")
        sys.exit(1)

    print("All vector tests passed!")
