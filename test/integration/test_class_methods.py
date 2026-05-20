"""Integration tests for class-body methods on @compile/@union/@enum.

The decorated class supports plain ``def f(...)`` members. Each such method
behaves exactly as if it had been written as a sibling ``@compile`` function
and manually attached as a class attribute (the existing pattern from
``pythoc/std/vector.py``). Methods do not receive ``self``; the first
parameter is a normal pythoc parameter.
"""
from __future__ import annotations

from pythoc import (
    i16, i32, i64, u32, u64, f64, ptr, compile, struct, union, enum, seq,
    sizeof,
)
from pythoc.libc.stdio import printf
from pythoc.libc.stdlib import malloc, free
from pythoc.libc.string import memset


# ---------------------------------------------------------------------------
# struct: methods on a @compile class
# ---------------------------------------------------------------------------

@compile
class Point:
    x: i32
    y: i32

    def make(px: i32, py: i32) -> "Point":
        p: Point
        p.x = px
        p.y = py
        return p

    def add(a: ptr["Point"], b: ptr["Point"]) -> "Point":
        out: Point
        out.x = a.x + b.x
        out.y = a.y + b.y
        return out

    def magnitude_sq(p: ptr["Point"]) -> i32:
        return p.x * p.x + p.y * p.y


@compile
def test_struct_methods() -> i32:
    a: Point = Point.make(3, 4)
    b: Point = Point.make(1, 2)

    pa = ptr(a)
    pb = ptr(b)

    sum_pt: Point = Point.add(pa, pb)
    msq: i32 = Point.magnitude_sq(pa)

    printf("Point sum: (%d, %d)\n", sum_pt.x, sum_pt.y)
    printf("Point magnitude_sq(a): %d\n", msq)
    return 0


@compile
class NamedA:
    x: i32

    def make(v: i32) -> "NamedA":
        a: NamedA
        a.x = v
        return a


@compile
class NamedB:
    x: i64
    y: i64

    def make(x: i64, y: i64) -> "NamedB":
        b: NamedB
        b.x = x
        b.y = y
        return b


@compile
def test_same_method_name_different_classes() -> i32:
    a: NamedA = NamedA.make(7)
    b: NamedB = NamedB.make(10, 20)
    return a.x + i32(b.x + b.y)


# ---------------------------------------------------------------------------
# union: methods on a @union class
# ---------------------------------------------------------------------------

@union
class Number:
    as_int: i64
    as_float: f64

    def from_int(v: i64) -> "Number":
        n: Number
        n.as_int = v
        return n

    def reinterpret_as_float(n: ptr["Number"]) -> f64:
        return n.as_float


@compile
def test_union_methods() -> i32:
    n: Number = Number.from_int(0x4045000000000000)
    pn = ptr(n)
    f: f64 = Number.reinterpret_as_float(pn)
    printf("Union reinterpret: int=%lld -> float=%f\n", n.as_int, f)
    return 0


# ---------------------------------------------------------------------------
# enum: methods on an @enum class (FunctionDef must not be parsed as variant)
# ---------------------------------------------------------------------------

@enum(i32)
class Color:
    Red: None
    Green: None
    Blue: None

    def is_red(c: "Color") -> i32:
        if c[0] == Color.Red:
            return 1
        return 0

    def to_code(c: "Color") -> i32:
        if c[0] == Color.Red:
            return 100
        if c[0] == Color.Green:
            return 200
        return 300


@compile
def test_enum_methods() -> i32:
    r: Color = Color.Red
    g: Color = Color.Green
    b: Color = Color.Blue

    printf("Color is_red(Red)=%d\n", Color.is_red(r))
    printf("Color is_red(Green)=%d\n", Color.is_red(g))
    printf("Color to_code(Blue)=%d\n", Color.to_code(b))
    return 0


# ---------------------------------------------------------------------------
# Generic factory: methods inherit class compile_suffix; reference enclosing
# class via ptr[Self] and capture closure variables.
# ---------------------------------------------------------------------------

def make_box(elem_type):
    @compile(suffix=elem_type)
    class Box:
        value: elem_type

        def make(v: elem_type) -> "Box":
            b: Box
            b.value = v
            return b

        def get(b: ptr["Box"]) -> elem_type:
            return b.value

        def set(b: ptr["Box"], v: elem_type) -> None:
            b.value = v
    return Box


IntBox = make_box(i32)
LongBox = make_box(i64)


@compile
def test_generic_box() -> i32:
    a: IntBox = IntBox.make(123)
    b: LongBox = LongBox.make(45678901234)

    pa = ptr(a)
    pb = ptr(b)

    IntBox.set(pa, 999)
    printf("IntBox: %d\n", IntBox.get(pa))
    printf("LongBox: %lld\n", LongBox.get(pb))
    return 0


# ---------------------------------------------------------------------------
# Method explicitly decorated with @compile(suffix=...) overrides class suffix.
# This proves the escape hatch: an already-compiled member is attached as-is.
# ---------------------------------------------------------------------------

@compile(suffix="custompair")
class Pair:
    a: i32
    b: i32

    @compile(suffix="explicit_sum")
    def sum(p: ptr["Pair"]) -> i32:
        return p.a + p.b

    def diff(p: ptr["Pair"]) -> i32:
        return p.a - p.b


@compile
def test_explicit_method_suffix() -> i32:
    p: Pair
    p.a = 10
    p.b = 3
    pp = ptr(p)
    printf("Pair sum=%d, diff=%d\n", Pair.sum(pp), Pair.diff(pp))
    return 0


# ---------------------------------------------------------------------------
# Vector-style: methods call other methods on the same class.
# ---------------------------------------------------------------------------

def make_stack(elem_type, capacity):
    @compile(suffix=(elem_type, capacity))
    class Stack:
        size: u32
        data: ptr[elem_type]

        def init(s: ptr["Stack"]) -> None:
            s.size = 0
            s.data = malloc(capacity * sizeof(elem_type))

        def destroy(s: ptr["Stack"]) -> None:
            free(s.data)
            s.size = 0

        def push(s: ptr["Stack"], v: elem_type) -> None:
            s.data[s.size] = v
            s.size += 1

        def top(s: ptr["Stack"]) -> elem_type:
            return s.data[s.size - 1]

        def pop_top(s: ptr["Stack"]) -> elem_type:
            v: elem_type = Stack.top(s)
            s.size -= 1
            return v
    return Stack


IntStack = make_stack(i32, 8)


@compile
def test_self_calls() -> i32:
    s: IntStack
    sp = ptr(s)
    IntStack.init(sp)
    for i in seq(5):
        IntStack.push(sp, i + 10)
    printf("Stack top=%d\n", IntStack.top(sp))
    printf("Stack pop=%d\n", IntStack.pop_top(sp))
    printf("Stack pop=%d\n", IntStack.pop_top(sp))
    IntStack.destroy(sp)
    return 0


@compile
def main() -> i32:
    test_struct_methods()
    if test_same_method_name_different_classes() != 37:
        return 1
    test_union_methods()
    test_enum_methods()
    test_generic_box()
    test_explicit_method_suffix()
    test_self_calls()
    return 0


if __name__ == "__main__":
    main()
    print("All class method tests passed!")
