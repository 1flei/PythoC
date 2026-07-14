#!/usr/bin/env python3
"""Static aggregate initializers with compile-time constants."""

from pythoc import compile, i32, i8, u32, u64, ptr, static, array, nullptr, offsetof, sizeof, char, void
from pythoc import enum


@enum(i32)
class Color:
    Red: None
    Green: None
    Blue: None


@compile
class State:
    a: i32
    b: i32
    c: i32


@compile
class S:
    a: ptr[i8]
    b: i32


@compile
class FlagDef:
    offset: i32
    flags: i32
    name: ptr[i8]


@compile
class Meta:
    size: i32
    name: ptr[i8]


@compile
class Record:
    tag: i32
    ch: i8
    name: ptr[i8]


@compile
class Entry:
    offset: i32
    name: ptr[i8]


@compile
def test_static_array_partial_string() -> i32:
    s: static[array[S, 3]] = ((ptr[i8]("x"), 1),)
    return s[0].b


@compile
def test_static_array_offsetof() -> i32:
    s: static[array[FlagDef, 2]] = (
        (offsetof("State", "a"), 1, ptr[i8]("a")),
        (offsetof("State", "c"), 0, ptr[i8]("c")),
    )
    return s[0].offset


@compile
def test_static_array_offsetof_arith() -> i32:
    s: static[array[i32, 2]] = (offsetof("State", "b") + 2, 0)
    return s[0]


@compile
def test_static_array_sizeof() -> i32:
    s: static[array[Meta, 2]] = (
        (sizeof(State), ptr[i8]("state")),
        (sizeof(i32), ptr[i8]("i32")),
    )
    return s[0].size


@compile
def test_static_array_sizeof_arith() -> i32:
    s: static[array[i32, 3]] = (sizeof(State) + 1, sizeof(i32) * 2, 0)
    return s[0] + s[1]


@compile
def test_static_array_enum_const() -> i32:
    s: static[array[i32, 3]] = (Color.Red, Color.Green, Color.Blue)
    return s[0] + s[1] + s[2]


@compile
def test_static_array_char_const() -> i32:
    s: static[array[i8, 3]] = (char('A'), char('B'), char(0))
    return s[0]


@compile
def test_static_array_cast_const() -> i32:
    s: static[array[i32, 2]] = (u32(5), i32(sizeof(State)))
    return s[0] + s[1]


@compile
def test_static_array_neg_const() -> i32:
    s: static[array[i32, 2]] = (-5, -sizeof(State))
    return s[0] + s[1]


@compile
def test_static_array_record_const() -> i32:
    s: static[array[Record, 2]] = (
        (1, char('X'), ptr[i8]("x")),
        (2, char('Y'), nullptr),
    )
    return s[0].tag + s[1].tag


@compile
def test_static_array_null_cast_void() -> i32:
    s: static[array[Entry, 2]] = (
        (0, ptr[i8]("all")),
        (0, ptr[void](0)),
    )
    return s[0].offset + s[1].offset


@compile
def test_static_array_null_cast_i8() -> i32:
    s: static[array[Entry, 2]] = (
        (0, ptr[i8]("all")),
        (0, ptr[i8](0)),
    )
    return s[0].offset + s[1].offset


@compile
def test_static_null_ptr_cast() -> i32:
    p: static[ptr[i8]] = ptr[i8](0)
    return i32(p == nullptr)


@compile
def test_static_int_to_ptr_addr() -> u64:
    p: static[ptr[void]] = ptr[void](0x1234)
    return u64(p)


@compile
def test_static_array_int_to_ptr() -> u64:
    s: static[array[Entry, 2]] = (
        (0, ptr[i8]("all")),
        (0, ptr[i8](0xABCD)),
    )
    return u64(s[1].name)


if __name__ == "__main__":
    assert test_static_array_partial_string() == 1
    assert test_static_array_offsetof() == 0
    assert test_static_array_offsetof_arith() == 6
    assert test_static_array_sizeof() == 12
    assert test_static_array_sizeof_arith() == 21
    assert test_static_array_enum_const() == 3
    assert test_static_array_char_const() == 65
    assert test_static_array_cast_const() == 17
    assert test_static_array_neg_const() == -17
    assert test_static_array_record_const() == 3
    assert test_static_array_null_cast_void() == 0
    assert test_static_array_null_cast_i8() == 0
    assert test_static_null_ptr_cast() == 1
    assert test_static_int_to_ptr_addr() == 0x1234
    assert test_static_array_int_to_ptr() == 0xABCD
    print("All static aggregate init tests passed!")
