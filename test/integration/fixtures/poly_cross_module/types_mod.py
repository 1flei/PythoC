"""Shared sum type for independent dispatchers in separate modules."""

from pythoc import enum, f64, i32


@enum(i32)
class Input:
    Integer: i32
    Float: f64
