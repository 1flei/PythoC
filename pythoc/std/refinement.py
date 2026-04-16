# varargs ABI v2
"""
Refinement predicate generators.

Provides generic predicate generators for common validation patterns.
Each generator takes a type T and returns:
1. A predicate function (T) -> bool
2. A refined type refined[predicate]

Usage:
    from pythoc.std.refinement import nonnull, positive, nonzero
    from pythoc import compile, ptr, i32

    is_valid_ptr, NonNullI32Ptr = nonnull(ptr[i32])
    is_positive_i32, PositiveI32 = positive(i32)

    @compile
    def process_data(data: NonNullI32Ptr, count: PositiveI32) -> i32:
        return data[0] * count
"""

from ..decorators import compile
from ..builtin_entities import refined, nullptr, bool


def nonnull(T):
    """Generate non-null predicate for pointer type T."""

    @compile(suffix=T)
    def pred(p: T) -> bool:
        return p != nullptr

    return pred, refined[pred]


def positive(T):
    """Generate positive predicate for numeric type T."""

    @compile(suffix=T)
    def pred(x: T) -> bool:
        return x > 0

    return pred, refined[pred]


def nonnegative(T):
    """Generate non-negative predicate for numeric type T."""

    @compile(suffix=T)
    def pred(x: T) -> bool:
        return x >= 0

    return pred, refined[pred]


def nonzero(T):
    """Generate non-zero predicate for numeric type T."""

    @compile(suffix=T)
    def pred(x: T) -> bool:
        return x != 0

    return pred, refined[pred]


def in_range(T, lower, upper):
    """Generate half-open range predicate for numeric type T: lower <= x < upper."""

    @compile(suffix=(T, lower, upper))
    def pred(x: T) -> bool:
        return x >= lower and x < upper

    return pred, refined[pred]
