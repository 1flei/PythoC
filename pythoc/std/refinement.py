# varargs ABI v2
"""
Refinement predicate generators.

Provides generic predicate generators for common validation patterns.
Each generator returns a refined type exposing its predicate as a method.

Usage:
    from pythoc.std.refinement import nonnull, positive, nonzero
    from pythoc import compile, ptr, i32

    NonNullI32Ptr = nonnull(ptr[i32])
    PositiveI32 = positive(i32)

    @compile
    def process_data(data: NonNullI32Ptr, count: PositiveI32) -> i32:
        return data[0] * count
"""

from ..decorators import compile
from ..builtin_entities import refined, nullptr, bool


def _refinement_result(predicate, kind, *parameters):
    """Declare a fixed predicate contract on its existing callable object."""
    predicate._pc_refinement_contract = ('pythoc.std.refinement', kind, parameters)
    return refined[predicate]


def nonnull(T):
    """Generate non-null predicate for pointer type T."""

    @compile(suffix=('nonnull', T))
    def pred(p: T) -> bool:
        return p != nullptr

    return _refinement_result(pred, 'nonnull')


def positive(T):
    """Generate positive predicate for numeric type T."""

    @compile(suffix=('positive', T))
    def pred(x: T) -> bool:
        return x > 0

    return _refinement_result(pred, 'positive')


def nonnegative(T):
    """Generate non-negative predicate for numeric type T."""

    @compile(suffix=('nonnegative', T))
    def pred(x: T) -> bool:
        return x >= 0

    return _refinement_result(pred, 'nonnegative')


def nonzero(T):
    """Generate non-zero predicate for numeric type T."""

    @compile(suffix=('nonzero', T))
    def pred(x: T) -> bool:
        return x != 0

    return _refinement_result(pred, 'nonzero')


def in_range(T, lower, upper):
    """Generate half-open range predicate for numeric type T: lower <= x < upper."""

    @compile(suffix=('in_range', T, lower, upper))
    def pred(x: T) -> bool:
        return x >= lower and x < upper

    return _refinement_result(pred, 'in_range', lower, upper)
