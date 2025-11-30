"""
Common refinement predicate generators

Provides generic predicate generators for common validation patterns.
Each generator takes a type T and returns:
1. A predicate function (T) -> bool
2. A refined type refined[predicate]

Usage:
    from pythoc.std.refine_wrapper import nonnull, positive, nonzero
    from pythoc import compile, ptr, i32, refined
    
    # Generate predicates for specific types
    is_valid_ptr, NonNullI32Ptr = nonnull(ptr[i32])
    is_positive_i32, PositiveI32 = positive(i32)
    is_nonzero_f64, NonZeroF64 = nonzero(f64)
    
    @compile
    def process_data(data: NonNullI32Ptr, count: PositiveI32) -> i32:
        return data[0] * count
"""

from ..decorators import compile
from ..builtin_entities import refined, ptr, nullptr, bool

# ============================================================================
# Predicate Generators
# ============================================================================

def nonnull_wrap(T):
    """Generate non-null predicate for pointer type T"""
    @compile(anonymous=True)
    def pred(p: T) -> bool:
        return p != nullptr
    
    return pred, refined[pred]


def positive_wrap(T):
    """Generate positive predicate for numeric type T"""
    @compile(anonymous=True)
    def pred(x: T) -> bool:
        return x > 0
    
    return pred, refined[pred]


def nonnegative_wrap(T):
    """Generate non-negative predicate for numeric type T"""
    @compile(anonymous=True)
    def pred(x: T) -> bool:
        return x >= 0
    
    return pred, refined[pred]


def nonzero_wrap(T):
    """Generate non-zero predicate for numeric type T"""
    @compile(anonymous=True)
    def pred(x: T) -> bool:
        return x != 0
    
    return pred, refined[pred]


def in_range_wrap(T, lower, upper):
    """Generate range predicate for numeric type T"""
    @compile(anonymous=True)
    def pred(x: T) -> bool:
        return x >= lower and x < upper
    
    return pred, refined[pred]
