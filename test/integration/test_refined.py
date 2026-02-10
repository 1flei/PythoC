#!/usr/bin/env python3
"""
Test refined types with predicate validation

Tests the refined type system including:
- Type definition with refined[Pred]
- assume() for unchecked construction
- refine() for runtime validation with for-else
- Field access on refined types
"""

from pythoc import compile, i32, bool, refined, refine


# Basic single-parameter refined type
@compile
def is_positive(x: i32) -> bool:
    return x > 0

PositiveInt = refined[is_positive]


from pythoc import assume

@compile
def test_assume_positive() -> i32:
    """Test assume: create refined type without checking"""
    x = assume(5, is_positive)
    return x  # Single-param: use directly


@compile
def test_refine_success() -> i32:
    """Test refine with valid value (for-else takes for branch)"""
    for x in refine(10, is_positive):
        return x  # Single-param: use directly
    else:
        return -1


@compile
def test_refine_failure() -> i32:
    """Test refine with invalid value (for-else takes else branch)"""
    for x in refine(-5, is_positive):
        return x  # Single-param: use directly
    else:
        return -999  # Should return this


# Multi-parameter refined type
@compile
def is_valid_range(start: i32, end: i32) -> bool:
    return start <= end and start >= 0

ValidRange = refined[is_valid_range]


@compile
def test_range_assume() -> i32:
    """Test assume with multi-parameter refined type"""
    r = assume(10, 20, is_valid_range)
    return r.start + r.end  # 30


@compile
def test_range_refine_valid() -> i32:
    """Test refine with valid range"""
    for r in refine(5, 15, is_valid_range):
        return r.start + r.end
    else:
        return -1


@compile
def test_range_refine_invalid_order() -> i32:
    """Test refine with invalid range (start > end)"""
    for r in refine(20, 10, is_valid_range):
        return r.start + r.end
    else:
        return -999


@compile
def test_range_refine_invalid_negative() -> i32:
    """Test refine with negative start"""
    for r in refine(-5, 10, is_valid_range):
        return r.start + r.end
    else:
        return -999


# Test field access by index
@compile
def test_range_index_access() -> i32:
    """Test accessing refined type fields by index"""
    r = assume(3, 7, is_valid_range)
    return r[0] + r[1]  # 10


@compile
def is_nonzero(x: i32) -> bool:
    return x != 0


# Test using refined type in computation
@compile
def safe_divide(dividend: i32, divisor: i32) -> i32:
    """Safely divide using refined type to ensure non-zero divisor"""
    for d in refine(divisor, is_nonzero):
        result: i32 = dividend / d  # Single-param: use directly
        return result
    else:
        return 0  # Return 0 if divisor is zero


@compile
def test_safe_divide_valid() -> i32:
    """Test safe division with non-zero divisor"""
    return safe_divide(100, 5)  # 20


@compile
def test_safe_divide_zero() -> i32:
    """Test safe division with zero divisor"""
    return safe_divide(100, 0)  # 0


# Test nested refined types
@compile
def is_even(x: i32) -> bool:
    return x % 2 == 0

EvenInt = refined[is_even]


@compile
def is_positive_even(x: i32) -> bool:
    return x > 0 and x % 2 == 0

PositiveEven = refined[is_positive_even]


@compile
def test_positive_even() -> i32:
    """Test refined type with combined predicates"""
    for x in refine(8, is_positive_even):
        return x  # Single-param: use directly
    else:
        return -1


@compile
def test_positive_even_fail_negative() -> i32:
    """Test that negative even number fails positive_even"""
    for x in refine(-4, is_positive_even):
        return x  # Single-param: use directly
    else:
        return -999


@compile
def test_positive_even_fail_odd() -> i32:
    """Test that odd number fails positive_even"""
    for x in refine(7, is_positive_even):
        return x  # Single-param: use directly
    else:
        return -999


# Test constructor syntax (alternative to assume)
@compile
def test_constructor_syntax() -> i32:
    """Test using refined[Pred](...) as constructor"""
    x = PositiveInt(42)
    return x  # Single-param: use directly


@compile
def test_range_constructor() -> i32:
    """Test multi-param constructor syntax"""
    r = ValidRange(1, 100)
    return r.start + r.end  # 101


# Complex example: array bounds checking
@compile
def is_valid_index(idx: i32, length: i32) -> bool:
    return idx >= 0 and idx < length

ValidIndex = refined[is_valid_index]


@compile
def get_element_at(idx: i32, arr_len: i32) -> i32:
    """Get element at index with bounds checking"""
    for valid_idx in refine(idx, arr_len, is_valid_index):
        # In real code, would access array[valid_idx.idx]
        return valid_idx[0] * 10  # Just return idx * 10 for testing
    else:
        return -1  # Index out of bounds


@compile
def test_valid_index_in_bounds() -> i32:
    """Test valid array index"""
    return get_element_at(3, 10)  # 30


@compile
def test_valid_index_out_of_bounds() -> i32:
    """Test invalid array index (>= length)"""
    return get_element_at(15, 10)  # -1


@compile
def test_valid_index_negative() -> i32:
    """Test negative array index"""
    return get_element_at(-1, 10)  # -1


import unittest


class TestRefined(unittest.TestCase):
    """Test refined types with predicate validation"""

    def test_assume_positive(self):
        """Test assume: create refined type without checking"""
        self.assertEqual(test_assume_positive(), 5)

    def test_refine_success(self):
        """Test refine with valid value (for-else takes for branch)"""
        self.assertEqual(test_refine_success(), 10)

    def test_refine_failure(self):
        """Test refine with invalid value (for-else takes else branch)"""
        self.assertEqual(test_refine_failure(), -999)

    def test_range_assume(self):
        """Test assume with multi-parameter refined type"""
        self.assertEqual(test_range_assume(), 30)

    def test_range_refine_valid(self):
        """Test refine with valid range"""
        self.assertEqual(test_range_refine_valid(), 20)

    def test_range_refine_invalid_order(self):
        """Test refine with invalid range (start > end)"""
        self.assertEqual(test_range_refine_invalid_order(), -999)

    def test_range_refine_invalid_negative(self):
        """Test refine with negative start"""
        self.assertEqual(test_range_refine_invalid_negative(), -999)

    def test_range_index_access(self):
        """Test accessing refined type fields by index"""
        self.assertEqual(test_range_index_access(), 10)

    def test_safe_divide_valid(self):
        """Test safe division with non-zero divisor"""
        self.assertEqual(test_safe_divide_valid(), 20)

    def test_safe_divide_zero(self):
        """Test safe division with zero divisor"""
        self.assertEqual(test_safe_divide_zero(), 0)

    def test_positive_even(self):
        """Test refined type with combined predicates"""
        self.assertEqual(test_positive_even(), 8)

    def test_positive_even_fail_negative(self):
        """Test that negative even number fails positive_even"""
        self.assertEqual(test_positive_even_fail_negative(), -999)

    def test_positive_even_fail_odd(self):
        """Test that odd number fails positive_even"""
        self.assertEqual(test_positive_even_fail_odd(), -999)

    def test_constructor_syntax(self):
        """Test using refined[Pred](...) as constructor"""
        self.assertEqual(test_constructor_syntax(), 42)

    def test_range_constructor(self):
        """Test multi-param constructor syntax"""
        self.assertEqual(test_range_constructor(), 101)

    def test_valid_index_in_bounds(self):
        """Test valid array index"""
        self.assertEqual(test_valid_index_in_bounds(), 30)

    def test_valid_index_out_of_bounds(self):
        """Test invalid array index (>= length)"""
        self.assertEqual(test_valid_index_out_of_bounds(), -1)

    def test_valid_index_negative(self):
        """Test negative array index"""
        self.assertEqual(test_valid_index_negative(), -1)


if __name__ == "__main__":
    unittest.main()
