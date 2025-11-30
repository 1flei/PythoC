#!/usr/bin/env python3
"""
Test refined types with predicate validation

Tests the refined type system including:
- Type definition with refined[Pred]
- assume() for unchecked construction
- refine() for runtime validation with for-else
- Field access on refined types
"""

from pythoc import compile, i32, bool, refined, assume, refine


# Basic single-parameter refined type
@compile
def is_positive(x: i32) -> bool:
    return x > 0

PositiveInt = refined[is_positive]


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


def main():
    """Run all tests"""
    print("Testing refined types...")
    print()
    
    # Test assume
    result = test_assume_positive()
    print(f"test_assume_positive: {'PASS' if result == 5 else 'FAIL'} (result={result}, expected=5)")
    assert result == 5
    
    # Test refine success
    result = test_refine_success()
    print(f"test_refine_success: {'PASS' if result == 10 else 'FAIL'} (result={result}, expected=10)")
    assert result == 10
    
    # Test refine failure
    result = test_refine_failure()
    print(f"test_refine_failure: {'PASS' if result == -999 else 'FAIL'} (result={result}, expected=-999)")
    assert result == -999
    
    # Test multi-parameter assume
    result = test_range_assume()
    print(f"test_range_assume: {'PASS' if result == 30 else 'FAIL'} (result={result}, expected=30)")
    assert result == 30
    
    # Test multi-parameter refine valid
    result = test_range_refine_valid()
    print(f"test_range_refine_valid: {'PASS' if result == 20 else 'FAIL'} (result={result}, expected=20)")
    assert result == 20
    
    # Test multi-parameter refine invalid (order)
    result = test_range_refine_invalid_order()
    print(f"test_range_refine_invalid_order: {'PASS' if result == -999 else 'FAIL'} (result={result}, expected=-999)")
    assert result == -999
    
    # Test multi-parameter refine invalid (negative)
    result = test_range_refine_invalid_negative()
    print(f"test_range_refine_invalid_negative: {'PASS' if result == -999 else 'FAIL'} (result={result}, expected=-999)")
    assert result == -999
    
    # Test index access
    result = test_range_index_access()
    print(f"test_range_index_access: {'PASS' if result == 10 else 'FAIL'} (result={result}, expected=10)")
    assert result == 10
    
    # Test safe divide
    result = test_safe_divide_valid()
    print(f"test_safe_divide_valid: {'PASS' if result == 20 else 'FAIL'} (result={result}, expected=20)")
    assert result == 20
    
    result = test_safe_divide_zero()
    print(f"test_safe_divide_zero: {'PASS' if result == 0 else 'FAIL'} (result={result}, expected=0)")
    assert result == 0
    
    # Test positive even
    result = test_positive_even()
    print(f"test_positive_even: {'PASS' if result == 8 else 'FAIL'} (result={result}, expected=8)")
    assert result == 8
    
    result = test_positive_even_fail_negative()
    print(f"test_positive_even_fail_negative: {'PASS' if result == -999 else 'FAIL'} (result={result}, expected=-999)")
    assert result == -999
    
    result = test_positive_even_fail_odd()
    print(f"test_positive_even_fail_odd: {'PASS' if result == -999 else 'FAIL'} (result={result}, expected=-999)")
    assert result == -999
    
    # Test constructor syntax
    result = test_constructor_syntax()
    print(f"test_constructor_syntax: {'PASS' if result == 42 else 'FAIL'} (result={result}, expected=42)")
    assert result == 42
    
    result = test_range_constructor()
    print(f"test_range_constructor: {'PASS' if result == 101 else 'FAIL'} (result={result}, expected=101)")
    assert result == 101
    
    # Test valid index
    result = test_valid_index_in_bounds()
    print(f"test_valid_index_in_bounds: {'PASS' if result == 30 else 'FAIL'} (result={result}, expected=30)")
    assert result == 30
    
    result = test_valid_index_out_of_bounds()
    print(f"test_valid_index_out_of_bounds: {'PASS' if result == -1 else 'FAIL'} (result={result}, expected=-1)")
    assert result == -1
    
    result = test_valid_index_negative()
    print(f"test_valid_index_negative: {'PASS' if result == -1 else 'FAIL'} (result={result}, expected=-1)")
    assert result == -1
    
    print()
    print("All refined type tests passed!")
    return 0


if __name__ == "__main__":
    main()
