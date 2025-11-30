#!/usr/bin/env python3
"""
Complex refined type tests

Tests advanced scenarios:
- Multiple predicates and tags in combination
- Type conversions between refined types
- Implicit conversions (more constraints -> fewer constraints)
- Expected compilation failures
"""

from pythoc import compile, i32, bool
from pythoc.builtin_entities import refined, assume, refine


# Define multiple predicates
@compile
def is_positive(x: i32) -> bool:
    return x > 0

@compile
def is_small(x: i32) -> bool:
    return x < 100

@compile
def is_even(x: i32) -> bool:
    return x % 2 == 0

@compile
def is_nonzero(x: i32) -> bool:
    return x != 0


# Test 1: Multiple predicates on single value
@compile
def test_multiple_predicates() -> i32:
    """Test value with multiple predicate constraints"""
    # x satisfies: positive AND small
    x = assume(42, is_positive, is_small)
    return x


@compile
def test_refine_multiple_predicates_success() -> i32:
    """Test refine with multiple predicates - all pass"""
    for x in refine(50, is_positive, is_small):
        return x
    else:
        return -1


@compile
def test_refine_multiple_predicates_fail() -> i32:
    """Test refine with multiple predicates - one fails"""
    # 150 is positive but not small
    for x in refine(150, is_positive, is_small):
        return x
    else:
        return -999


# Test 2: Predicates + tags combination
@compile
def test_predicates_with_tags() -> i32:
    """Test combining predicates and tags"""
    # x is positive, validated, and owned
    x = assume(10, is_positive, "validated", "owned")
    return x


@compile
def test_refine_predicates_with_tags() -> i32:
    """Test refine with predicates and tags"""
    for x in refine(20, is_positive, "checked"):
        return x
    else:
        return -1


# Test 3: Multiple tags only (always true)
@compile
def test_multiple_tags_only() -> i32:
    """Test refined type with only tags (no predicates)"""
    x = assume(100, "owned", "nonnull", "initialized")
    return x


@compile
def test_refine_tags_only() -> i32:
    """Test refine with only tags - always succeeds"""
    for x in refine(200, "special", "magic"):
        return x
    else:
        return -999  # Should never reach here


# Test 4: Complex combination
@compile
def test_complex_combination() -> i32:
    """Test complex combination: multiple predicates + multiple tags"""
    # Value must be: positive AND small AND even, plus has tags
    x = assume(42, is_positive, is_small, is_even, "validated", "trusted")
    return x


@compile
def test_refine_complex_success() -> i32:
    """Test complex refine - all predicates pass"""
    for x in refine(48, is_positive, is_small, is_even):
        return x
    else:
        return -1


@compile
def test_refine_complex_fail() -> i32:
    """Test complex refine - one predicate fails"""
    # 47 is positive and small but not even
    for x in refine(47, is_positive, is_small, is_even):
        return x
    else:
        return -999


# Test 5: Type conversions - implicit upcast
@compile
def accepts_positive(x: i32) -> i32:
    """Function that accepts plain i32"""
    return x + 1


@compile
def test_implicit_cast_to_base() -> i32:
    """Test implicit cast from refined type to base type"""
    x = assume(10, is_positive, "validated")
    # refined[i32, is_positive, "validated"] -> i32 (implicit)
    return accepts_positive(x)


# Test 6: Explicit cast with assume
@compile
def test_explicit_cast_add_constraints() -> i32:
    """Test explicit cast to add more constraints using assume"""
    x: i32 = 50
    # Add constraints: i32 -> refined[i32, is_positive, is_small]
    y = assume(x, is_positive, is_small)
    return y


@compile
def test_explicit_cast_change_tags() -> i32:
    """Test explicit cast to change tags"""
    x = assume(30, is_positive, "unvalidated")
    # Change tag: "unvalidated" -> "validated"
    y = assume(x, "validated")
    return y


# Test 7: Refine as type refinement
@compile
def test_refine_as_refinement() -> i32:
    """Test using refine to add runtime-checked constraints"""
    x: i32 = 75
    # Try to refine x to be both positive and small
    for y in refine(x, is_positive, is_small):
        return y  # Success: 75 is both positive and small
    else:
        return -1


@compile
def test_refine_refinement_fail() -> i32:
    """Test refine fails when constraints not met"""
    x: i32 = 150
    # Try to refine x to be both positive and small
    for y in refine(x, is_positive, is_small):
        return y
    else:
        return -999  # 150 is not small


# Test 8: Multi-param refined with additional single-param checks
@compile
def is_valid_range(start: i32, end: i32) -> bool:
    return start <= end and start >= 0


@compile
def test_multiarg_basic() -> i32:
    """Test multi-arg refined type creation"""
    r = assume(10, 20, is_valid_range)
    return r.start + r.end


@compile
def test_multiarg_refine_pass() -> i32:
    """Test multi-arg refine success"""
    for r in refine(5, 15, is_valid_range):
        return r.start + r.end
    else:
        return -1


@compile
def test_multiarg_refine_fail() -> i32:
    """Test multi-arg refine failure"""
    for r in refine(20, 10, is_valid_range):
        return r.start + r.end
    else:
        return -999


# Test 9: Nested refine checks
@compile
def test_nested_refine() -> i32:
    """Test nested refine - checking multiple constraints in sequence"""
    x: i32 = 48
    
    # First check: is it positive?
    for y in refine(x, is_positive):
        # Second check: is it also small?
        for z in refine(y, is_small):
            # Third check: is it even?
            for w in refine(z, is_even):
                return w  # All checks passed
            else:
                return -3
        else:
            return -2
    else:
        return -1


@compile
def test_nested_refine_early_fail() -> i32:
    """Test nested refine with early failure"""
    x: i32 = -5
    
    for y in refine(x, is_positive):
        for z in refine(y, is_small):
            return z
        else:
            return -2
    else:
        return -999  # Fails at first check


# Test 10: Safe division with refined types
@compile
def safe_divide_refined(a: i32, b: i32) -> i32:
    """Safe division using refined type to ensure non-zero"""
    for divisor in refine(b, is_nonzero):
        return a / divisor
    else:
        return 0


@compile
def test_safe_divide_refined_valid() -> i32:
    return safe_divide_refined(100, 4)  # 25


@compile
def test_safe_divide_refined_zero() -> i32:
    return safe_divide_refined(100, 0)  # 0


# Test 11: Combining tags from different sources
@compile
def test_tag_accumulation() -> i32:
    """Test that tags can be added through multiple assume calls"""
    x: i32 = 50
    # Start with one tag
    y = assume(x, "initialized")
    # Add more tags (re-assume to change tags)
    z = assume(y, "validated", "trusted")
    return z


def main():
    """Run all complex refined type tests"""
    print("Testing complex refined type scenarios...")
    print()
    
    # Multiple predicates
    result = test_multiple_predicates()
    print(f"test_multiple_predicates: {'PASS' if result == 42 else 'FAIL'} (result={result}, expected=42)")
    assert result == 42
    
    result = test_refine_multiple_predicates_success()
    print(f"test_refine_multiple_predicates_success: {'PASS' if result == 50 else 'FAIL'} (result={result}, expected=50)")
    assert result == 50
    
    result = test_refine_multiple_predicates_fail()
    print(f"test_refine_multiple_predicates_fail: {'PASS' if result == -999 else 'FAIL'} (result={result}, expected=-999)")
    assert result == -999
    
    # Predicates + tags
    result = test_predicates_with_tags()
    print(f"test_predicates_with_tags: {'PASS' if result == 10 else 'FAIL'} (result={result}, expected=10)")
    assert result == 10
    
    result = test_refine_predicates_with_tags()
    print(f"test_refine_predicates_with_tags: {'PASS' if result == 20 else 'FAIL'} (result={result}, expected=20)")
    assert result == 20
    
    # Multiple tags only
    result = test_multiple_tags_only()
    print(f"test_multiple_tags_only: {'PASS' if result == 100 else 'FAIL'} (result={result}, expected=100)")
    assert result == 100
    
    result = test_refine_tags_only()
    print(f"test_refine_tags_only: {'PASS' if result == 200 else 'FAIL'} (result={result}, expected=200)")
    assert result == 200
    
    # Complex combinations
    result = test_complex_combination()
    print(f"test_complex_combination: {'PASS' if result == 42 else 'FAIL'} (result={result}, expected=42)")
    assert result == 42
    
    result = test_refine_complex_success()
    print(f"test_refine_complex_success: {'PASS' if result == 48 else 'FAIL'} (result={result}, expected=48)")
    assert result == 48
    
    result = test_refine_complex_fail()
    print(f"test_refine_complex_fail: {'PASS' if result == -999 else 'FAIL'} (result={result}, expected=-999)")
    assert result == -999
    
    # Type conversions
    result = test_implicit_cast_to_base()
    print(f"test_implicit_cast_to_base: {'PASS' if result == 11 else 'FAIL'} (result={result}, expected=11)")
    assert result == 11
    
    result = test_explicit_cast_add_constraints()
    print(f"test_explicit_cast_add_constraints: {'PASS' if result == 50 else 'FAIL'} (result={result}, expected=50)")
    assert result == 50
    
    result = test_explicit_cast_change_tags()
    print(f"test_explicit_cast_change_tags: {'PASS' if result == 30 else 'FAIL'} (result={result}, expected=30)")
    assert result == 30
    
    # Refine as refinement
    result = test_refine_as_refinement()
    print(f"test_refine_as_refinement: {'PASS' if result == 75 else 'FAIL'} (result={result}, expected=75)")
    assert result == 75
    
    result = test_refine_refinement_fail()
    print(f"test_refine_refinement_fail: {'PASS' if result == -999 else 'FAIL'} (result={result}, expected=-999)")
    assert result == -999
    
    # Multi-param
    result = test_multiarg_basic()
    print(f"test_multiarg_basic: {'PASS' if result == 30 else 'FAIL'} (result={result}, expected=30)")
    assert result == 30
    
    result = test_multiarg_refine_pass()
    print(f"test_multiarg_refine_pass: {'PASS' if result == 20 else 'FAIL'} (result={result}, expected=20)")
    assert result == 20
    
    result = test_multiarg_refine_fail()
    print(f"test_multiarg_refine_fail: {'PASS' if result == -999 else 'FAIL'} (result={result}, expected=-999)")
    assert result == -999
    
    # Nested refine
    result = test_nested_refine()
    print(f"test_nested_refine: {'PASS' if result == 48 else 'FAIL'} (result={result}, expected=48)")
    assert result == 48
    
    result = test_nested_refine_early_fail()
    print(f"test_nested_refine_early_fail: {'PASS' if result == -999 else 'FAIL'} (result={result}, expected=-999)")
    assert result == -999
    
    # Safe divide
    result = test_safe_divide_refined_valid()
    print(f"test_safe_divide_refined_valid: {'PASS' if result == 25 else 'FAIL'} (result={result}, expected=25)")
    assert result == 25
    
    result = test_safe_divide_refined_zero()
    print(f"test_safe_divide_refined_zero: {'PASS' if result == 0 else 'FAIL'} (result={result}, expected=0)")
    assert result == 0
    
    # Tag accumulation
    result = test_tag_accumulation()
    print(f"test_tag_accumulation: {'PASS' if result == 50 else 'FAIL'} (result={result}, expected=50)")
    assert result == 50
    
    print()
    print(f"All {22} complex refined type tests passed!")
    return 0


if __name__ == "__main__":
    main()
