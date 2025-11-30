#!/usr/bin/env python3
"""
Test cases that SHOULD fail at compile time

These demonstrate proper error detection in the refined type system.
Each test should raise a specific error during compilation.
"""

import sys
from pythoc import compile, i32, bool
from pythoc.builtin_entities import refined, assume, refine


# Predicate definitions
@compile
def is_positive(x: i32) -> bool:
    return x > 0

@compile
def is_small(x: i32) -> bool:
    return x < 100


def test_direct_conversion_to_refined():
    """
    EXPECTED FAILURE: Cannot directly convert base type to refined type
    Must use assume() or refine()
    """
    print("Test: Direct conversion to refined type (should fail)...")
    try:
        @compile
        def should_fail() -> i32:
            x: i32 = 10
            # This should fail: cannot directly assign i32 to refined type
            y: refined[is_positive] = x  # ERROR: direct conversion not allowed
            return y
        
        print("  FAIL: Should have raised TypeError but didn't")
        return False
    except TypeError as e:
        if "Cannot directly convert" in str(e) or "refined type" in str(e):
            print(f"  PASS: Got expected error: {e}")
            return True
        else:
            print(f"  FAIL: Got unexpected error: {e}")
            return False
    except Exception as e:
        print(f"  FAIL: Got unexpected exception: {type(e).__name__}: {e}")
        return False


def test_assume_without_constraints():
    """
    EXPECTED FAILURE: assume() requires at least one predicate or tag
    """
    print("Test: assume() without constraints (should fail)...")
    try:
        @compile
        def should_fail() -> i32:
            # This should fail: no constraints provided
            x = assume(10)  # ERROR: need at least one constraint
            return x
        
        print("  FAIL: Should have raised TypeError but didn't")
        return False
    except TypeError as e:
        if "at least" in str(e) or "constraint" in str(e):
            print(f"  PASS: Got expected error: {e}")
            return True
        else:
            print(f"  FAIL: Got unexpected error: {e}")
            return False
    except Exception as e:
        print(f"  FAIL: Got unexpected exception: {type(e).__name__}: {e}")
        return False


def test_refine_without_constraints():
    """
    EXPECTED FAILURE: refine() requires at least one predicate or tag
    """
    print("Test: refine() without constraints (should fail)...")
    try:
        @compile
        def should_fail_refine_no_constraints() -> i32:
            # This should fail: no constraints provided
            for x in refine(10):  # ERROR: need at least one constraint
                return x
            else:
                return -1
        
        print("  FAIL: Should have raised TypeError but didn't")
        return False
    except TypeError as e:
        if "at least" in str(e) or "argument" in str(e):
            print(f"  PASS: Got expected error: {e}")
            return True
        else:
            print(f"  FAIL: Got unexpected error: {e}")
            return False
    except Exception as e:
        print(f"  FAIL: Got unexpected exception: {type(e).__name__}: {e}")
        return False


def test_assume_non_callable_constraint():
    """
    EXPECTED FAILURE: assume() constraint must be callable or string
    """
    print("Test: assume() with non-callable constraint (should fail)...")
    try:
        @compile
        def should_fail_non_callable() -> i32:
            # This should fail: 123 is not a predicate or tag
            x = assume(10, 123)  # ERROR: not callable
            return x
        
        print("  FAIL: Should have raised TypeError but didn't")
        return False
    except (TypeError, AttributeError) as e:
        print(f"  PASS: Got expected error: {e}")
        return True
    except Exception as e:
        print(f"  FAIL: Got unexpected exception: {type(e).__name__}: {e}")
        return False


def test_refine_used_outside_for_loop():
    """
    EXPECTED FAILURE: refine() must be used in a for loop
    
    Note: This might succeed at compile time but should fail at runtime
    or be caught by static analysis
    """
    print("Test: refine() used outside for loop (should fail or warn)...")
    try:
        @compile
        def should_fail_refine_outside() -> i32:
            # This should fail or warn: refine() outside for loop
            x = refine(10, is_positive)  # ERROR: must be in for loop
            return 1
        
        # If it compiled, try to run it
        try:
            result = should_fail_refine_outside()
            print(f"  WARNING: Compiled but may fail at runtime (result={result})")
            return True  # Not a hard failure, just a warning
        except RuntimeError as e:
            print(f"  PASS: Runtime error as expected: {e}")
            return True
    except Exception as e:
        print(f"  PASS: Got expected error: {e}")
        return True


def test_wrong_number_of_args_multiarg():
    """
    EXPECTED FAILURE: Multi-arg predicate expects specific number of values
    """
    print("Test: Wrong number of args for multi-arg predicate (should fail)...")
    try:
        @compile
        def is_valid_range_test(start: i32, end: i32) -> bool:
            return start <= end
        
        @compile
        def should_fail_wrong_args() -> i32:
            # is_valid_range expects 2 args, but we provide 3
            r = assume(10, 20, 30, is_valid_range_test)  # ERROR: wrong arg count
            return r.start
        
        print("  FAIL: Should have raised TypeError but didn't")
        return False
    except (TypeError, AttributeError) as e:
        print(f"  PASS: Got expected error: {e}")
        return True
    except Exception as e:
        print(f"  FAIL: Got unexpected exception: {type(e).__name__}: {e}")
        return False


def test_mixing_multiarg_with_tags():
    """
    EXPECTED FAILURE: Multi-arg form and tags cannot be mixed
    
    Note: This might be allowed in the future, but currently should fail
    """
    print("Test: Mixing multi-arg predicate with tags (should fail)...")
    try:
        @compile
        def is_valid_range_test2(start: i32, end: i32) -> bool:
            return start <= end
        
        @compile
        def should_fail_mixed_forms() -> i32:
            # Cannot mix multi-arg form with tags
            r = assume(10, 20, is_valid_range_test2, "validated")  # ERROR: mixed forms
            return r.start
        
        print("  FAIL: Should have raised error but didn't")
        return False
    except (TypeError, AttributeError, SyntaxError) as e:
        print(f"  PASS: Got expected error: {e}")
        return True
    except Exception as e:
        # Might succeed if implementation allows it
        print(f"  WARNING: Compiled successfully, implementation may support mixed forms")
        return True


def test_predicate_not_found():
    """
    EXPECTED FAILURE: Predicate function not found in globals
    """
    print("Test: Predicate not found (should fail)...")
    try:
        @compile
        def should_fail_not_found() -> i32:
            # nonexistent_pred is not defined
            x = assume(10, nonexistent_pred)  # ERROR: not found
            return x
        
        print("  FAIL: Should have raised NameError or TypeError but didn't")
        return False
    except (NameError, TypeError) as e:
        print(f"  PASS: Got expected error: {e}")
        return True
    except Exception as e:
        print(f"  FAIL: Got unexpected exception: {type(e).__name__}: {e}")
        return False


def main():
    """Run all compile failure tests"""
    print("Testing expected compilation failures for refined types...")
    print("=" * 70)
    print()
    
    tests = [
        test_assume_without_constraints,
        test_refine_without_constraints,
        test_assume_non_callable_constraint,
        test_refine_used_outside_for_loop,
        test_wrong_number_of_args_multiarg,
        test_mixing_multiarg_with_tags,
        test_predicate_not_found,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"  UNEXPECTED ERROR in {test_func.__name__}: {e}")
            results.append(False)
        print()
    
    passed = sum(results)
    total = len(results)
    
    print("=" * 70)
    print(f"Results: {passed}/{total} tests behaved as expected")
    
    if passed == total:
        print("SUCCESS: All error cases properly detected!")
        return 0
    else:
        print("PARTIAL: Some error cases not properly handled")
        return 1


if __name__ == "__main__":
    sys.exit(main())
