#!/usr/bin/env python3
"""
Test importing and using yield functions from other modules

This test should reproduce the cross-module yield function issue.
"""

from pythoc import compile, i32
from test.integration.test_yield_import_helper import generator_with_builtin


@compile
def test_imported_yield_with_builtin() -> i32:
    """Test using imported yield generator that calls builtin"""
    sum: i32 = 0
    for val in generator_with_builtin(5):
        sum = sum + val
    return sum


def main():
    print("Testing imported yield functions...")
    print()
    
    print("Test: Using imported yield function with assume()")
    try:
        result = test_imported_yield_with_builtin()
        expected = 0 + 1 + 2 + 3 + 4  # 10
        if result == expected:
            print(f"  PASS: result={result}, expected={expected}")
            return 0
        else:
            print(f"  FAIL: result={result}, expected={expected}")
            return 1
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
