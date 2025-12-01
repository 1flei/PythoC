#!/usr/bin/env python3
"""
Test yield functions calling @compile functions from other modules

This test reproduces the cross-module issue where yield functions 
cannot find functions imported from other modules.
"""

from pythoc import compile, i32
from test.integration.test_yield_cross_module_helper import external_add


@compile
def generator_with_external_call(n: i32) -> i32:
    """Yield generator that calls function from another module"""
    i: i32 = 0
    while i < n:
        result: i32 = external_add(i, 10)
        yield result
        i = i + 1


@compile
def test_yield_with_external_call() -> i32:
    """Test using yield generator that calls external function"""
    sum: i32 = 0
    for val in generator_with_external_call(3):
        sum = sum + val
    return sum


def main():
    print("Testing yield functions with cross-module function calls...")
    print()
    
    print("Test: Yield calling external_add() from another module")
    try:
        result = test_yield_with_external_call()
        expected = 10 + 11 + 12  # 33
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
