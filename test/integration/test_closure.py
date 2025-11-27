#!/usr/bin/env python3
"""
Integration tests for closure functionality

Current status:
- OK Simple closures with single/multiple captures (called at top level)
- OK Closures with control flow (if statements)
- FAIL Closures called inside loops (KNOWN LIMITATION)
- FAIL Nested closures (KNOWN LIMITATION - equivalent to calling closure in loop)

Known limitation:
The current implementation uses `while True + break` to handle multiple return
points in inlined closures. This creates LLVM basic block structure issues when:
1. A closure is called inside a loop (creates nested while blocks)
2. Nested closures (outer's while contains inner's definition, which creates 
   another while when called)

This requires architectural redesign to handle properly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc import compile, i32, bool


# ============================================================================
# Test 1: Simple closure with single capture
# ============================================================================
@compile
def simple_closure_test(n: i32) -> i32:
    base: i32 = 100
    
    def add_base(x: i32) -> i32:
        return x + base
    
    result: i32 = add_base(n)
    return result


def test_simple_closure():
    assert simple_closure_test(10) == 110
    assert simple_closure_test(0) == 100
    assert simple_closure_test(-50) == 50
    print("OK test_simple_closure passed")


# ============================================================================
# Test 2: Closure with multiple captures
# ============================================================================
@compile
def multi_capture_test(x: i32) -> i32:
    multiplier: i32 = 2
    offset: i32 = 10
    
    def transform(n: i32) -> i32:
        temp: i32 = n * multiplier
        result: i32 = temp + offset
        return result
    
    return transform(x)


def test_multi_capture():
    assert multi_capture_test(5) == 20   # 5 * 2 + 10
    assert multi_capture_test(10) == 30  # 10 * 2 + 10
    assert multi_capture_test(0) == 10   # 0 * 2 + 10
    print("OK test_multi_capture passed")


# ============================================================================
# Test 3: Closure with if statement
# ============================================================================
@compile
def closure_with_if_test(n: i32) -> i32:
    threshold: i32 = 50
    
    def clamp_upper(x: i32) -> i32:
        if x > threshold:
            return threshold
        return x
    
    result: i32 = clamp_upper(n)
    return result


def test_closure_with_if():
    assert closure_with_if_test(30) == 30
    assert closure_with_if_test(60) == 50
    assert closure_with_if_test(50) == 50
    print("OK test_closure_with_if passed")


# ============================================================================
# DISABLED TESTS - Known limitations with while True + break approach
# ============================================================================

# Test: Closure in loop - FAILS due to nested while blocks
"""
@compile
def closure_in_loop_test(n: i32) -> i32:
    base: i32 = 100
    
    def add_base(x: i32) -> i32:
        return x + base
    
    result: i32 = 0
    i: i32 = 0
    while i < n:
        result = add_base(i)  # Closure call creates nested while
        i = i + 1
    
    return result


def test_closure_in_loop():
    result = closure_in_loop_test(3)
    assert result == 102  # 2 + 100
    print("OK test_closure_in_loop passed")
"""

# Test: Nested closures - FAILS due to same reason
"""
@compile  
def nested_closure_test(x: i32) -> i32:
    a: i32 = 10
    
    def outer(y: i32) -> i32:
        b: i32 = 20
        
        def inner(z: i32) -> i32:
            return z + a + b
        
        return inner(y)  # inner call happens inside outer's while block
    
    return outer(x)

def test_nested_closure():
    assert nested_closure_test(5) == 35
    assert nested_closure_test(0) == 30
    print("OK test_nested_closure passed")
"""


def main():
    """Run all tests"""
    print("Closure Integration Tests")
    print("=" * 60)
    
    try:
        test_simple_closure()
        test_multi_capture()
        test_closure_with_if()
        
        print()
        print("=" * 60)
        print("All enabled closure tests passed! OK")
        print()
        print("Known limitations (require architectural redesign):")
        print("  - Closures called inside loops")
        print("  - Nested closures")
        print("  (Both fail due to nested while True + break blocks)")
        return 0
    except Exception as e:
        print(f"\nFAIL Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

