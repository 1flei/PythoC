"""
Test nested loops with returns and nested function definitions
This tests the new flag-based return transformation that supports:
1. Nested loops with multiple return points
2. Closures (nested functions with captures) that are inlined
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc import compile, i32, bool


print("\nNested Loops and Closures Tests")
print("=" * 60)


# Test 1: Simple nested loops with return (no closure, just control flow)
@compile
def test_nested_loops(n: i32) -> i32:
    i: i32 = 0
    while i < 10:
        j: i32 = 0
        while j < 10:
            if i + j == n:
                return i * 100 + j
            j = j + 1
        i = i + 1
    return -1


# Test 2: Closure with loop inside (closure gets inlined)
@compile
def test_closure_with_loop(n: i32) -> i32:
    base: i32 = 100
    
    def find_match(target: i32) -> i32:
        i: i32 = 0
        while i < 10:
            if i == target:
                return i + base  # Uses captured variable
            i = i + 1
        return -1
    
    # Closure call - this triggers inlining
    result: i32 = find_match(n)
    return result


# Test 3: Nested loops (3 levels) with returns
@compile
def test_triple_nested(n: i32) -> i32:
    i: i32 = 0
    while i < 5:
        j: i32 = 0
        while j < 5:
            k: i32 = 0
            while k < 5:
                if i + j + k == n:
                    return i * 100 + j * 10 + k
                k = k + 1
            j = j + 1
        i = i + 1
    return -1


# Test 4: Closure inside a loop (the hard case)
# This tests that inlined closure's while True wrapper doesn't conflict with outer loop
@compile
def test_closure_in_loop(n: i32) -> i32:
    base: i32 = 10
    
    def add_base(x: i32) -> i32:
        return x + base
    
    i: i32 = 0
    result: i32 = 0
    while i < 5:
        val: i32 = add_base(i)  # Closure call inside loop
        if val == n:
            result = val
            break
        i = i + 1
    return result


# Test 5: Multiple closures
@compile
def test_multiple_closures(x: i32) -> i32:
    base1: i32 = 10
    base2: i32 = 100
    
    def add1(y: i32) -> i32:
        return y + base1
    
    def add2(y: i32) -> i32:
        return y + base2
    
    r1: i32 = add1(x)
    r2: i32 = add2(x)
    return r1 + r2


# Test 6: Closure with nested loop and multiple returns
@compile
def test_closure_nested_loop(n: i32) -> i32:
    threshold: i32 = 50
    
    def search_pair(target: i32) -> i32:
        i: i32 = 0
        while i < 10:
            j: i32 = 0
            while j < 10:
                val: i32 = i * 10 + j
                if val == target:
                    if val < threshold:  # Uses captured variable
                        return val
                    return -val
                j = j + 1
            i = i + 1
        return -1
    
    result: i32 = search_pair(n)
    return result


# Test 7: Complex return pattern (multiple early exits)
@compile  
def test_complex_returns(n: i32) -> i32:
    i: i32 = 0
    while i < 10:
        while i < 5:
            if i == n:
                return i  # Early return inside inner loop
            i = i + 1
        if i == n:
            return i + 100  # Early return after inner loop
        i = i + 1
    return -1


# Run all tests
print("\nRunning tests...")
print("-" * 60)

result = test_nested_loops(5)
assert result == 5 or result == 104 or result == 203 or result == 302 or result == 401 or result == 500, f"Expected valid sum pair, got {result}"
print(f"OK test_nested_loops passed (result: {result})")

result = test_closure_with_loop(3)
assert result == 103, f"Expected 103, got {result}"
print(f"OK test_closure_with_loop passed (result: {result})")

result = test_triple_nested(6)
expected = [6, 15, 24, 33, 42, 51, 105, 114, 123, 132, 141, 204, 213, 222, 231, 303, 312, 321, 402, 411, 420]
assert result in expected, f"Expected one of {expected}, got {result}"
print(f"OK test_triple_nested passed (result: {result})")

result = test_closure_in_loop(13)
assert result == 13, f"Expected 13, got {result}"
print(f"OK test_closure_in_loop passed (result: {result})")

result = test_multiple_closures(5)
assert result == 120, f"Expected 120, got {result}"
print(f"OK test_multiple_closures passed (result: {result})")

result = test_closure_nested_loop(25)
assert result == 25, f"Expected 25, got {result}"
print(f"OK test_closure_nested_loop passed (result: {result})")

result = test_complex_returns(3)
assert result == 3, f"Expected 3, got {result}"
result2 = test_complex_returns(7)
assert result2 == 107, f"Expected 107, got {result2}"
print(f"OK test_complex_returns passed")

print("-" * 60)
print("\n" + "=" * 60)
print("All nested loops and closures tests passed! OK")
print("=" * 60)