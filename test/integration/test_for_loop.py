"""
Test for-in loop with range iterator

This demonstrates the new iterator protocol for for-in loops.
"""

from pythoc import compile, i32, seq

@compile
def test_range_basic() -> i32:
    """Test basic range iteration"""
    sum: i32 = 0
    for i in seq(10):
        sum = sum + i
    return sum

@compile
def test_range_with_start() -> i32:
    """Test range with start and stop"""
    sum: i32 = 0
    for i in seq(5, 15):
        sum = sum + i
    return sum

@compile
def test_range_with_step() -> i32:
    """Test range with start, stop, and step"""
    sum: i32 = 0
    for i in seq(0, 20, 2):
        sum = sum + i
    return sum

@compile
def test_range_negative_step() -> i32:
    """Test range with negative step"""
    sum: i32 = 0
    for i in seq(10, 0, -1):
        sum = sum + i
    return sum

@compile
def test_range_nested() -> i32:
    """Test nested range loops"""
    sum: i32 = 0
    for i in seq(5):
        for j in seq(3):
            sum = sum + i * 10 + j
    return sum

@compile
def test_range_with_break() -> i32:
    """Test range with early exit"""
    sum: i32 = 0
    for i in seq(100):
        sum = sum + i
        if sum > 50:
            break
    return sum

@compile
def test_range_empty() -> i32:
    """Test empty range"""
    count: i32 = 0
    for i in seq(0):
        count = count + 1
    return count

@compile
def test_range_single() -> i32:
    """Test single iteration range"""
    result: i32 = 0
    for i in seq(1):
        result = i
    return result

@compile
def test_lowercase_range() -> i32:
    """Test lowercase range (Python-style)"""
    sum: i32 = 0
    for i in seq(5):
        sum = sum + i
    return sum

@compile
def test_mixed_case() -> i32:
    """Test mixing range and range"""
    sum: i32 = 0
    for i in seq(3):
        for j in seq(2):
            sum = sum + i + j
    return sum

@compile
def test_continue() -> i32:
    """Test continue statement"""
    sum: i32 = 0
    for i in seq(10):
        if i % 2 == 0:
            continue
        sum = sum + i
    return sum

@compile
def test_break_in_nested() -> i32:
    """Test break in nested loop"""
    sum: i32 = 0
    for i in seq(5):
        for j in seq(5):
            if j == 3:
                break
            sum = sum + i * 10 + j
    return sum

@compile
def test_continue_in_nested() -> i32:
    """Test continue in nested loop"""
    sum: i32 = 0
    for i in seq(5):
        for j in seq(5):
            if j == 2:
                continue
            sum = sum + i * 10 + j
    return sum

@compile
def test_while_with_break() -> i32:
    """Test while loop with break"""
    sum: i32 = 0
    i: i32 = 0
    while i < 100:
        sum = sum + i
        i = i + 1
        if sum > 50:
            break
    return sum

@compile
def test_while_with_continue() -> i32:
    """Test while loop with continue"""
    sum: i32 = 0
    i: i32 = 0
    while i < 10:
        i = i + 1
        if i % 2 == 0:
            continue
        sum = sum + i
    return sum

if __name__ == '__main__':
    print("\n=== Testing Range Iterator ===\n")
    
    print("test_range_basic():", test_range_basic())
    print("Expected: 45 (sum of 0..9)")
    
    print("\ntest_range_with_start():", test_range_with_start())
    print("Expected: 95 (sum of 5..14)")
    
    print("\ntest_range_with_step():", test_range_with_step())
    print("Expected: 90 (sum of 0,2,4,6,8,10,12,14,16,18)")
    
    print("\ntest_range_negative_step():", test_range_negative_step())
    print("Expected: 55 (sum of 10,9,8,7,6,5,4,3,2,1)")
    
    print("\ntest_range_nested():", test_range_nested())
    print("Expected: 315 (nested sum: 0+1+2 + 10+11+12 + 20+21+22 + 30+31+32 + 40+41+42)")
    
    print("\ntest_range_with_break():", test_range_with_break())
    print("Expected: 55 (0+1+...+10 = 55, stops when sum > 50)")
    
    print("\ntest_range_empty():", test_range_empty())
    print("Expected: 0 (no iterations)")
    
    print("\ntest_range_single():", test_range_single())
    print("Expected: 0 (single iteration with i=0)")
    
    print("\ntest_lowercase_range():", test_lowercase_range())
    print("Expected: 10 (sum of 0+1+2+3+4)")
    
    print("\ntest_mixed_case():", test_mixed_case())
    print("Expected: 9 (i=0: j=0,1 -> 0+1=1; i=1: j=0,1 -> 1+2=3; i=2: j=0,1 -> 2+3=5; total=9)")
    
    print("\n=== Testing Break and Continue ===\n")
    
    print("test_continue():", test_continue())
    print("Expected: 25 (sum of odd numbers: 1+3+5+7+9)")
    
    print("\ntest_break_in_nested():", test_break_in_nested())
    print("Expected: 315 (break only exits inner loop; each i: j=0,1,2)")
    
    print("\ntest_continue_in_nested():", test_continue_in_nested())
    print("Expected: 440 (each i: j=0,1,3,4; skip j=2)")
    
    print("\ntest_while_with_break():", test_while_with_break())
    print("Expected: 55 (0+1+...+10, stops when sum > 50)")
    
    print("\ntest_while_with_continue():", test_while_with_continue())
    print("Expected: 25 (sum of odd numbers: 1+3+5+7+9)")
    
    print("\n=== All tests completed ===\n")
