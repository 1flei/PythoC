"""
Test constant for loop unrolling

This tests the compile-time loop unrolling feature for constant iterables.
"""

from pythoc import compile, i32

@compile
def test_constant_list() -> i32:
    """Test for loop with constant list - should unroll at compile time"""
    sum: i32 = 0
    for i in [1, 2, 3, 4, 5]:
        sum = sum + i
    return sum

@compile
def test_constant_tuple() -> i32:
    """Test for loop with constant tuple"""
    sum: i32 = 0
    for i in (10, 20, 30):
        sum = sum + i
    return sum

@compile
def test_constant_empty() -> i32:
    """Test for loop with empty constant list"""
    sum: i32 = 0
    for i in []:
        sum = sum + 1
    return sum

@compile
def test_constant_single() -> i32:
    """Test for loop with single element"""
    result: i32 = 0
    for i in [42]:
        result = i
    return result

@compile
def test_constant_nested() -> i32:
    """Test nested constant for loops"""
    sum: i32 = 0
    for i in [1, 2, 3]:
        for j in [10, 20]:
            sum = sum + i + j
    return sum

@compile
def test_constant_with_computation() -> i32:
    """Test constant loop with computation inside body"""
    result: i32 = 0
    for i in [1, 2, 3, 4]:
        result = result + i * i
    return result

@compile
def test_constant_overwrite() -> i32:
    """Test that loop variable is properly overwritten each iteration"""
    last: i32 = 0
    for i in [5, 10, 15, 20]:
        last = i
    return last

@compile
def test_constant_mixed_with_range() -> i32:
    """Test mixing constant loop with range loop"""
    sum: i32 = 0
    for i in [1, 2]:
        for j in range(3):
            sum = sum + i * 10 + j
    return sum

@compile
def test_constant_loop_with_return() -> i32:
    """Test mixing constant loop with range loop"""
    sum: i32 = 0
    for i in range(10):
        for j in range(20):
            if sum > 1000:
                return sum
            sum = sum + i * 10 + j
    return sum

@compile
def test_constant_loop_with_break() -> i32:
    """Test mixing constant loop with range loop"""
    sum: i32 = 0
    for i in range(10):
        for j in range(20):
            if sum > 1000:
                break
            sum = sum + i * 10 + j
    return sum

@compile
def test_constant_loop_with_continue() -> i32:
    """Test mixing constant loop with range loop"""
    sum: i32 = 0
    for i in range(10):
        for j in range(20):
            if sum > 1000:
                continue
            sum = sum + i * 10 + j
    return sum

if __name__ == '__main__':
    print("\n=== Testing Constant Loop Unrolling ===\n")
    
    print("test_constant_list():", test_constant_list())
    print("Expected: 15 (1+2+3+4+5)")
    
    print("\ntest_constant_tuple():", test_constant_tuple())
    print("Expected: 60 (10+20+30)")
    
    print("\ntest_constant_empty():", test_constant_empty())
    print("Expected: 0 (no iterations)")
    
    print("\ntest_constant_single():", test_constant_single())
    print("Expected: 42")
    
    print("\ntest_constant_nested():", test_constant_nested())
    print("Expected: 99 (i=1: 11+21=32; i=2: 12+22=34; i=3: 13+23=36; total=102)")
    print("Actual calculation: i=1,j=10: 11, i=1,j=20: 21, i=2,j=10: 12, i=2,j=20: 22, i=3,j=10: 13, i=3,j=20: 23")
    print("Sum: 11+21+12+22+13+23 = 102")
    
    print("\ntest_constant_with_computation():", test_constant_with_computation())
    print("Expected: 30 (1*1 + 2*2 + 3*3 + 4*4 = 1+4+9+16)")
    
    print("\ntest_constant_overwrite():", test_constant_overwrite())
    print("Expected: 20 (last value in list)")
    
    print("\ntest_constant_mixed_with_range():", test_constant_mixed_with_range())
    print("Expected: 66")
    print("Calculation: i=1: (10+11+12)=33; i=2: (20+21+22)=63; but wait...")
    print("i=1,j=0: 10, i=1,j=1: 11, i=1,j=2: 12")
    print("i=2,j=0: 20, i=2,j=1: 21, i=2,j=2: 22")
    print("Sum: 10+11+12+20+21+22 = 96")

    print(test_constant_loop_with_break())
    print(test_constant_loop_with_continue())
    print(test_constant_loop_with_return())
    
    print("\n=== All constant loop tests completed ===\n")
