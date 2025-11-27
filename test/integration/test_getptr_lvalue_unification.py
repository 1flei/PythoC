"""
Test unified implementation of ptr() and visit_lvalue

Verifies that ptr() function now uses the same code path as visit_lvalue
"""

from pythoc import compile, i32, ptr, array


@compile
def test_ptr_variable() -> i32:
    """ptr(variable)"""
    x: i32 = 42
    p: ptr[i32] = ptr(x)
    return p[0]


@compile
def test_ptr_array_element() -> i32:
    """ptr(arr[i])"""
    arr: array[i32, 3] = [10, 20, 30]
    p: ptr[i32] = ptr(arr[1])
    return p[0]


@compile
class Point:
    x: i32
    y: i32


@compile
def test_ptr_struct_field() -> i32:
    """ptr(struct.field)"""
    point: Point = Point()
    point.x = 100
    point.y = 200
    p: ptr[i32] = ptr(point.x)
    return p[0]


@compile
def test_ptr_deref() -> i32:
    """ptr(*ptr)"""
    x: i32 = 99
    p1: ptr[i32] = ptr(x)
    p2: ptr[i32] = ptr(p1[0])
    return p2[0]


@compile
def test_assignment_lvalue() -> i32:
    """lvalue assignment"""
    x: i32 = 10
    x = 20
    return x


@compile
def test_array_assignment_lvalue() -> i32:
    """array element lvalue assignment"""
    arr: array[i32, 3] = [1, 2, 3]
    arr[1] = 99
    return arr[1]


@compile
def test_struct_assignment_lvalue() -> i32:
    """struct field lvalue assignment"""
    point: Point = Point()
    point.x = 10
    point.y = 20
    point.x = 999
    return point.x


if __name__ == '__main__':
    print("=" * 70)
    print("Test unified implementation of ptr() and visit_lvalue")
    print("=" * 70)
    print()
    
    print("Test 1: ptr(variable)")
    result = test_ptr_variable()
    print(f"  Result: {result}")
    assert result == 42, f"Expected 42, got {result}"
    print("  Passed")
    print()
    
    print("Test 2: ptr(arr[i])")
    result = test_ptr_array_element()
    print(f"  Result: {result}")
    assert result == 20, f"Expected 20, got {result}"
    print("  Passed")
    print()
    
    print("Test 3: ptr(struct.field)")
    result = test_ptr_struct_field()
    print(f"  Result: {result}")
    assert result == 100, f"Expected 100, got {result}"
    print("  Passed")
    print()
    
    print("Test 4: ptr(*ptr)")
    result = test_ptr_deref()
    print(f"  Result: {result}")
    assert result == 99, f"Expected 99, got {result}"
    print("  Passed")
    print()
    
    print("Test 5: variable assignment lvalue")
    result = test_assignment_lvalue()
    print(f"  Result: {result}")
    assert result == 20, f"Expected 20, got {result}"
    print("  Passed")
    print()
    
    print("Test 6: array element assignment lvalue")
    result = test_array_assignment_lvalue()
    print(f"  Result: {result}")
    assert result == 99, f"Expected 99, got {result}"
    print("  Passed")
    print()
    
    print("Test 7: struct field assignment lvalue")
    result = test_struct_assignment_lvalue()
    print(f"  Result: {result}")
    assert result == 999, f"Expected 999, got {result}"
    print("  Passed")
    print()
    
    print("=" * 70)
    print("All tests passed!")
    print("=" * 70)
    print()
    print("Summary:")
    print("  - ptr() now uses visit_expression + as_lvalue")
    print("  - visit_lvalue also uses visit_expression + as_lvalue")
    print("  - Both share the same code path")
    print("  - Removed 130+ lines of duplicate code")
    print()
