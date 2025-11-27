"""
Test local Python type variables in type annotations

Verifies that:
1. Local Python type assignments (MyType = i32) work
2. Type annotations can reference local Python type variables (x: MyType)
3. Type subscripts can use local type variables (ptr[T])
"""

from pythoc import compile, i32, f64, ptr, sizeof
from pythoc.libc.stdlib import malloc


@compile
def test_local_type_alias() -> i32:
    """Test local type alias: MyType = i32"""
    MyType = i32
    x: MyType = 42
    return x


@compile
def test_ptr_subscript() -> i32:
    """Test ptr subscript with local type variable"""
    T = i32
    pt = ptr[T]
    p: pt = malloc(sizeof(T))
    p[0] = 99
    result: i32 = p[0]
    return result


@compile
def test_nested_scopes() -> i32:
    """Test type variables in nested scopes"""
    outer_type = i32
    x: outer_type = 10
    
    for T in [i32]:
        inner_type = ptr[T]
        p: inner_type = malloc(sizeof(T))
        p[0] = 20
        x = p[0]
    
    return x


if __name__ == '__main__':
    print("=" * 70)
    print("Test local Python type variables in type annotations")
    print("=" * 70)
    print()
    
    print("Test 1: Local type alias")
    result = test_local_type_alias()
    print(f"  Result: {result}")
    assert result == 42, f"Expected 42, got {result}"
    print("  Passed")
    print()
    
    print("Test 2: ptr subscript with local type variable")
    result = test_ptr_subscript()
    print(f"  Result: {result}")
    assert result == 99, f"Expected 99, got {result}"
    print("  Passed")
    print()
    
    print("Test 3: Nested scopes")
    result = test_nested_scopes()
    print(f"  Result: {result}")
    assert result == 20, f"Expected 20, got {result}"
    print("  Passed")
    print()
    
    print("=" * 70)
    print("All tests passed!")
    print("=" * 70)
    print()
    print("Summary:")
    print("  - Local Python type variables (MyType = i32) work")
    print("  - Type annotations can reference local variables (x: MyType)")
    print("  - Type subscripts work with local variables (ptr[T])")
    print("  - TypeResolver can access visitor's symbol table")
    print()
