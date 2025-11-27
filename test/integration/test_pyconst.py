"""
Test pyconst and typeof features
"""
from pythoc import *


def test_pyconst_basic():
    """Test basic pyconst type"""
    
    # Test pyconst[value] creates zero-sized type
    T = pyconst[42]
    assert hasattr(T, 'get_size_bytes')
    assert T.get_size_bytes() == 0
    print("test_pyconst_basic: PASS")


def test_typeof_basic():
    """Test basic typeof functionality"""
    
    # Test typeof with Python constants
    t1 = typeof(5)
    t2 = typeof(100)
    assert t1 is not t2  # Different values = different types
    
    # Test typeof with types
    t3 = typeof(i32)
    # t3 should be i32
    
    print("test_typeof_basic: PASS")


def test_struct_with_pyconst():
    """Test struct with pyconst fields"""
    
    # Create struct with pyconst field
    S = struct[('a', i32), ('b', pyconst[42]), ('c', i32)]
    
    # Size should exclude pyconst field (zero-sized)
    # sizeof(S) == sizeof(i32) + sizeof(i32) = 8
    expected_size = 8
    actual_size = S.get_size_bytes()
    assert actual_size == expected_size, f"Expected {expected_size}, got {actual_size}"
    
    print("test_struct_with_pyconst: PASS")


# Define struct outside @compile
TestStruct = struct[('a', i32), ('b', pyconst[42])]


@compile
def test_pyconst_field_access() -> i32:
    """Test accessing pyconst field in struct"""
    
    @compile(suffix=TestStruct)
    def get_b(x: TestStruct) -> i32:
        return i32(x.b)  # Should return constant 42
    
    s: TestStruct
    s.a = 10
    s.b = 42  # Should type-check and be no-op
    
    return get_b(s)


def test_typeof_in_struct():
    """Test using typeof in struct definition"""
    
    def Vec(T, size_spec):
        return struct[('size', typeof(size_spec)), ('data', ptr[T])]
    
    # Static vector - size is pyconst[100]
    StaticVec = Vec(i32, 100)
    # Dynamic vector - size is i32
    DynamicVec = Vec(i32, i32)
    
    # StaticVec should be smaller (no size field storage)
    static_size = StaticVec.get_size_bytes()
    dynamic_size = DynamicVec.get_size_bytes()
    
    # Static: only ptr (8 bytes on 64-bit)
    # Dynamic: i32 + ptr (12-16 bytes depending on alignment)
    assert static_size < dynamic_size, f"Static ({static_size}) should be smaller than dynamic ({dynamic_size})"
    
    print("test_typeof_in_struct: PASS")


if __name__ == "__main__":
    test_pyconst_basic()
    test_typeof_basic()
    test_struct_with_pyconst()
    test_typeof_in_struct()
    
    # Test compiled function
    result = test_pyconst_field_access()
    assert result == 42, f"Expected 42, got {result}"
    print("test_pyconst_field_access: PASS")
    
    print("\nAll pyconst tests PASSED!")
