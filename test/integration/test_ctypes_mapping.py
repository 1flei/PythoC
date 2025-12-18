"""
Test ctypes mapping for pythoc types.

Verifies that get_ctypes_type() returns correct ctypes types for FFI,
especially for signed/unsigned distinction.
"""

from pythoc import (
    compile, struct,
    i8, i16, i32, i64,
    u8, u16, u32, u64,
    f32, f64,
    bool as pc_bool,
    ptr, array, func, void
)
from pythoc.libc.stdio import printf


# =============================================================================
# Test unsigned return values (should not be interpreted as signed)
# =============================================================================

@compile
def test_u32_max() -> u32:
    """u32 max value should be 0xFFFFFFFF, not -1"""
    return u32(0xFFFFFFFF)

@compile
def test_u8_max() -> u8:
    """u8 max value should be 255, not -1"""
    return u8(255)

@compile
def test_u16_max() -> u16:
    """u16 max value should be 65535, not -1"""
    return u16(65535)

@compile
def test_u64_large() -> u64:
    """u64 large value should be positive"""
    return u64(0x8000000000000000)


# =============================================================================
# Test unsigned shift operations
# =============================================================================

@compile
def test_u32_rshift() -> u32:
    """Unsigned right shift should use logical shift (fill with zeros)"""
    x: u32 = 0x80000000
    # Logical shift: 0x80000000 >> 2 = 0x20000000
    return x >> 2

@compile
def test_i32_rshift() -> i32:
    """Signed right shift should use arithmetic shift (preserve sign)"""
    x: i32 = i32(-2147483648)  # 0x80000000 as signed
    # Arithmetic shift: preserves sign bit
    return x >> 2


# =============================================================================
# Test bool conversion
# =============================================================================

@compile
def test_bool_from_nonzero() -> pc_bool:
    """bool(42) should be True (not False from truncation)"""
    x: i32 = 42
    return pc_bool(x)

@compile
def test_bool_from_zero() -> pc_bool:
    """bool(0) should be False"""
    x: i32 = 0
    return pc_bool(x)

@compile
def test_bool_from_one() -> pc_bool:
    """bool(1) should be True"""
    x: i32 = 1
    return pc_bool(x)


# =============================================================================
# Test struct return values
# =============================================================================

@struct
class Point:
    x: i32
    y: i32

@compile
def test_struct_return() -> Point:
    """Struct return should work correctly"""
    p: Point = Point()
    p.x = 123
    p.y = 456
    return p


# =============================================================================
# Test mixed signed/unsigned operations
# =============================================================================

@compile
def test_u32_left_shift() -> u32:
    """Left shift to high bit"""
    x: u32 = 1
    return x << 31  # Should be 0x80000000 = 2147483648

@compile
def test_u8_wrap() -> u8:
    """u8 should wrap correctly"""
    x: u8 = 200
    y: u8 = 100
    return x + y  # 300 wraps to 44


# =============================================================================
# Run all tests
# =============================================================================

if __name__ == '__main__':
    print("Testing ctypes mapping...")
    
    # Unsigned return values
    result = test_u32_max()
    assert result == 0xFFFFFFFF, f"test_u32_max: expected {0xFFFFFFFF}, got {result}"
    print(f"  test_u32_max: {result} (0x{result:08X}) OK")
    
    result = test_u8_max()
    assert result == 255, f"test_u8_max: expected 255, got {result}"
    print(f"  test_u8_max: {result} OK")
    
    result = test_u16_max()
    assert result == 65535, f"test_u16_max: expected 65535, got {result}"
    print(f"  test_u16_max: {result} OK")
    
    result = test_u64_large()
    assert result == 0x8000000000000000, f"test_u64_large: expected {0x8000000000000000}, got {result}"
    print(f"  test_u64_large: {result} (0x{result:016X}) OK")
    
    # Unsigned shift
    result = test_u32_rshift()
    assert result == 0x20000000, f"test_u32_rshift: expected {0x20000000}, got {result}"
    print(f"  test_u32_rshift: {result} (0x{result:08X}) OK")
    
    result = test_i32_rshift()
    assert result == -536870912, f"test_i32_rshift: expected -536870912, got {result}"
    print(f"  test_i32_rshift: {result} OK")
    
    # Bool conversion
    result = test_bool_from_nonzero()
    assert result == True, f"test_bool_from_nonzero: expected True, got {result}"
    print(f"  test_bool_from_nonzero: {result} OK")
    
    result = test_bool_from_zero()
    assert result == False, f"test_bool_from_zero: expected False, got {result}"
    print(f"  test_bool_from_zero: {result} OK")
    
    result = test_bool_from_one()
    assert result == True, f"test_bool_from_one: expected True, got {result}"
    print(f"  test_bool_from_one: {result} OK")
    
    # Struct return
    result = test_struct_return()
    assert result.x == 123, f"test_struct_return.x: expected 123, got {result.x}"
    assert result.y == 456, f"test_struct_return.y: expected 456, got {result.y}"
    print(f"  test_struct_return: ({result.x}, {result.y}) OK")
    
    # Left shift
    result = test_u32_left_shift()
    assert result == 2147483648, f"test_u32_left_shift: expected 2147483648, got {result}"
    print(f"  test_u32_left_shift: {result} OK")
    
    # Wrap
    result = test_u8_wrap()
    assert result == 44, f"test_u8_wrap: expected 44, got {result}"
    print(f"  test_u8_wrap: {result} OK")
    
    print("\nAll ctypes mapping tests passed!")
