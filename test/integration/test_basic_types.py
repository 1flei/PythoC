#!/usr/bin/env python3
"""
Basic types and type conversion tests
"""

from pythoc import i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool, ptr, compile

@compile
def test_basic_integers() -> i32:
    """Test basic integer types"""
    a: i8 = 127
    b: i16 = 32767
    c: i32 = 100
    d: i64 = 1000
    
    ua: u8 = 255
    ub: u16 = 65535
    
    result: i32 = i32(a) + i32(ua)
    
    if b > 0:
        result = result + 1
    if c > 0:
        result = result + 1
    if d > 0:
        result = result + 1
    if ub > 0:
        result = result + 1
    
    return result

@compile
def test_basic_floats() -> i32:
    """Test floating point types"""
    a: f32 = f32(3.14)
    b: f64 = 2.718
    
    result: i32 = 0
    
    threshold_f32: f32 = f32(3.0)
    if a > threshold_f32:
        result = result + 3
    
    threshold_f64: f64 = 2.0
    if b > threshold_f64:
        result = result + 2
    
    return result

@compile
def test_bool_type() -> i32:
    """Test boolean type"""
    t: bool = True
    f: bool = False
    
    result: i32 = 0
    if t:
        result = result + 1
    if not f:
        result = result + 1
    
    return result

@compile
def test_int_to_int_conversion() -> i32:
    """Test integer to integer conversions"""
    a: i8 = 100
    b: i32 = i32(a)
    c: i64 = i64(b)
    d: i16 = i16(c)
    
    return i32(d)

@compile
def test_int_to_float_conversion() -> i32:
    """Test integer to float conversions"""
    a: i32 = 42
    b: f64 = f64(a)
    c: i32 = i32(b)
    
    return c

@compile
def test_float_to_int_conversion() -> i32:
    """Test float to integer conversions"""
    a: f64 = 3.14
    b: i32 = i32(a)
    
    return b

@compile
def test_ptr_int_conversion() -> i32:
    """Test pointer to integer and back conversions"""
    x: i32 = 42
    p: ptr[i32] = ptr(x)
    addr: i64 = i64(p)
    p2: ptr[i32] = ptr[i32](addr)
    
    return p2[0]

if __name__ == "__main__":
    print(f"test_basic_integers: {test_basic_integers()}")
    print(f"test_basic_floats: {test_basic_floats()}")
    print(f"test_bool_type: {test_bool_type()}")
    print(f"test_int_to_int_conversion: {test_int_to_int_conversion()}")
    print(f"test_int_to_float_conversion: {test_int_to_float_conversion()}")
    print(f"test_float_to_int_conversion: {test_float_to_int_conversion()}")
    print(f"test_ptr_int_conversion: {test_ptr_int_conversion()}")
