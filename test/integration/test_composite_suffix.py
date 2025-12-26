from pythoc import *
from pythoc.libc.stdio import printf

"""
Test suffix parameter for all composite types (struct/union/enum)

This test verifies:
1. Static declaration with suffix - @struct(suffix=T) class Point_i32: ...
2. Dynamic creation with suffix - def make_point(T): @struct(suffix=T) class Point: ...

Expected behavior:
- Different suffix values should generate different type names in LLVM IR
- Generates clean type names: Point_i32, Point_f64, Value_i64, etc.
- File names also use clean suffixes: test_composite_suffix.compile.i32.ll

The test covers all three composite types: struct, union, enum
"""

# Test 1: Static struct with suffix
@struct(suffix=i32)
class Point_i32:
    x: i32
    y: i32

@struct(suffix=f64)
class Point_f64:
    x: f64
    y: f64

@compile
def test_static_struct_suffix() -> i32:
    p1: Point_i32 = Point_i32()
    p1.x = 10
    p1.y = 20
    printf("Point_i32: x=%d, y=%d\n", p1.x, p1.y)
    
    p2: Point_f64 = Point_f64()
    p2.x = 1.5
    p2.y = 2.5
    printf("Point_f64: x=%f, y=%f\n", p2.x, p2.y)
    return 0

# Test 2: Dynamic struct with suffix - parameterized type creation
def make_point_add(T):
    @struct(suffix=T)
    class Point:
        x: T
        y: T
    
    @compile(suffix=T)
    def add_points(p1: Point, p2: Point) -> Point:
        result: Point = Point()
        result.x = p1.x + p2.x
        result.y = p1.y + p2.y
        return result
    
    return Point, add_points

Point_i32_dyn, add_points_i32 = make_point_add(i32)
Point_f64_dyn, add_points_f64 = make_point_add(f64)

@compile
def test_dynamic_struct_suffix() -> i32:
    p1: Point_i32_dyn = Point_i32_dyn()
    p1.x = 10
    p1.y = 20
    
    p2: Point_i32_dyn = Point_i32_dyn()
    p2.x = 5
    p2.y = 15
    
    result: Point_i32_dyn = add_points_i32(p1, p2)
    printf("Dynamic Point_i32: (%d, %d)\n", result.x, result.y)
    
    p3: Point_f64_dyn = Point_f64_dyn()
    p3.x = 1.5
    p3.y = 2.5
    
    p4: Point_f64_dyn = Point_f64_dyn()
    p4.x = 0.5
    p4.y = 0.5
    
    result2: Point_f64_dyn = add_points_f64(p3, p4)
    printf("Dynamic Point_f64: (%f, %f)\n", result2.x, result2.y)
    return 0

# Test 3: Static union with suffix
@union(suffix=i32)
class Value_i32:
    int_val: i32
    float_val: f32

@union(suffix=i64)
class Value_i64:
    int_val: i32
    long_val: i64

@compile
def test_static_union_suffix() -> i32:
    v1: Value_i32 = Value_i32()
    v1.int_val = 42
    printf("Value_i32: int_val=%d\n", v1.int_val)
    
    v2: Value_i64 = Value_i64()
    v2.long_val = 9876543210
    printf("Value_i64: long_val=%ld\n", v2.long_val)
    return 0

# Test 4: Dynamic union with suffix
def make_value_get(T):
    @union(suffix=T)
    class Value:
        int_val: i32
        type_val: T
    
    @compile(suffix=T)
    def get_type_val(v: Value) -> T:
        return v.type_val
    
    return Value, get_type_val

Value_i64_dyn, get_i64 = make_value_get(i64)
Value_f32_dyn, get_f32 = make_value_get(f32)

@compile
def test_dynamic_union_suffix() -> i32:
    v1: Value_i64_dyn = Value_i64_dyn()
    v1.type_val = 9876543210
    result1: i64 = get_i64(v1)
    printf("Dynamic Value_i64: %ld\n", result1)
    
    v2: Value_f32_dyn = Value_f32_dyn()
    v2.type_val = 3.14
    result2: f32 = get_f32(v2)
    printf("Dynamic Value_f32: %f\n", result2)
    
    return 0

# Test 5: Static enum with suffix
@enum(i32, suffix=i32)
class Result_i32:
    Ok: i32
    Error: i32

@enum(i64, suffix=i64)
class Result_i64:
    Ok: i64
    Error: i64

@compile
def test_static_enum_suffix() -> i32:
    r1: Result_i32 = Result_i32(Result_i32.Ok, 100)
    r2: Result_i64 = Result_i64(Result_i64.Ok, 9999999999)
    printf("Static Result_i32 and Result_i64 created\n")
    return 0

# Test 6: Dynamic enum with suffix
def make_result(T):
    @enum(T, suffix=T)
    class Result:
        Ok: T
        Error: None
    
    @compile(suffix=T)
    def create_ok(value: T) -> Result:
        r: Result = Result(Result.Ok, value)
        return r
    
    return Result, create_ok

Result_i32_dyn, create_ok_i32 = make_result(i32)
Result_f64_dyn, create_ok_f64 = make_result(f64)

@compile
def test_dynamic_enum_suffix() -> i32:
    r1: Result_i32_dyn = create_ok_i32(42)
    r2: Result_f64_dyn = create_ok_f64(3.14)
    printf("Dynamic Result_i32 and Result_f64 created with suffix\n")
    return 0

@compile
def main() -> i32:
    test_static_struct_suffix()
    test_dynamic_struct_suffix()
    test_static_union_suffix()
    test_dynamic_union_suffix()
    test_static_enum_suffix()
    test_dynamic_enum_suffix()
    return 0

import unittest


class TestCompositeSuffix(unittest.TestCase):
    """Test suffix parameter for all composite types"""

    def test_main(self):
        """Run main test function"""
        result = main()
        self.assertEqual(result, 0)


if __name__ == '__main__':
    unittest.main()
