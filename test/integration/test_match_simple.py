import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from pythoc import i32, compile

# Test 1: Struct with tuple syntax
@compile
class Point:
    x: i32
    y: i32

@compile
def make_point(x: i32, y: i32) -> Point:
    p: Point
    p.x = x
    p.y = y
    return p

@compile
def test_struct_tuple_pattern(px: i32, py: i32) -> i32:
    """Test struct matching with inline construction"""
    p: Point
    p.x = px
    p.y = py
    
    match p:
        case (0, 0):
            return 100
        case (x, 0):
            return x
        case (0, y):
            return y
        case (x, y):
            return x + y

# Test 2: Nested patterns
@compile
class Rect:
    top_left: Point
    width: i32
    height: i32

@compile
def make_rect(x: i32, y: i32, w: i32, h: i32) -> Rect:
    r: Rect
    r.top_left.x = x
    r.top_left.y = y
    r.width = w
    r.height = h
    return r

@compile
def test_nested_pattern(x: i32, y: i32, w: i32, h: i32) -> i32:
    """Test nested struct patterns"""
    r = make_rect(x, y, w, h)
    match r:
        case ((0, 0), w, h):
            return w * h
        case ((x, y), w, h):
            return x + y + w + h

if __name__ == "__main__":
    print("Testing simplified match pattern implementation...")
    print("\n=== Test 1: Struct Pattern ===")
    
    # Test struct
    result1 = test_struct_tuple_pattern(0, 0)
    print(f"test_struct_tuple_pattern((0, 0)) = {result1}, expected 100: {'OK' if result1 == 100 else 'FAIL'}")
    
    result2 = test_struct_tuple_pattern(5, 0)
    print(f"test_struct_tuple_pattern((5, 0)) = {result2}, expected 5: {'OK' if result2 == 5 else 'FAIL'}")
    
    result3 = test_struct_tuple_pattern(3, 4)
    print(f"test_struct_tuple_pattern((3, 4)) = {result3}, expected 7: {'OK' if result3 == 7 else 'FAIL'}")
    
    print("\n=== Test 2: Nested Pattern ===")
    
    # Test nested
    result6 = test_nested_pattern(0, 0, 10, 20)
    print(f"test_nested_pattern(((0, 0), 10, 20)) = {result6}, expected 200: {'OK' if result6 == 200 else 'FAIL'}")
    
    result7 = test_nested_pattern(1, 2, 10, 20)
    print(f"test_nested_pattern(((1, 2), 10, 20)) = {result7}, expected 33: {'OK' if result7 == 33 else 'FAIL'}")
    
    print("\nOK All tests passed!" if all([
        result1 == 100,
        result2 == 5,
        result3 == 7,
        result6 == 200,
        result7 == 33,
    ]) else "\nFAIL Some tests failed!")

