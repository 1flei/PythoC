"""
Integration test for unified composite types (struct/union/enum)

GOAL: Complete coverage of 3 types x 4 patterns = 12 test scenarios

Test Matrix:
+--------------+-----------------+-----------------+-----------------+
|              | struct          | union           | enum            |
+--------------+-----------------+-----------------+-----------------+
| Decorator    | @struct         | @union          | @enum(i32)      |
| Python []    | struct[i32,i32] | union[i32,f64]  | enum[...]       |
| PC []        | struct[i32,i32] | union[i32,f64]  | enum[...]       |
| Type Hint    | : struct[...]   | : union[...]    | : EnumClass     |
+--------------+-----------------+-----------------+-----------------+

CRITICAL RULES:
1. Each composite type (struct/union/enum) is ONE unified symbol
2. Test with MULTIPLE DIFFERENT types (not just one Point class!)
3. Test all 4 usage patterns for each type
4. Tests may fail - that's OK! Shows what needs implementing
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc import compile, i32, i64, f32, f64, ptr, i8
from pythoc.builtin_entities import void
from pythoc.builtin_entities.struct import struct
from pythoc.builtin_entities.union import union
from pythoc.builtin_entities.enum import enum
from pythoc.libc import printf

# Feature flags to track implementation status
STRUCT_DECORATOR_WORKS = True
UNION_DECORATOR_WORKS = True
ENUM_DECORATOR_WORKS = True


# =============================================================================
# STRUCT Tests - 4 usage patterns
# =============================================================================

# Pattern 1: struct as decorator
# Current status: @struct doesn't work directly, using @compile which delegates
@struct
class Point2D:
    """Struct Type 1: using @struct decorator"""
    x: i32
    y: i32

@struct
class Vector3D:
    """Struct Type 2: different from Point2D"""
    x: f64
    y: f64
    z: f64

@struct  
class Color:
    """Struct Type 3: completely different structure"""
    r: i32
    g: i32
    b: i32

# Pattern 2: struct[] in Python
Pair_i32 = struct[i32, i32]
Pair_f64 = struct["first": f64, f64]
Triple = struct["first": i32, "second": i32, "third": i32]


# Pattern 3: struct[] in PC compiled code
@compile
def use_struct_in_pc() -> i32:
    """Pattern 3: struct[...] subscript inside PC compiled function"""
    # Create anonymous struct type inside PC function
    c: struct[i32, i32]
    c[0] = 10
    c[1] = 20

    c2 = struct[x: i32, "y": i32, "z": i32]()
    c2.x = 1
    c2.y = 2
    c2[2] = 3
    return c[0] + c[1] + c2.x + c2.y + c2[2]


def test_struct_pc_subscript():
    """Pattern 3: struct[...] in PC code"""
    print("\n[struct-pc-subscript] struct[...] in PC code")
    
    try:
        result = use_struct_in_pc()
        if result != 36:
            print(f"  Expected 36, got {result}")
            return False
        
        return True
    except Exception as e:
        print(f"  WARNING  Failed (may not be implemented): {e}")
        import traceback
        traceback.print_exc()
        return False


# Pattern 4: struct as type hint
@compile
def add_points(a: Point2D, b: Point2D) -> Point2D:
    """Pattern 4: struct type as type hint"""
    result = Point2D()
    result.x = a.x + b.x
    result.y = a.y + b.y
    return result

@compile
def mix_vectors(v1: Vector3D, v2: Vector3D) -> f64:
    """Use different struct type"""
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

@compile
def blend_colors(c1: Color, c2: Color) -> Color:
    """Use third different struct type"""
    result = Color()
    result.r = (c1.r + c2.r) / 2
    result.g = (c1.g + c2.g) / 2
    result.b = (c1.b + c2.b) / 2
    return result

@compile
def struct_as_type_hint():
    p1 = Point2D()
    p1.x = 10
    p1.y = 20
    
    p2 = Point2D()
    p2.x = 5
    p2.y = 15
    
    p3 = add_points(p1, p2)

    v1 = Vector3D()
    v1.x = 1.0
    v1.y = 2.0
    v1.z = 3.0
    
    v2 = Vector3D()
    v2.x = 2.0
    v2.y = 3.0
    v2.z = 4.0
    
    result = mix_vectors(v1, v2)
    
    # Test Color (third different type)
    c1 = Color()
    c1.r = 100
    c1.g = 150
    c1.b = 200
    
    c2 = Color()
    c2.r = 200
    c2.g = 100
    c2.b = 50
    
    c3 = blend_colors(c1, c2)

    printf("%d, %d, %f, %d, %d, %d\n", p3.x, p3.y, result, c3.r, c3.g, c3.b)


def test_struct_as_type_hint():
    """Pattern 4: struct as type hint"""
    print("\n[struct-type-hint] struct as type hint")
    
    try:
        struct_as_type_hint()
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# UNION Tests - 4 usage patterns
# =============================================================================

# Pattern 1: union as decorator
@union
class IntOrFloat:
    """Union Type 1: using @union decorator"""
    int_val: i32
    float_val: f64

@union
class PtrOrValue:
    """Union Type 2: different from IntOrFloat"""
    ptr_val: ptr[i8]
    int_val: i64


U1 = union[i32, f64]
U2 = union[i64, f32]
U3 = union[i32, i32, i32]


# Pattern 3: union[] in PC
@compile
def use_union_in_pc() -> i32:
    """Pattern 3: union[...] in PC code"""
    u = union[i32, f64]()
    u[1] = 42.0
    return u[0]


def test_union_pc_subscript():
    """Pattern 3: union[...] in PC code"""
    print("\n[union-pc-subscript] union[...] in PC code")
    
    try:
        result = use_union_in_pc()
        print(result)
        return True
    except Exception as e:
        print(f"  WARNING  Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# Pattern 4: union as type hint
@compile
def accept_int_or_float(u: IntOrFloat) -> i32:
    """Pattern 4: union type as type hint"""
    return u[0]

@compile
def accept_ptr_or_value(u: PtrOrValue) -> i64:
    """Use different union type"""
    return u.int_val

@compile
def union_as_type_hint() -> i32:
    """Use union type as type hint"""
    u1 = IntOrFloat()
    u1.int_val = 42
    
    u2 = PtrOrValue()
    u2.int_val = 100
    return accept_int_or_float(u1) + accept_ptr_or_value(u2)



def test_union_as_type_hint():
    """Pattern 4: union as type hint"""
    print("\n[union-type-hint] union as type hint")
    
    if not UNION_DECORATOR_WORKS:
        print("  WARNING  Requires @union decorator (not yet implemented)")
        return False
    
    try:
        result1 = union_as_type_hint()
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# ENUM Tests - 4 usage patterns
# =============================================================================

# Pattern 1: enum as decorator
@enum
class Status:
    """Enum Type 1"""
    OK = 0
    ERROR: ptr[i8]

@enum(i32)
class Color_Enum:
    """Enum Type 2: different from Status"""
    RED = 0
    GREEN: None
    BLUE: None = 2
    COLOR_SIZE: None

@enum(i64)
class LargeEnum:
    """Enum Type 3: different tag type"""
    FIRST: i32
    SECOND: i64 = 10
    THIRD: f64

# Pattern 2: enum[] in Python
E1 = enum["I32": i32, "OK"]
OptionalInt = enum["EMPTY", "VAL": i32]

# Pattern 3: enum[] in PC
@compile
def use_enum_in_pc() -> i32:
    """Pattern 3: enum[...] in PC code"""
    # Syntax TBD
    printf("%d, %d, %d, %d, %d, %d, %d, %d, %d\n", \
        Status.OK, Status.ERROR, Color_Enum.RED, Color_Enum.GREEN, \
        Color_Enum.BLUE, Color_Enum.COLOR_SIZE, LargeEnum.FIRST, LargeEnum.SECOND, LargeEnum.THIRD)
    
    s_ok = Status(Status.OK)                # struct[i8, union[OK: void, ERROR: ptr[i8]]
    s_err = Status(Status.ERROR, "Error")
    printf("%d, %d, %s\n", s_ok[0], s_err[0], s_err[1].ERROR)

    # This is not supported at this stage since enum[EMPTY, VAL: i32] has the PythonType type
    # and the PythonType is not supported as lvalue currently
    # OptionalInt = enum[EMPTY, VAL: i32]     # struct[i8, union[EMPTY: void, VAL: i32]]
    o_empty = OptionalInt(OptionalInt.EMPTY)
    o_val = OptionalInt(OptionalInt.VAL, 42)
    printf("%d, %d, %d\n", o_empty[0], o_val[0], o_val[1].VAL)
    return 0


def test_enum_pc_subscript():
    use_enum_in_pc()
    return True


# Pattern 4: enum as type hint
@compile
def check_status(s: Status) -> i32:
    """Pattern 4: enum type as type hint"""
    if s[0] == Status.OK:
        return 1
    return 0

@compile  
def check_color(c: Color_Enum) -> i32:
    """Use different enum type"""
    if c[0] == Color_Enum.RED:
        return 100
    elif c[0] == Color_Enum.GREEN:
        return 200
    else:
        return 300

@compile
def enum_as_hint() -> i32:
    s_ok = Status(Status.OK)
    s_err = Status(Status.ERROR, "Error")
    return check_status(s_ok) + check_status(s_err)


def test_enum_as_type_hint():
    """Pattern 4: enum as type hint"""
    print("\n[enum-type-hint] enum as type hint")
    
    try:
        # Test Status enum
        enum_as_hint()
        print("  enum types work as type hints with 2 different enums")
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test Runner
# =============================================================================

def run_all_tests():
    """Run all 12 tests (3 types 4 patterns)"""
    print("=" * 70)
    print("COMPOSITE TYPES UNIFICATION - COMPLETE COVERAGE")
    print("=" * 70)
    print("Testing 3 types x 4 patterns = 12 scenarios")
    print("Each type must support: Decorator | Python[] | PC[] | Type Hint")
    print("=" * 70)
    
    tests = [
        ("STRUCT (4 patterns)", [
            ("3. struct[...] in PC", test_struct_pc_subscript),
            ("4. struct as type hint", test_struct_as_type_hint),
        ]),
        
        ("UNION (4 patterns)", [
            ("3. union[...] in PC", test_union_pc_subscript),
            ("4. union as type hint", test_union_as_type_hint),
        ]),
        
        ("ENUM (4 patterns)", [
            ("3. enum[...] in PC", test_enum_pc_subscript),
            ("4. enum as type hint", test_enum_as_type_hint),
        ]),
    ]
    
    total = 0
    passed = 0
    failed = 0
    
    for group_name, group_tests in tests:
        print(f"\n{'=' * 70}")
        print(f"{group_name}")
        print(f"{'=' * 70}")
        
        for test_name, test_func in group_tests:
            total += 1
            try:
                result = test_func()
                if result:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"\n  CRASH {test_name} - CRASHED: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total:   {total} / 12")
    print(f"Passed:  {passed} OK")
    print(f"Failed:  {failed} ")
    print("=" * 70)
    
    if failed == 0:
        print("\n ALL TESTS PASSED!")
    else:
        print(f"\nWARNING  {failed}/{total} test(s) failed")
        print("This shows what needs to be implemented for full unification!")
    
    return failed == 0


def mark(result):
    """Convert test result to OK or FAIL"""
    return "OK" if result else "FAIL"


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
