#!/usr/bin/env python3
"""
Comprehensive tests for enum types including tagged unions,
pattern matching with enums, and complex enum scenarios.
"""

import unittest
from pythoc import (
    i8, i16, i32, i64,
    f32, f64, bool, ptr, array, struct, enum, compile
)


# =============================================================================
# Basic Enum Definitions
# =============================================================================

@enum
class SimpleColor:
    """Simple enum with no payload"""
    Red: None
    Green: None
    Blue: None


@enum
class ExplicitColor:
    """Enum with explicit tag values"""
    Red: None = 0
    Green: None = 5
    Blue: None = 10


@enum(i32)
class ResultI32:
    """Result enum with i32 payload"""
    Ok: i32
    Err: i32


@enum(i16)
class Option:
    """Option-like enum"""
    Some: i32
    NoneVal: None


@enum(i32)
class MixedPayload:
    """Enum with mixed payload types"""
    IntVal: i32
    FloatVal: f64
    PtrVal: ptr[i8]
    NoPayload: None


@enum(i8)
class SmallEnum:
    """Enum with small tag type"""
    A: None
    B: None
    C: None


@enum(i64)
class LargeEnum:
    """Enum with large tag type"""
    First: i64
    Second: f64
    Third: None


# =============================================================================
# Basic Enum Operations
# =============================================================================

@compile
def test_simple_enum_tags() -> i32:
    """Test simple enum tag values"""
    r: i32 = SimpleColor.Red
    g: i32 = SimpleColor.Green
    b: i32 = SimpleColor.Blue
    return r + g + b  # 0 + 1 + 2 = 3


@compile
def test_explicit_enum_tags() -> i32:
    """Test explicit enum tag values"""
    r: i32 = ExplicitColor.Red
    g: i32 = ExplicitColor.Green
    b: i32 = ExplicitColor.Blue
    return r + g + b  # 0 + 5 + 10 = 15


@compile
def test_enum_create_no_payload() -> i32:
    """Test creating enum with no payload"""
    c: SimpleColor = SimpleColor(SimpleColor.Red)
    return 1  # Just test it compiles and runs


@compile
def test_enum_create_with_payload() -> i32:
    """Test creating enum with payload"""
    ok: ResultI32 = ResultI32(ResultI32.Ok, 42)
    return 1


@compile
def test_enum_tuple_init() -> i32:
    """Test enum initialization from tuple"""
    ok: ResultI32 = (ResultI32.Ok, 100)
    return 1


# =============================================================================
# Enum Tag Comparison
# =============================================================================

@compile
def test_enum_tag_compare_equal() -> i32:
    """Test enum tag equality"""
    if SimpleColor.Red == 0:
        return 1
    return 0


@compile
def test_enum_tag_compare_not_equal() -> i32:
    """Test enum tag inequality"""
    if SimpleColor.Red != SimpleColor.Blue:
        return 1
    return 0


@compile
def test_enum_tag_in_switch() -> i32:
    """Test enum tag in switch-like pattern"""
    tag: i32 = SimpleColor.Green
    result: i32 = 0
    
    if tag == SimpleColor.Red:
        result = 10
    elif tag == SimpleColor.Green:
        result = 20
    elif tag == SimpleColor.Blue:
        result = 30
    
    return result  # 20


# =============================================================================
# Enum with Different Tag Types
# =============================================================================

@compile
def test_small_enum() -> i32:
    """Test enum with i8 tag type"""
    a: i32 = SmallEnum.A
    b: i32 = SmallEnum.B
    c: i32 = SmallEnum.C
    return a + b + c  # 0 + 1 + 2 = 3


@compile
def test_large_enum() -> i32:
    """Test enum with i64 tag type"""
    f: i32 = LargeEnum.First
    s: i32 = LargeEnum.Second
    t: i32 = LargeEnum.Third
    return f + s + t  # 0 + 1 + 2 = 3


# =============================================================================
# Enum Payload Access
# =============================================================================

@compile
def test_option_some() -> i32:
    """Test Option Some variant"""
    opt: Option = Option(Option.Some, 42)
    return 1


@compile
def test_option_none() -> i32:
    """Test Option None variant"""
    opt: Option = Option(Option.NoneVal)
    return 1


@compile
def test_mixed_payload_int() -> i32:
    """Test mixed payload enum with int"""
    val: MixedPayload = MixedPayload(MixedPayload.IntVal, 42)
    return 1


@compile
def test_mixed_payload_float() -> i32:
    """Test mixed payload enum with float"""
    val: MixedPayload = MixedPayload(MixedPayload.FloatVal, 3.14)
    return 1


@compile
def test_mixed_payload_none() -> i32:
    """Test mixed payload enum with no payload"""
    val: MixedPayload = MixedPayload(MixedPayload.NoPayload)
    return 1


# =============================================================================
# Enum in Control Flow
# =============================================================================

@compile
def test_enum_in_if() -> i32:
    """Test enum tag in if condition"""
    tag: i32 = SimpleColor.Blue
    if tag == SimpleColor.Blue:
        return 1
    return 0


@compile
def test_enum_in_loop() -> i32:
    """Test enum in loop"""
    count: i32 = 0
    i: i32 = 0
    while i < 3:
        if i == SimpleColor.Red:
            count = count + 1
        elif i == SimpleColor.Green:
            count = count + 2
        elif i == SimpleColor.Blue:
            count = count + 4
        i = i + 1
    return count  # 1 + 2 + 4 = 7


# =============================================================================
# Enum Patterns
# =============================================================================

@compile
def test_enum_as_return() -> i32:
    """Test returning enum tag"""
    return SimpleColor.Green  # 1


@compile
def test_enum_tag_arithmetic() -> i32:
    """Test arithmetic with enum tags"""
    base: i32 = ExplicitColor.Green  # 5
    return base * 2  # 10


@compile
def test_enum_tag_array_index() -> i32:
    """Test using enum tag as array index"""
    arr: array[i32, 3] = [100, 200, 300]
    idx: i32 = SimpleColor.Green  # 1
    return arr[idx]  # 200


# =============================================================================
# Complex Enum Scenarios
# =============================================================================

@enum(i32)
class TreeNode:
    """Tree-like enum for recursive structures"""
    Leaf: i32
    Branch: struct[i32, i32]


@compile
def test_tree_leaf() -> i32:
    """Test tree leaf node"""
    leaf: TreeNode = TreeNode(TreeNode.Leaf, 42)
    return 1


@compile
def test_tree_branch() -> i32:
    """Test tree branch node"""
    branch: TreeNode = TreeNode(TreeNode.Branch, (10, 20))
    return 1


@enum(i32)
class Expression:
    """Expression-like enum"""
    Const: i32
    Add: struct[i32, i32]
    Mul: struct[i32, i32]


@compile
def test_expr_const() -> i32:
    """Test expression constant"""
    e: Expression = Expression(Expression.Const, 42)
    return 1


@compile
def test_expr_add() -> i32:
    """Test expression add"""
    e: Expression = Expression(Expression.Add, (10, 20))
    return 1


@compile
def test_expr_mul() -> i32:
    """Test expression multiply"""
    e: Expression = Expression(Expression.Mul, (5, 6))
    return 1


# =============================================================================
# Enum Array
# =============================================================================

@compile
def test_enum_tag_array() -> i32:
    """Test array of enum tags"""
    tags: array[i32, 3] = [SimpleColor.Red, SimpleColor.Green, SimpleColor.Blue]
    return tags[0] + tags[1] + tags[2]  # 3


# =============================================================================
# Edge Cases
# =============================================================================

@compile
def test_enum_single_variant() -> i32:
    """Test enum with effectively single variant usage"""
    tag: i32 = SimpleColor.Red
    return tag  # 0


@compile
def test_enum_max_tag() -> i32:
    """Test enum with maximum explicit tag"""
    return ExplicitColor.Blue  # 10


@compile
def test_enum_tag_zero() -> i32:
    """Test enum tag is zero"""
    if SimpleColor.Red == 0:
        return 1
    return 0


# =============================================================================
# Test Runner
# =============================================================================

class TestBasicEnum(unittest.TestCase):
    def test_simple_tags(self):
        self.assertEqual(test_simple_enum_tags(), 3)
    
    def test_explicit_tags(self):
        self.assertEqual(test_explicit_enum_tags(), 15)
    
    def test_create_no_payload(self):
        self.assertEqual(test_enum_create_no_payload(), 1)
    
    def test_create_with_payload(self):
        self.assertEqual(test_enum_create_with_payload(), 1)
    
    def test_tuple_init(self):
        self.assertEqual(test_enum_tuple_init(), 1)


class TestEnumTagCompare(unittest.TestCase):
    def test_equal(self):
        self.assertEqual(test_enum_tag_compare_equal(), 1)
    
    def test_not_equal(self):
        self.assertEqual(test_enum_tag_compare_not_equal(), 1)
    
    def test_in_switch(self):
        self.assertEqual(test_enum_tag_in_switch(), 20)


class TestEnumTagTypes(unittest.TestCase):
    def test_small(self):
        self.assertEqual(test_small_enum(), 3)
    
    def test_large(self):
        self.assertEqual(test_large_enum(), 3)


class TestEnumPayload(unittest.TestCase):
    def test_option_some(self):
        self.assertEqual(test_option_some(), 1)
    
    def test_option_none(self):
        self.assertEqual(test_option_none(), 1)
    
    def test_mixed_int(self):
        self.assertEqual(test_mixed_payload_int(), 1)
    
    def test_mixed_float(self):
        self.assertEqual(test_mixed_payload_float(), 1)
    
    def test_mixed_none(self):
        self.assertEqual(test_mixed_payload_none(), 1)


class TestEnumControlFlow(unittest.TestCase):
    def test_in_if(self):
        self.assertEqual(test_enum_in_if(), 1)
    
    def test_in_loop(self):
        self.assertEqual(test_enum_in_loop(), 7)


class TestEnumPatterns(unittest.TestCase):
    def test_as_return(self):
        self.assertEqual(test_enum_as_return(), 1)
    
    def test_arithmetic(self):
        self.assertEqual(test_enum_tag_arithmetic(), 10)
    
    def test_array_index(self):
        self.assertEqual(test_enum_tag_array_index(), 200)


class TestComplexEnum(unittest.TestCase):
    def test_tree_leaf(self):
        self.assertEqual(test_tree_leaf(), 1)
    
    def test_tree_branch(self):
        self.assertEqual(test_tree_branch(), 1)
    
    def test_expr_const(self):
        self.assertEqual(test_expr_const(), 1)
    
    def test_expr_add(self):
        self.assertEqual(test_expr_add(), 1)
    
    def test_expr_mul(self):
        self.assertEqual(test_expr_mul(), 1)


class TestEnumArray(unittest.TestCase):
    def test_tag_array(self):
        self.assertEqual(test_enum_tag_array(), 3)


class TestEdgeCases(unittest.TestCase):
    def test_single_variant(self):
        self.assertEqual(test_enum_single_variant(), 0)
    
    def test_max_tag(self):
        self.assertEqual(test_enum_max_tag(), 10)
    
    def test_tag_zero(self):
        self.assertEqual(test_enum_tag_zero(), 1)


if __name__ == '__main__':
    unittest.main()
