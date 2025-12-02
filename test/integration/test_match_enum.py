#!/usr/bin/env python3
"""
Match/case enum pattern tests
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc import i8, i32, compile, ptr, enum
import unittest


@enum(i32)
class Status:
    Ok: i32
    Error: ptr[i8]
    Pending: None


@enum(i8)
class Color:
    Red: None
    Green: None  
    Blue: None


@compile
def make_status_ok(code: i32) -> Status:
    """Create Status.Ok variant"""
    s: Status = Status(Status.Ok, code)
    return s


@compile
def make_status_error(msg: ptr[i8]) -> Status:
    """Create Status.Error variant"""
    s: Status = Status(Status.Error, msg)
    return s


@compile
def make_status_pending() -> Status:
    """Create Status.Pending variant (no payload)"""
    s: Status = Status(Status.Pending)
    return s


@compile
def test_enum_tag_match() -> i32:
    """Match enum by tag value"""
    s: Status = make_status_ok(42)
    tag: i32 = s[0]
    match tag:
        case 0:
            return 1
        case 1:
            return 2
        case 2:
            return 3
        case _:
            return 99


@compile
def test_enum_tag_guard() -> i32:
    """Match enum tag with guard"""
    s: Status = make_status_pending()
    tag: i32 = s[0]
    match tag:
        case t if t == Status.Ok:
            return 10
        case t if t == Status.Error:
            return 20
        case t if t == Status.Pending:
            return 30
        case _:
            return 99


@compile
def test_enum_simple_tag() -> i32:
    """Match simple enum (no payload variants)"""
    c: Color = Color(Color.Red)
    tag: i8 = c[0]
    match tag:
        case 0:
            return 100
        case 1:
            return 200
        case 2:
            return 300
        case _:
            return 999


@compile
def test_enum_tag_binding() -> i32:
    """Bind and use enum tag"""
    s: Status = make_status_ok(123)
    tag: i32 = s[0]
    match tag:
        case t if t >= 0 and t <= 1:
            return t * 10
        case t:
            return t * 100


@compile
def test_enum_tuple_literal_ok() -> i32:
    """Match enum using tuple syntax: (Status.Ok, 42)"""
    s: Status = make_status_ok(42)
    match s:
        case (Status.Ok, 42):
            return 1
        case (Status.Ok, _):
            return 2
        case _:
            return 99


@compile
def test_enum_tuple_literal_pending() -> i32:
    """Match enum no-payload variant with tuple: (Status.Pending)"""
    s: Status = make_status_pending()
    match s:
        case (Status.Pending):
            return 30
        case (Status.Ok, _):
            return 10
        case _:
            return 99


@compile
def test_enum_tuple_bind_value() -> i32:
    """Bind enum payload using tuple syntax"""
    s: Status = make_status_ok(123)
    match s:
        case (Status.Ok, code):
            return code
        case (Status.Error, _):
            return -1
        case _:
            return 99


@compile
def test_enum_tuple_guard_value() -> i32:
    """Tuple syntax with guard on payload"""
    s: Status = make_status_ok(100)
    match s:
        case (Status.Ok, code) if code > 50:
            return 1
        case (Status.Ok, code) if code <= 50:
            return 2
        case _:
            return 3


@compile
def test_enum_tuple_guard_range() -> i32:
    """Tuple syntax with range guard"""
    s: Status = make_status_ok(75)
    match s:
        case (Status.Ok, code) if 1 <= code <= 50:
            return 1
        case (Status.Ok, code) if 51 <= code <= 100:
            return 2
        case (Status.Ok, code):
            return 3
        case _:
            return 99


@compile
def test_enum_tuple_multiple_variants() -> i32:
    """Match different enum variants with tuple syntax"""
    s: Status = make_status_pending()
    match s:
        case (Status.Ok, code):
            return code
        case (Status.Pending):
            return 999
        case _:
            return -1


@compile
def test_enum_simple_tuple() -> i32:
    """Match simple enum (no payload) with tuple"""
    c: Color = Color(Color.Green)
    match c:
        case (Color.Red):
            return 1
        case (Color.Green):
            return 2
        case (Color.Blue):
            return 3
        case _:
            return 99


@compile
def test_enum_tuple_extract_ok() -> i32:
    """Extract Ok payload value 777"""
    s: Status = make_status_ok(777)
    match s:
        case (Status.Ok, val):
            return val
        case _:
            return -1


@compile
def test_enum_tuple_nested_match() -> i32:
    """Nested match on enum variant and payload"""
    s: Status = make_status_ok(25)
    match s:
        case (Status.Ok, code):
            match code:
                case n if n < 10:
                    return 1
                case n if 10 <= n <= 50:
                    return 2
                case n:
                    return 3
        case _:
            return 99


@compile
def test_enum_constructor_literal() -> i32:
    """Match enum using constructor syntax"""
    s: Status = make_status_ok(42)
    match s:
        case (Status.Ok, 42):
            return 1
        case (Status.Ok, _):
            return 2
        case _:
            return 99


@compile
def test_enum_constructor_bind() -> i32:
    """Bind enum payload using constructor syntax"""
    s: Status = make_status_ok(777)
    match s:
        case (Status.Ok, code):
            return code
        case (Status.Error, _):
            return -1
        case _:
            return 99


@compile
def test_enum_constructor_guard() -> i32:
    """Constructor syntax with guard"""
    s: Status = make_status_ok(150)
    match s:
        case (Status.Ok, code) if code > 100:
            return 1
        case (Status.Ok, code):
            return 2
        case _:
            return 3


@compile
def test_enum_constructor_pending() -> i32:
    """Match no-payload variant with constructor"""
    s: Status = make_status_pending()
    match s:
        case (Status.Pending):
            return 888
        case (Status.Ok, _):
            return 111
        case _:
            return 999


@compile
def test_enum_constructor_simple() -> i32:
    """Simple enum with constructor syntax"""
    c: Color = Color(Color.Blue)
    match c:
        case (Color.Red):
            return 100
        case (Color.Green):
            return 200
        case (Color.Blue):
            return 300
        case _:
            return 999


@compile
def test_enum_constructor_tag_only() -> i32:
    """Constructor syntax matching only tag"""
    s: Status = make_status_ok(456)
    match s:
        case (Status.Ok, _):
            return 111
        case (Status.Error, _):
            return 222
        case _:
            return 999


@compile
def test_enum_result_handler() -> i32:
    """Realistic Result-like error handling"""
    s: Status = make_status_ok(42)
    match s:
        case (Status.Ok, 0):
            return 0
        case (Status.Ok, code) if code > 0:
            return code
        case (Status.Ok, code):
            return -code
        case (Status.Error, _):
            return -999
        case (Status.Pending):
            return -1
        case _:
            return -9999


@compile  
def test_enum_state_transition() -> i32:
    """State machine using enum matching"""
    s: Status = make_status_pending()
    match s:
        case (Status.Pending):
            return 1
        case (Status.Ok, code) if code == 0:
            return 2
        case (Status.Ok, code):
            return 3
        case _:
            return 99


@compile
def test_enum_all_wildcards() -> i32:
    """Match enum with all wildcards"""
    s: Status = make_status_ok(123)
    match s:
        case (_, _):
            return 1
        case _:
            return 99


@compile
def test_enum_compare_variants() -> i32:
    """Compare different Status variants"""
    s1: Status = make_status_ok(10)
    s2: Status = make_status_pending()
    
    tag1: i32 = s1[0]
    tag2: i32 = s2[0]
    
    match tag1:
        case t if t == tag2:
            return 0
        case t if t < tag2:
            return -1
        case t:
            return 1


@compile
def test_enum_extract_and_compute() -> i32:
    """Extract payload and perform computation"""
    s: Status = make_status_ok(20)
    match s:
        case (Status.Ok, code):
            result: i32 = code * 2 + 5
            return result
        case _:
            return -1


@compile
def test_enum_multiple_conditions() -> i32:
    """Multiple guard conditions on payload"""
    s: Status = make_status_ok(45)
    match s:
        case (Status.Ok, code) if code < 0:
            return 1
        case (Status.Ok, code) if code == 0:
            return 2
        case (Status.Ok, code) if 1 <= code <= 50:
            return 3
        case (Status.Ok, code) if 51 <= code <= 100:
            return 4
        case (Status.Ok, code):
            return 5
        case _:
            return 99


class TestMatchEnumPatterns(unittest.TestCase):
    """Test enum pattern matching (legacy tag-based)"""
    
    def test_enum_tag_match(self):
        self.assertEqual(test_enum_tag_match(), 1)
    
    def test_enum_tag_guard(self):
        self.assertEqual(test_enum_tag_guard(), 30)
    
    def test_enum_simple_tag(self):
        self.assertEqual(test_enum_simple_tag(), 100)
    
    def test_enum_tag_binding(self):
        self.assertEqual(test_enum_tag_binding(), 0)


class TestMatchEnumTupleSyntax(unittest.TestCase):
    """Test enum matching with tuple syntax"""
    
    def test_tuple_literal_ok(self):
        self.assertEqual(test_enum_tuple_literal_ok(), 1)
    
    def test_tuple_literal_pending(self):
        self.assertEqual(test_enum_tuple_literal_pending(), 30)
    
    def test_tuple_bind_value(self):
        self.assertEqual(test_enum_tuple_bind_value(), 123)
    
    def test_tuple_guard_value(self):
        self.assertEqual(test_enum_tuple_guard_value(), 1)
    
    def test_tuple_guard_range(self):
        self.assertEqual(test_enum_tuple_guard_range(), 2)
    
    def test_tuple_multiple_variants(self):
        self.assertEqual(test_enum_tuple_multiple_variants(), 999)
    
    def test_simple_tuple(self):
        self.assertEqual(test_enum_simple_tuple(), 2)
    
    def test_tuple_extract_ok(self):
        self.assertEqual(test_enum_tuple_extract_ok(), 777)
    
    def test_tuple_nested_match(self):
        self.assertEqual(test_enum_tuple_nested_match(), 2)


class TestMatchEnumConstructorSyntax(unittest.TestCase):
    """Test enum matching with constructor syntax"""
    
    def test_constructor_literal(self):
        self.assertEqual(test_enum_constructor_literal(), 1)
    
    def test_constructor_bind(self):
        self.assertEqual(test_enum_constructor_bind(), 777)
    
    def test_constructor_guard(self):
        self.assertEqual(test_enum_constructor_guard(), 1)
    
    def test_constructor_pending(self):
        self.assertEqual(test_enum_constructor_pending(), 888)
    
    def test_constructor_simple(self):
        self.assertEqual(test_enum_constructor_simple(), 300)
    
    def test_constructor_tag_only(self):
        self.assertEqual(test_enum_constructor_tag_only(), 111)


class TestMatchEnumComplexPatterns(unittest.TestCase):
    """Test complex enum matching patterns"""
    
    def test_result_handler(self):
        self.assertEqual(test_enum_result_handler(), 42)
    
    def test_state_transition(self):
        self.assertEqual(test_enum_state_transition(), 1)
    
    def test_all_wildcards(self):
        self.assertEqual(test_enum_all_wildcards(), 1)
    
    def test_compare_variants(self):
        self.assertEqual(test_enum_compare_variants(), -1)
    
    def test_extract_and_compute(self):
        self.assertEqual(test_enum_extract_and_compute(), 45)
    
    def test_multiple_conditions(self):
        self.assertEqual(test_enum_multiple_conditions(), 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
