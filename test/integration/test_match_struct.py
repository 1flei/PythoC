#!/usr/bin/env python3
"""
Match/case struct destructuring tests
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc import i32, compile
from pythoc.decorators import clear_registry
import unittest


@compile
class Point:
    x: i32
    y: i32


@compile
def make_point(x: i32, y: i32) -> Point:
    """Helper to create Point instances"""
    p: Point
    p.x = x
    p.y = y
    return p


@compile
class Point3D:
    x: i32
    y: i32
    z: i32


@compile
def make_point3d(x: i32, y: i32, z: i32) -> Point3D:
    """Helper to create Point3D instances"""
    p: Point3D
    p.x = x
    p.y = y
    p.z = z
    return p


@compile
class Rect:
    top_left: Point
    width: i32
    height: i32


@compile
def make_rect(px: i32, py: i32, w: i32, h: i32) -> Rect:
    """Helper to create Rect instances"""
    p: Point = make_point(px, py)
    r: Rect
    r.top_left = p
    r.width = w
    r.height = h
    return r


@compile
def test_struct_literal_origin() -> i32:
    """Match Point(0,0)"""
    p: Point = make_point(0, 0)
    match p:
        case (0, 0):
            return 0
        case (1, 1):
            return 1
        case _:
            return 99


@compile
def test_struct_literal_unit() -> i32:
    """Match Point(1,1)"""
    p: Point = make_point(1, 1)
    match p:
        case (0, 0):
            return 0
        case (1, 1):
            return 1
        case _:
            return 99


@compile
def test_struct_literal_other() -> i32:
    """Match Point(5,7) - no match"""
    p: Point = make_point(5, 7)
    match p:
        case (0, 0):
            return 0
        case (1, 1):
            return 1
        case _:
            return 99


@compile
def test_struct_bind_origin() -> i32:
    """Bind Point(0,0) fields"""
    p: Point = make_point(0, 0)
    match p:
        case (0, 0):
            return 0
        case (px, py):
            return px + py


@compile
def test_struct_bind_values() -> i32:
    """Bind Point(3,4) fields"""
    p: Point = make_point(3, 4)
    match p:
        case (0, 0):
            return 0
        case (px, py):
            return px + py


@compile
def test_struct_wildcard_y_axis() -> i32:
    """Wildcard: Point(0,5) on Y-axis"""
    p: Point = make_point(0, 5)
    match p:
        case (0, _):
            return 1
        case (_, 0):
            return 2
        case (_, _):
            return 3


@compile
def test_struct_wildcard_x_axis() -> i32:
    """Wildcard: Point(5,0) on X-axis"""
    p: Point = make_point(5, 0)
    match p:
        case (0, _):
            return 1
        case (_, 0):
            return 2
        case (_, _):
            return 3


@compile
def test_struct_wildcard_other() -> i32:
    """Wildcard: Point(3,4) general"""
    p: Point = make_point(3, 4)
    match p:
        case (0, _):
            return 1
        case (_, 0):
            return 2
        case (_, _):
            return 3


@compile
def test_struct_mixed_diagonal() -> i32:
    """Mixed pattern: Point(5,5) diagonal"""
    p: Point = make_point(5, 5)
    match p:
        case (0, 0):
            return 0
        case (0, y):
            return y
        case (x, 0):
            return x
        case (x, y):
            if x == y:
                return 999
            return x + y


@compile
def test_struct_mixed_y_axis() -> i32:
    """Mixed pattern: Point(0,7) on Y-axis"""
    p: Point = make_point(0, 7)
    match p:
        case (0, 0):
            return 0
        case (0, y):
            return y
        case (x, 0):
            return x
        case (x, y):
            if x == y:
                return 999
            return x + y


@compile
def test_struct_mixed_general() -> i32:
    """Mixed pattern: Point(3,4) general"""
    p: Point = make_point(3, 4)
    match p:
        case (0, 0):
            return 0
        case (0, y):
            return y
        case (x, 0):
            return x
        case (x, y):
            if x == y:
                return 999
            return x + y


@compile
def test_struct_guard_diagonal() -> i32:
    """Guard: Point(5,5) diagonal"""
    p: Point = make_point(5, 5)
    match p:
        case (0, 0):
            return 0
        case (x, y) if x == y:
            return 1
        case (x, y) if x > y:
            return 2
        case (x, y) if x < y:
            return 3
        case _:
            return 99


@compile
def test_struct_guard_above() -> i32:
    """Guard: Point(10,3) above diagonal"""
    p: Point = make_point(10, 3)
    match p:
        case (0, 0):
            return 0
        case (x, y) if x == y:
            return 1
        case (x, y) if x > y:
            return 2
        case (x, y) if x < y:
            return 3
        case _:
            return 99


@compile
def test_struct_guard_below() -> i32:
    """Guard: Point(3,10) below diagonal"""
    p: Point = make_point(3, 10)
    match p:
        case (0, 0):
            return 0
        case (x, y) if x == y:
            return 1
        case (x, y) if x > y:
            return 2
        case (x, y) if x < y:
            return 3
        case _:
            return 99


@compile
def test_struct_3d_origin() -> i32:
    """3D: Point3D(0,0,0) origin"""
    p: Point3D = make_point3d(0, 0, 0)
    match p:
        case (0, 0, 0):
            return 0
        case (x, 0, 0):
            return 1
        case (0, y, 0):
            return 2
        case (0, 0, z):
            return 3
        case (x, y, z):
            return x + y + z


@compile
def test_struct_3d_x_axis() -> i32:
    """3D: Point3D(5,0,0) on X-axis"""
    p: Point3D = make_point3d(5, 0, 0)
    match p:
        case (0, 0, 0):
            return 0
        case (x, 0, 0):
            return 1
        case (0, y, 0):
            return 2
        case (0, 0, z):
            return 3
        case (x, y, z):
            return x + y + z


@compile
def test_struct_3d_general() -> i32:
    """3D: Point3D(1,2,3) general"""
    p: Point3D = make_point3d(1, 2, 3)
    match p:
        case (0, 0, 0):
            return 0
        case (x, 0, 0):
            return 1
        case (0, y, 0):
            return 2
        case (0, 0, z):
            return 3
        case (x, y, z):
            return x + y + z


@compile
def test_nested_struct_origin() -> i32:
    """Nested: Rect at origin"""
    r: Rect = make_rect(0, 0, 10, 20)
    match r:
        case ((0, 0), _, _):
            return 1
        case ((px, py), w, h):
            return px + py + w + h


@compile
def test_nested_struct_general() -> i32:
    """Nested: Rect at (1,2) with size 3x4"""
    r: Rect = make_rect(1, 2, 3, 4)
    match r:
        case ((0, 0), _, _):
            return 1
        case ((px, py), w, h):
            return px + py + w + h


@compile
def test_nested_struct_tuple() -> i32:
    """Nested: Rect and Point3D tuple"""
    r: Rect = make_rect(1, 2, 3, 4)
    p: Point3D = make_point3d(5, 6, 7)
    match r, p:
        case (((0, 0), _, _), (5, 6, 7)):
            return 1
        case (((px, py), w, h), _):
            return px + py + w + h
        case (((0, 0), _, _), (x, y, z)):
            return x * y + z


@compile
def test_struct_tuple_literal() -> i32:
    """Match Point using tuple syntax: (0, 0)"""
    p: Point = make_point(0, 0)
    match p:
        case (0, 0):
            return 100
        case (1, 1):
            return 111
        case _:
            return 999


@compile
def test_struct_tuple_bind() -> i32:
    """Bind Point fields using tuple syntax: (x, y)"""
    p: Point = make_point(3, 4)
    match p:
        case (0, 0):
            return 0
        case (x, y):
            return x * 10 + y


@compile
def test_struct_tuple_wildcard() -> i32:
    """Wildcard with tuple syntax: (0, _)"""
    p: Point = make_point(0, 5)
    match p:
        case (0, _):
            return 1
        case (_, 0):
            return 2
        case (_, _):
            return 3


@compile
def test_struct_tuple_guard() -> i32:
    """Tuple syntax with guard"""
    p: Point = make_point(5, 5)
    match p:
        case (0, 0):
            return 0
        case (x, y) if x == y:
            return 1
        case (x, y) if x > y:
            return 2
        case _:
            return 3


@compile
def test_struct_tuple_3d() -> i32:
    """3D point with tuple syntax"""
    p: Point3D = make_point3d(1, 2, 3)
    match p:
        case (0, 0, 0):
            return 0
        case (x, 0, 0):
            return 1
        case (x, y, z):
            return x + y + z


class TestMatchStructDestructuring(unittest.TestCase):
    """Test struct destructuring feature"""
    
    def setUp(self):
        clear_registry()
    
    def tearDown(self):
        clear_registry()
    
    def test_struct_literal_origin(self):
        self.assertEqual(test_struct_literal_origin(), 0)
    
    def test_struct_literal_unit(self):
        self.assertEqual(test_struct_literal_unit(), 1)
    
    def test_struct_literal_other(self):
        self.assertEqual(test_struct_literal_other(), 99)
    
    def test_struct_bind_origin(self):
        self.assertEqual(test_struct_bind_origin(), 0)
    
    def test_struct_bind_values(self):
        self.assertEqual(test_struct_bind_values(), 7)
    
    def test_struct_wildcard_y_axis(self):
        self.assertEqual(test_struct_wildcard_y_axis(), 1)
    
    def test_struct_wildcard_x_axis(self):
        self.assertEqual(test_struct_wildcard_x_axis(), 2)
    
    def test_struct_wildcard_other(self):
        self.assertEqual(test_struct_wildcard_other(), 3)
    
    def test_struct_mixed_diagonal(self):
        self.assertEqual(test_struct_mixed_diagonal(), 999)
    
    def test_struct_mixed_y_axis(self):
        self.assertEqual(test_struct_mixed_y_axis(), 7)
    
    def test_struct_mixed_general(self):
        self.assertEqual(test_struct_mixed_general(), 7)
    
    def test_struct_with_guard_diagonal(self):
        self.assertEqual(test_struct_guard_diagonal(), 1)
    
    def test_struct_with_guard_above(self):
        self.assertEqual(test_struct_guard_above(), 2)
    
    def test_struct_with_guard_below(self):
        self.assertEqual(test_struct_guard_below(), 3)
    
    def test_struct_3d_origin(self):
        self.assertEqual(test_struct_3d_origin(), 0)
    
    def test_struct_3d_x_axis(self):
        self.assertEqual(test_struct_3d_x_axis(), 1)
    
    def test_struct_3d_general(self):
        self.assertEqual(test_struct_3d_general(), 6)
    
    def test_nested_struct_origin(self):
        self.assertEqual(test_nested_struct_origin(), 1)
    
    def test_nested_struct_general(self):
        self.assertEqual(test_nested_struct_general(), 10)

    def test_nested_struct_tuple(self):
        self.assertEqual(test_nested_struct_tuple(), 10)


class TestMatchStructTupleSyntax(unittest.TestCase):
    """Test struct matching with tuple syntax"""
    
    def setUp(self):
        clear_registry()
    
    def tearDown(self):
        clear_registry()
    
    def test_tuple_literal(self):
        self.assertEqual(test_struct_tuple_literal(), 100)
    
    def test_tuple_bind(self):
        self.assertEqual(test_struct_tuple_bind(), 34)
    
    def test_tuple_wildcard(self):
        self.assertEqual(test_struct_tuple_wildcard(), 1)
    
    def test_tuple_guard(self):
        self.assertEqual(test_struct_tuple_guard(), 1)
    
    def test_tuple_3d(self):
        self.assertEqual(test_struct_tuple_3d(), 6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
