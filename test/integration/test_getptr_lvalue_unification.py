"""
Test unified implementation of ptr() and visit_lvalue

Verifies that ptr() function now uses the same code path as visit_lvalue
"""

from pythoc import compile, i32, ptr, array


@compile
def test_ptr_variable() -> i32:
    """ptr(variable)"""
    x: i32 = 42
    p: ptr[i32] = ptr(x)
    return p[0]


@compile
def test_ptr_array_element() -> i32:
    """ptr(arr[i])"""
    arr: array[i32, 3] = [10, 20, 30]
    p: ptr[i32] = ptr(arr[1])
    return p[0]


@compile
class Point:
    x: i32
    y: i32


@compile
def test_ptr_struct_field() -> i32:
    """ptr(struct.field)"""
    point: Point = Point()
    point.x = 100
    point.y = 200
    p: ptr[i32] = ptr(point.x)
    return p[0]


@compile
def test_ptr_deref() -> i32:
    """ptr(*ptr)"""
    x: i32 = 99
    p1: ptr[i32] = ptr(x)
    p2: ptr[i32] = ptr(p1[0])
    return p2[0]


@compile
def test_assignment_lvalue() -> i32:
    """lvalue assignment"""
    x: i32 = 10
    x = 20
    return x


@compile
def test_array_assignment_lvalue() -> i32:
    """array element lvalue assignment"""
    arr: array[i32, 3] = [1, 2, 3]
    arr[1] = 99
    return arr[1]


@compile
def test_struct_assignment_lvalue() -> i32:
    """struct field lvalue assignment"""
    point: Point = Point()
    point.x = 10
    point.y = 20
    point.x = 999
    return point.x


import unittest


class TestGetptrLvalueUnification(unittest.TestCase):
    """Test unified implementation of ptr() and visit_lvalue"""

    def test_ptr_variable(self):
        """ptr(variable)"""
        self.assertEqual(test_ptr_variable(), 42)

    def test_ptr_array_element(self):
        """ptr(arr[i])"""
        self.assertEqual(test_ptr_array_element(), 20)

    def test_ptr_struct_field(self):
        """ptr(struct.field)"""
        self.assertEqual(test_ptr_struct_field(), 100)

    def test_ptr_deref(self):
        """ptr(*ptr)"""
        self.assertEqual(test_ptr_deref(), 99)

    def test_assignment_lvalue(self):
        """lvalue assignment"""
        self.assertEqual(test_assignment_lvalue(), 20)

    def test_array_assignment_lvalue(self):
        """array element lvalue assignment"""
        self.assertEqual(test_array_assignment_lvalue(), 99)

    def test_struct_assignment_lvalue(self):
        """struct field lvalue assignment"""
        self.assertEqual(test_struct_assignment_lvalue(), 999)


if __name__ == '__main__':
    unittest.main()
