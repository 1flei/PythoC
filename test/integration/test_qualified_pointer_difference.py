"""Pointer subtraction uses the pointee type through storage qualifiers."""

import unittest

import pythoc
from pythoc import array, compile, i32, i64, ptr, static, volatile

pythoc.init()


@compile
def qualified_pointer_difference() -> i64:
    items: array[i32, 5] = [1, 2, 3, 4, 5]
    first: static[ptr[i32]]
    last: volatile[ptr[i32]] = ptr(items[4])
    first = ptr(items[1])
    return last - first


class QualifiedPointerDifferenceTest(unittest.TestCase):
    def test_qualified_storage_with_matching_pointees(self):
        self.assertEqual(qualified_pointer_difference(), 3)


if __name__ == '__main__':
    unittest.main()
