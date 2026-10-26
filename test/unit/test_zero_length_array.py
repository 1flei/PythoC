"""Type construction and decay for zero-length array extensions."""

import unittest
from unittest.mock import patch

from pythoc import array, i32
from pythoc.logger import logger


class TestZeroLengthArrayType(unittest.TestCase):
    def test_zero_dimensions_keep_element_alignment(self):
        for dimensions in ((0,), (3, 0), (0, 3)):
            with self.subTest(dimensions=dimensions):
                value = array[(i32, *dimensions)]
                self.assertEqual(value.dimensions, dimensions)
                self.assertEqual(value.get_size_bytes(), 0)
                self.assertEqual(value.get_alignment(), i32.get_alignment())

    def test_decay_preserves_nested_zero_dimension(self):
        value = array[i32, 3, 0].get_decay_pointer_type()
        self.assertEqual(value.pointee_type.dimensions, (0,))
        self.assertIs(value.pointee_type.element_type, i32)

    def test_invalid_dimensions_are_rejected(self):
        with patch.object(logger, 'raise_on_error', True):
            for dimension in (-1, 1.5, 'zero'):
                with self.subTest(dimension=dimension):
                    with self.assertRaises(TypeError):
                        array[i32, dimension]


if __name__ == '__main__':
    unittest.main()
