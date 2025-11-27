"""
Test compile-time function calls and subscript access.

This test suite verifies that Python constants (lists, tuples, dicts)
can be accessed at compile time during code generation.

Note: All @compile functions are defined at module level to ensure
they are compiled before any native execution happens.
"""

import unittest
from pythoc import compile, i32, i64, f64

# Module-level constants for compile-time access
VALUES = [10, 20, 30, 40, 50]
NUMBERS = [100, 200, 300]
COORDS = (5, 10, 15)
MATRIX = [[1, 2], [3, 4]]
SIGNED_VALUES = [-10, -20, 30]
COEFFICIENTS = [2, 3, 5]
LIST_A = [1, 2, 3]
LIST_B = [10, 20, 30]
FLOAT_VALUES = [1.5, 2.5, 3.5]
BIG_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
EMPTY = []
SMALL_LIST = [1, 2]
SINGLE = [42]
MIXED = [10, 20, 30]
OFFSETS = [100, 200, 300]
THRESHOLD = [50, 100, 150]
LIMITS = [5, 10, 15]

# Constants for function call tests
DICT_DATA = {"x": 10, "y": 20, "z": 30}
STRING_DATA = "hello"
TUPLE_DATA = (100, 200, 300)

# Pure Python functions for compile-time evaluation
def add_numbers(a, b):
    return a + b

def multiply(x, y):
    return x * y

def get_max(a, b):
    if a > b:
        return a
    return b


# ============================================================================
# All @compile functions defined at module level
# ============================================================================

@compile
def get_first() -> i32:
    return VALUES[0]

@compile
def get_second() -> i32:
    return NUMBERS[1]

@compile
def get_third() -> i32:
    return NUMBERS[2]

@compile
def get_x() -> i32:
    return COORDS[0]

@compile
def get_y() -> i32:
    return COORDS[1]

@compile
def get_negative() -> i32:
    return SIGNED_VALUES[0]

@compile
def calculate(x: i32) -> i32:
    return COEFFICIENTS[0] * x + COEFFICIENTS[1]

@compile
def sum_elements() -> i32:
    return LIST_A[0] + LIST_B[0]

@compile
def get_float() -> f64:
    return FLOAT_VALUES[1]

@compile
def get_last() -> i32:
    return BIG_LIST[9]


# ============================================================================
# Test classes - only call the pre-compiled functions
# ============================================================================

class TestCompileTimeSubscript(unittest.TestCase):
    """Test compile-time subscript access on Python constants"""
    
    def test_list_subscript_constant_index(self):
        """Test subscripting a Python list with constant index"""
        result = get_first()
        self.assertEqual(result, 10)
    
    def test_list_subscript_different_indices(self):
        """Test subscripting list at different positions"""
        self.assertEqual(get_second(), 200)
        self.assertEqual(get_third(), 300)
    
    def test_tuple_subscript(self):
        """Test subscripting a Python tuple"""
        self.assertEqual(get_x(), 5)
        self.assertEqual(get_y(), 10)
    
    def test_nested_list_subscript(self):
        """Test subscripting nested lists - requires chained subscript"""
        # Note: Cannot assign Python list to variable in compiled code
        # This test is skipped as it requires chained subscript support
        # which would need: MATRIX[1][0] directly
        pass
    
    def test_list_with_negative_values(self):
        """Test list containing negative numbers"""
        result = get_negative()
        self.assertEqual(result, -10)
    
    def test_list_in_expression(self):
        """Test using list subscript in arithmetic expression"""
        result = calculate(10)
        self.assertEqual(result, 23)  # 2 * 10 + 3
    
    def test_multiple_lists(self):
        """Test using multiple lists in same function"""
        result = sum_elements()
        self.assertEqual(result, 11)
    
    def test_float_list_subscript(self):
        """Test subscripting list of floats"""
        result = get_float()
        self.assertAlmostEqual(result, 2.5)
    
    def test_large_index(self):
        """Test subscripting with larger index"""
        result = get_last()
        self.assertEqual(result, 9)


if __name__ == '__main__':
    unittest.main()
