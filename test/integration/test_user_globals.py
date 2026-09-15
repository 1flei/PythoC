"""
Test user globals support - accessing constants, type aliases, and imported names
from the user's module in compiled functions.

This enables:
- Constant definitions: DEFAULT_SIZE = 100
- Type aliases: MyInt = i32
- Function/type renaming: my_malloc = malloc

Note: All @compile functions are defined at module level to ensure
they are compiled before any native execution happens.
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pythoc import compile, i32, i64, f64, ptr, nullptr, sizeof
from pythoc.libc.stdlib import malloc, free
from pythoc.build.output_manager import flush_all_pending_outputs, clear_failed_group

from test.utils.test_utils import DeferredTestCase

# Define constants that will be used in compiled functions
DEFAULT_VALUE = 42
MAX_SIZE = 100
PI = 3.14159

# Type aliases
MyInt = i32
MyFloat = f64

# Computed constants
DOUBLE_DEFAULT = DEFAULT_VALUE * 2

# Negative constant
NEGATIVE_VALUE = -50

# Zero constant
ZERO = 0

# Function renaming: module-level aliases of imported libc functions
my_malloc = malloc
my_free = free


# ============================================================================
# All @compile functions defined at module level
# ============================================================================

@compile
def get_default_value() -> i32:
    return DEFAULT_VALUE

@compile
def get_pi() -> f64:
    return PI

@compile
def compute_with_constants() -> i32:
    x: i32 = DEFAULT_VALUE
    y: i32 = MAX_SIZE
    return x + y

@compile
def use_type_alias(x: MyInt) -> MyInt:
    y: MyInt = x + 10
    return y

@compile
def compute_area(radius: f64) -> f64:
    area: f64 = PI * radius * radius
    return area

@compile
def is_max_size(x: i32) -> i32:
    if x == MAX_SIZE:
        return 1
    else:
        return 0

@compile
def sum_to_default() -> i32:
    total: i32 = 0
    i: i32 = 0
    while i < DEFAULT_VALUE:
        total = total + i
        i = i + 1
    return total

@compile
def get_double_default() -> i32:
    return DOUBLE_DEFAULT

@compile
def get_negative() -> i32:
    return NEGATIVE_VALUE

@compile
def get_zero() -> i32:
    return ZERO


@compile(suffix="user_globals_ext")
def allocate_with_renamed() -> i32:
    """Use renamed malloc/free from module globals"""
    p = ptr[i32](my_malloc(sizeof(i32)))
    p[0] = 77
    v: i32 = p[0]
    my_free(p)
    return v


@compile(suffix="user_globals_ext")
def check_nullptr() -> i32:
    """nullptr imported into the user module works in compiled code"""
    p: ptr[i32] = nullptr
    if p == nullptr:
        return 1
    return 0


# ============================================================================
# Test classes - only call the pre-compiled functions
# ============================================================================

class TestUserGlobals(unittest.TestCase):
    """Test accessing user module's global namespace in compiled functions"""
    
    def test_integer_constant(self):
        """Test using integer constants from module globals"""
        result = get_default_value()
        self.assertEqual(result, 42)
    
    def test_float_constant(self):
        """Test using float constants from module globals"""
        result = get_pi()
        self.assertAlmostEqual(result, 3.14159, places=5)
    
    def test_multiple_constants(self):
        """Test using multiple constants in one function"""
        result = compute_with_constants()
        self.assertEqual(result, 142)
    
    def test_type_alias(self):
        """Test using type aliases from module globals"""
        result = use_type_alias(5)
        self.assertEqual(result, 15)
    
    def test_function_renaming(self):
        """Test using renamed functions from module globals"""
        result = allocate_with_renamed()
        self.assertEqual(result, 77)

    def test_nullptr_from_import(self):
        """Test that nullptr works when imported in user module"""
        result = check_nullptr()
        self.assertEqual(result, 1)
    
    def test_constant_in_expression(self):
        """Test using constants in complex expressions"""
        result = compute_area(2.0)
        expected = 3.14159 * 2.0 * 2.0
        self.assertAlmostEqual(result, expected, places=4)
    
    def test_constant_comparison(self):
        """Test using constants in comparisons"""
        self.assertEqual(is_max_size(100), 1)
        self.assertEqual(is_max_size(50), 0)
    
    def test_constant_in_loop(self):
        """Test using constants in loop conditions"""
        # Sum from 0 to 41 = 41 * 42 / 2 = 861
        result = sum_to_default()
        expected = sum(range(42))
        self.assertEqual(result, expected)


class TestUserGlobalsEdgeCases(DeferredTestCase):
    """Test edge cases and error handling for user globals"""

    def test_undefined_constant_error(self):
        """Test that using undefined constant raises NameError"""
        source_file = os.path.abspath(__file__)
        group_key = (source_file, 'module', 'ug_undefined_const')
        try:
            with self.assertRaises((NameError, RuntimeError)) as ctx:
                @compile(suffix="ug_undefined_const")
                def use_undefined() -> i32:
                    return UNDEFINED_CONSTANT_XYZ
                flush_all_pending_outputs()
            self.assertIn("not defined", str(ctx.exception))
        finally:
            clear_failed_group(group_key)

    def test_type_as_value_error(self):
        """Test that using type as value raises TypeError"""
        source_file = os.path.abspath(__file__)
        group_key = (source_file, 'module', 'ug_type_as_value')
        try:
            with self.assertRaises((TypeError, RuntimeError)) as ctx:
                @compile(suffix="ug_type_as_value")
                def use_type_as_value() -> i32:
                    x: i32 = i32
                    return x
                flush_all_pending_outputs()
            self.assertIn("promote", str(ctx.exception))
        finally:
            clear_failed_group(group_key)


class TestConstantDefinitions(unittest.TestCase):
    """Test various ways to define and use constants"""
    
    def test_computed_constant(self):
        """Test using computed constants"""
        result = get_double_default()
        self.assertEqual(result, 84)
    
    def test_negative_constant(self):
        """Test using negative constants"""
        result = get_negative()
        self.assertEqual(result, -50)
    
    def test_zero_constant(self):
        """Test using zero constant"""
        result = get_zero()
        self.assertEqual(result, 0)


if __name__ == '__main__':
    unittest.main()
