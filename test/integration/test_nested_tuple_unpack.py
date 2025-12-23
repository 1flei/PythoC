"""
Test nested tuple unpacking support in pythoc.

Goal: Support `for k, (a, b) in dict.items()` syntax.

Current limitation: Only flat tuple unpacking is supported.
"""
from __future__ import annotations
import unittest
import ast

from pythoc import compile, i32, i8, ptr, void, struct
from pythoc.libc.stdio import printf


# =============================================================================
# Test data at module level
# =============================================================================

_type_info = {
    1: ('int', True),
    2: ('float', False),
    3: ('char', True),
}


# =============================================================================
# Test 1: Current workaround - works
# =============================================================================

@compile
def test_workaround(val: i32) -> i32:
    """Current workaround: access tuple elements via indexing"""
    for k in _type_info:
        name = _type_info[k][0]
        flag = _type_info[k][1]
        if val == k:
            if flag:
                return len(name) + 100
            else:
                return len(name)
    return 0


# =============================================================================
# Test 2: Flat tuple unpacking - should work
# =============================================================================

_flat_data = [
    (1, 10),
    (2, 20),
    (3, 30),
]

@compile
def test_flat_unpack(val: i32) -> i32:
    """Flat tuple unpacking: for a, b in list"""
    for k, v in _flat_data:
        if val == k:
            return v
    return 0


# =============================================================================
# Test 3: Nested tuple unpacking - NOW SUPPORTED
# =============================================================================

@compile
def test_nested_unpack(val: i32) -> i32:
    """Nested tuple unpacking: for k, (name, flag) in dict.items()"""
    for k, (name, flag) in _type_info.items():
        if val == k:
            if flag:
                return len(name) + 100
            else:
                return len(name)
    return 0


# =============================================================================
# Test 4: Alternative - use .items() with flat access
# =============================================================================

@compile
def test_items_flat(val: i32) -> i32:
    """Use .items() but access tuple elements via indexing"""
    for k, v in _type_info.items():
        # v is the tuple (name, flag)
        name = v[0]
        flag = v[1]
        if val == k:
            if flag:
                return len(name) + 100
            else:
                return len(name)
    return 0


# =============================================================================
# Test 5: Deeper nested tuple unpacking
# =============================================================================

_deep_nested = [
    (1, (('a', 'b'), 10)),
    (2, (('c', 'd'), 20)),
    (3, (('e', 'f'), 30)),
]

@compile
def test_deep_nested(val: i32) -> i32:
    """Test 3-level nested tuple unpacking: for k, ((a, b), n) in list"""
    for k, ((a, b), n) in _deep_nested:
        if val == k:
            # Return len(a) + len(b) + n
            return len(a) + len(b) + n
    return 0


# =============================================================================
# Test 6: Mixed nested patterns
# =============================================================================

_mixed_nested = {
    'x': ((1, 2), 'hello'),
    'y': ((3, 4), 'world'),
}

@compile
def test_mixed_nested(key: i32) -> i32:
    """Test mixed nested: for k, ((a, b), s) in dict.items()"""
    for k, ((a, b), s) in _mixed_nested.items():
        if key == 0 and k == 'x':
            return a + b + len(s)  # 1 + 2 + 5 = 8
        if key == 1 and k == 'y':
            return a + b + len(s)  # 3 + 4 + 5 = 12
    return 0


# =============================================================================
# Main
# =============================================================================

@compile
def main() -> i32:
    printf("=== Nested Tuple Unpacking Tests ===\n\n")
    
    errors: i32 = 0
    
    # Test 1: Workaround
    printf("Test 1: Workaround (index access)\n")
    r1 = test_workaround(1)
    printf("  test_workaround(1) = %d (expected 103)\n", r1)
    if r1 != 103:
        errors = errors + 1
    
    # Test 2: Flat unpack
    printf("\nTest 2: Flat tuple unpacking\n")
    r2 = test_flat_unpack(2)
    printf("  test_flat_unpack(2) = %d (expected 20)\n", r2)
    if r2 != 20:
        errors = errors + 1
    
    # Test 3: Nested unpack - NOW SUPPORTED
    printf("\nTest 3: Nested tuple unpacking\n")
    r3 = test_nested_unpack(2)
    printf("  test_nested_unpack(2) = %d (expected 5 for 'float')\n", r3)
    if r3 != 5:
        errors = errors + 1
    r3b = test_nested_unpack(1)
    printf("  test_nested_unpack(1) = %d (expected 103 for 'int'+100)\n", r3b)
    if r3b != 103:
        errors = errors + 1
    
    # Test 4: items() with flat access
    printf("\nTest 4: items() with flat access\n")
    r4 = test_items_flat(3)
    printf("  test_items_flat(3) = %d (expected 104)\n", r4)
    if r4 != 104:
        errors = errors + 1
    
    # Test 5: Deep nested (3 levels)
    printf("\nTest 5: Deep nested tuple unpacking (3 levels)\n")
    r5 = test_deep_nested(2)
    printf("  test_deep_nested(2) = %d (expected 22 = 1+1+20)\n", r5)
    if r5 != 22:
        errors = errors + 1
    
    # Test 6: Mixed nested with dict
    printf("\nTest 6: Mixed nested with dict\n")
    r6a = test_mixed_nested(0)
    printf("  test_mixed_nested(0) = %d (expected 8 = 1+2+5)\n", r6a)
    if r6a != 8:
        errors = errors + 1
    r6b = test_mixed_nested(1)
    printf("  test_mixed_nested(1) = %d (expected 12 = 3+4+5)\n", r6b)
    if r6b != 12:
        errors = errors + 1
    
    printf("\n=== Results: %d errors ===\n", errors)
    return errors


class TestNestedTupleUnpack(unittest.TestCase):
    def test_all(self):
        result = main()
        self.assertEqual(result, 0)


if __name__ == '__main__':
    unittest.main()
