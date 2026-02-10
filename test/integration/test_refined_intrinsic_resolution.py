#!/usr/bin/env python3
"""
Test that compiler-generated intrinsic references (assume, move, label, etc.)
are resolved via the __pc_intrinsics namespace, not user_globals.

Key scenario: refine() generates AST that calls assume(), but the user never
imported assume.  Before the fix this would fail with
"Variable 'assume' not defined".
"""

# NOTE: intentionally do NOT import assume here
from pythoc import compile, i32, bool, refine


@compile
def is_positive(x: i32) -> bool:
    return x > 0


@compile
def test_refine_without_assume_import() -> i32:
    """refine() should work even though assume is not in user_globals."""
    for x in refine(10, is_positive):
        return x
    else:
        return -1


@compile
def test_refine_failure_without_assume_import() -> i32:
    """else branch should execute when predicate fails."""
    for x in refine(-5, is_positive):
        return x
    else:
        return -999


import unittest


class TestRefinedIntrinsicResolution(unittest.TestCase):
    """Test intrinsic resolution for compiler-generated AST."""

    def test_refine_without_assume_import_success(self):
        """refine() works without importing assume (success path)."""
        self.assertEqual(test_refine_without_assume_import(), 10)

    def test_refine_without_assume_import_failure(self):
        """refine() works without importing assume (failure path)."""
        self.assertEqual(test_refine_failure_without_assume_import(), -999)


if __name__ == "__main__":
    unittest.main()
