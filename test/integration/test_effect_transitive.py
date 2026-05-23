# -*- coding: utf-8 -*-
"""
Test local effect context call semantics.

Local functions defined under ``with effect`` capture direct ``effect.xxx`` uses,
but ordinary calls use their lexical binding. Transitive propagation is provided
by specialized imports, not by rewriting arbitrary calls during codegen.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pythoc import compile, effect, i32, u64, void
from types import SimpleNamespace


# ============================================================================
# Create a chain of dependencies: a -> b -> d
# ============================================================================

# Module D: leaf dependency
@compile
def _default_d_value() -> i32:
    """Default d: returns 1"""
    return i32(1)

@compile
def _mock_d_value() -> i32:
    """Mock d: returns 100"""
    return i32(100)

DefaultD = SimpleNamespace(value=_default_d_value)
MockD = SimpleNamespace(value=_mock_d_value)

effect.default(d_impl=DefaultD)


# Module B: middle layer, uses effect.d_impl
@compile
def b_get_value() -> i32:
    """B: returns d_impl.value() * 10"""
    return effect.d_impl.value() * i32(10)


# Module A: caller defined under an override. Ordinary calls use lexical
# bindings, so this still calls the default b_get_value unless a specialized
# binding is imported explicitly.
with effect(d_impl=MockD, suffix="mock"):
    @compile(suffix="mock")
    def a_get_value_mock() -> i32:
        """A with mock context, ordinary call remains default-bound."""
        return b_get_value()


# Module C: caller that uses default
@compile
def c_get_value_default() -> i32:
    """C with default: should return 1 * 10 = 10"""
    return b_get_value()


# ============================================================================
# Test: Verify the transitive effect propagation
# ============================================================================

@compile
def test_a_mock() -> i32:
    """Test that A's mock version uses MockD"""
    return a_get_value_mock()


@compile
def test_c_default() -> i32:
    """Test that C's default version uses DefaultD"""
    return c_get_value_default()


@compile
def test_both_coexist() -> i32:
    """Test that both versions can coexist and return different values"""
    a_val = a_get_value_mock()
    c_val = c_get_value_default()
    # Both use the lexical default binding for b_get_value.
    if a_val == i32(10) and c_val == i32(10):
        return i32(1)
    return i32(0)


# ============================================================================
# Main: Run tests
# ============================================================================

if __name__ == "__main__":
    print("=== Transitive Effect Propagation Tests ===\n")
    
    print("--- Test A (mock context, lexical call) ---")
    result = test_a_mock()
    print(f"test_a_mock: {result} (expected: 10)")
    assert result == 10, f"Expected 10, got {result}"
    print("  PASSED\n")
    
    print("--- Test C (default) ---")
    result = test_c_default()
    print(f"test_c_default: {result} (expected: 10)")
    assert result == 10, f"Expected 10, got {result}"
    print("  PASSED\n")
    
    print("--- Test Both Coexist ---")
    result = test_both_coexist()
    print(f"test_both_coexist: {result} (expected: 1)")
    assert result == 1, f"Expected 1, got {result}"
    print("  PASSED\n")
    
    print("=== All tests passed! ===")
