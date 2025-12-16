# -*- coding: utf-8 -*-
"""
Test transitive effect propagation.

Test scenario:
    a -> b (with effect override)
    c -> b (without effect override)
    b -> d (d is a dependency of b)

When a imports b with effect suffix, it should:
1. Get b_suffix (renamed version of b)
2. Also get d_suffix (renamed version of d, since b_suffix depends on d_suffix)
3. c should still use the default b (not b_suffix)
4. c's b should use the default d (not d_suffix)
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


# Module A: caller that overrides effect
# This should create b_get_value_mock which internally calls _mock_d_value
with effect(d_impl=MockD, suffix="mock"):
    @compile(suffix="mock")
    def a_get_value_mock() -> i32:
        """A with mock: should return 100 * 10 = 1000"""
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
    # a_val should be 1000, c_val should be 10
    # Return 1 if correct, 0 if wrong
    if a_val == i32(1000) and c_val == i32(10):
        return i32(1)
    return i32(0)


# ============================================================================
# Main: Run tests
# ============================================================================

if __name__ == "__main__":
    print("=== Transitive Effect Propagation Tests ===\n")
    
    print("--- Test A (mock) ---")
    result = test_a_mock()
    print(f"test_a_mock: {result} (expected: 1000)")
    assert result == 1000, f"Expected 1000, got {result}"
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
