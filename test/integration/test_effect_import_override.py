# -*- coding: utf-8 -*-
"""
Integration test for effect override via import.

This test demonstrates the "Caller Override" feature described in the design doc,
where importing a module within a `with effect(...)` block should produce
a version of that module's functions compiled with the overridden effects.

Design Doc Reference (Section 6):
    with effect(rng=crypto_rng, suffix="crypto"):
        from rng_lib import random as crypto_random

This should produce a version of `random` that uses `crypto_rng` instead of
the library's default RNG.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pythoc import compile, effect, u64, i32, void
from types import SimpleNamespace


# ============================================================================
# Define alternative RNG implementations for testing import override
# Simple implementations that return fixed values (no state needed)
# ============================================================================

@compile
def _mock_rng_next() -> u64:
    """Mock RNG: always returns 999"""
    return u64(999)

@compile
def _mock_rng_seed(s: u64) -> void:
    pass

MockRNG = SimpleNamespace(next=_mock_rng_next, seed=_mock_rng_seed)


@compile
def _fixed_100_next() -> u64:
    """Fixed RNG: always returns 100"""
    return u64(100)

@compile
def _fixed_100_seed(s: u64) -> void:
    pass

Fixed100RNG = SimpleNamespace(next=_fixed_100_next, seed=_fixed_100_seed)


# ============================================================================
# Test 1: Default import uses library's default RNG
# ============================================================================

# Import the library normally (uses DefaultRNG from rng_lib)
from effect_lib.rng_lib import random as lib_random, seed as lib_seed


@compile
def test_default_import() -> u64:
    """Test that normal import uses library's default RNG (LCG)"""
    lib_seed(u64(12345))
    return lib_random()


# ============================================================================
# Test 2: Effect override for functions defined in same module (should work)
# ============================================================================

# This pattern SHOULD work: define effect, then define function using it
effect.default(test_rng=MockRNG)


@compile
def test_same_module_default() -> u64:
    """Function using effect defined in same module"""
    return effect.test_rng.next()


# Override and define new function
with effect(test_rng=Fixed100RNG, suffix="fixed100"):
    @compile(suffix="fixed100")
    def test_same_module_override() -> u64:
        """Function with overridden effect (uses Fixed100RNG)"""
        return effect.test_rng.next()


# ============================================================================
# Test 3: Import override
# ============================================================================

# Attempt import override
with effect(rng=MockRNG, suffix="mock"):
    from effect_lib.rng_lib import random as mock_random


@compile
def test_import_override() -> u64:
    """Test import override.
    
    mock_random should use MockRNG (return 999)
    """
    return mock_random()


# ============================================================================
# Test 4: Both versions coexist
# ============================================================================

@compile
def test_both_versions() -> i32:
    """Test that default and override versions can coexist.
    
    Returns:
        1 if both work correctly (mock=999, default!=999)
        0 if override failed (both use default)
    """
    default_val = lib_random()
    mock_val = mock_random()
    
    # If import override works: mock_val == 999, default_val != 999
    if mock_val == u64(999) and default_val != u64(999):
        return i32(1)
    return i32(0)


# ============================================================================
# Test 5: Effect propagation through library functions
# ============================================================================

@compile
def test_library_function_uses_effect() -> u64:
    """Test that library functions use effect correctly.
    
    lib_random() calls effect.rng.next() internally.
    The effect should be resolved at compile time.
    """
    lib_seed(u64(42))
    return lib_random()


# ============================================================================
# Test 6: Fixed100 RNG import override
# ============================================================================

with effect(rng=Fixed100RNG, suffix="fixed100_lib"):
    from effect_lib.rng_lib import random as fixed100_random


@compile
def test_fixed100_import_override() -> u64:
    """Test Fixed100 RNG import override.
    
    fixed100_random should use Fixed100RNG (return 100)
    """
    return fixed100_random()


# ============================================================================
# Test 7: Transitive import override
# This tests that when importing a function, its transitive dependencies
# also use the overridden effect.
# ============================================================================

# Import transitive library
from effect_lib.rng_lib_transitive import get_random_sum as default_random_sum


@compile
def _fixed_200_next() -> u64:
    """Fixed RNG: always returns 200"""
    return u64(200)

@compile
def _fixed_200_seed(s: u64) -> void:
    pass

Fixed200RNG = SimpleNamespace(next=_fixed_200_next, seed=_fixed_200_seed)


# Import with effect override - should propagate to random_a and random_b
with effect(rng=Fixed200RNG, suffix="fixed200"):
    from effect_lib.rng_lib_transitive import get_random_sum as fixed200_random_sum


@compile
def test_transitive_import_default() -> u64:
    """Test default import of transitive function.
    
    get_random_sum() calls random_a() + random_b(), both use DefaultRNG.
    """
    return default_random_sum()


@compile
def test_transitive_import_override() -> u64:
    """Test transitive import override.
    
    fixed200_random_sum calls random_a_fixed200 + random_b_fixed200
    Both return 200, so sum = 400
    """
    return fixed200_random_sum()


# ============================================================================
# Main: Run all tests
# ============================================================================

if __name__ == "__main__":
    print("=== Effect Import Override Tests ===\n")
    
    # Test 1: Default import
    print("--- Test 1: Default Import ---")
    result = test_default_import()
    print(f"test_default_import: {result}")
    print("  (Should be LCG result for seed 12345)")
    # LCG: state = 12345 * 1103515245 + 12345 = 13615648770190
    # But u64 wraps, so we just check it's not 999
    assert result != 999, "Default import should NOT return mock value 999"
    print("  PASSED: Not mock value\n")
    
    # Test 2: Same module effect (should work)
    print("--- Test 2: Same Module Effect ---")
    result = test_same_module_default()
    print(f"test_same_module_default: {result} (expected: 999)")
    assert result == 999, f"Expected 999, got {result}"
    print("  PASSED\n")
    
    result = test_same_module_override()
    print(f"test_same_module_override: {result} (expected: 100)")
    assert result == 100, f"Expected 100, got {result}"
    print("  PASSED\n")
    
    # Test 3: Import override
    print("--- Test 3: Import Override ---")
    result = test_import_override()
    print(f"test_import_override: {result}")
    assert result == 999, f"Expected 999, got {result}"
    print("  PASSED: Import override works!\n")
    
    # Test 4: Both versions coexist
    print("--- Test 4: Both Versions Coexist ---")
    result = test_both_versions()
    print(f"test_both_versions: {result}")
    assert result == 1, f"Expected 1, got {result}"
    print("  PASSED: Both versions work independently!\n")
    
    # Test 5: Library function uses effect
    print("--- Test 5: Library Function Uses Effect ---")
    result = test_library_function_uses_effect()
    print(f"test_library_function_uses_effect: {result}")
    # LCG: 42 * 1103515245 + 12345 = 46347640332
    assert result != 999, "Library function should use DefaultRNG, not mock"
    print("  PASSED: Uses library's default RNG\n")
    
    # Test 6: Fixed100 import override
    print("--- Test 6: Fixed100 Import Override ---")
    result = test_fixed100_import_override()
    print(f"test_fixed100_import_override: {result}")
    assert result == 100, f"Expected 100, got {result}"
    print("  PASSED: Fixed100 import override works!\n")
    
    # Test 7: Transitive import override
    print("--- Test 7: Transitive Import Override ---")
    
    # 7a: Default import
    result = test_transitive_import_default()
    print(f"test_transitive_import_default: {result}")
    print("  (Should be LCG result, not 400)")
    assert result != 400, "Default import should NOT return fixed value 400"
    print("  PASSED: Uses default RNG\n")
    
    # 7b: Override import - tests transitive propagation
    result = test_transitive_import_override()
    print(f"test_transitive_import_override: {result}")
    assert result == 400, f"Expected 400, got {result}"
    print("  PASSED: Transitive import override works!")
    print("  Both random_a and random_b used Fixed200RNG (200 + 200 = 400)\n")
    
    print("=== Summary ===")
    print("All tests PASSED!")
