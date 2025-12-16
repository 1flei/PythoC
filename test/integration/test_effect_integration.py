# -*- coding: utf-8 -*-
"""
Integration tests for the effect system.

Tests effect system integration with the pythoc compiler by verifying
the actual runtime behavior of compiled functions that use effects.

Pattern: Define @compile functions, call them, verify return values.

These tests focus on verifying that effect.xxx.method() calls are correctly
resolved to the appropriate implementation at compile time.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pythoc import compile, effect, i32, u64, void
from types import SimpleNamespace


# ============================================================================
# Effect implementations for testing
# Simple implementations that return fixed values (no state needed)
# ============================================================================

@compile
def _rng_v1_next() -> u64:
    """RNG v1: returns 42"""
    return u64(42)

@compile
def _rng_v1_seed(s: u64) -> void:
    pass

RNG_V1 = SimpleNamespace(next=_rng_v1_next, seed=_rng_v1_seed)


@compile
def _rng_v2_next() -> u64:
    """RNG v2: returns 999"""
    return u64(999)

@compile
def _rng_v2_seed(s: u64) -> void:
    pass

RNG_V2 = SimpleNamespace(next=_rng_v2_next, seed=_rng_v2_seed)


@compile
def _rng_v3_next() -> u64:
    """RNG v3: returns 777"""
    return u64(777)

@compile
def _rng_v3_seed(s: u64) -> void:
    pass

RNG_V3 = SimpleNamespace(next=_rng_v3_next, seed=_rng_v3_seed)


# ============================================================================
# Test: Basic effect usage in compiled code
# ============================================================================

# Set default effect
effect.default(rng=RNG_V1)


@compile
def test_effect_basic_call() -> u64:
    """Test basic effect.rng.next() call with default effect"""
    return effect.rng.next()


@compile
def test_effect_call_twice() -> u64:
    """Test calling effect.rng.next() twice"""
    a = effect.rng.next()
    b = effect.rng.next()
    return a + b


# ============================================================================
# Test: Effect override with suffix
# ============================================================================

# Define function with override to RNG_V2
with effect(rng=RNG_V2, suffix="v2"):
    @compile(suffix="v2")
    def test_effect_override_v2() -> u64:
        """This function uses RNG_V2 (returns 999)"""
        return effect.rng.next()


# Define function with override to RNG_V3
with effect(rng=RNG_V3, suffix="v3"):
    @compile(suffix="v3")
    def test_effect_override_v3() -> u64:
        """This function uses RNG_V3 (returns 777)"""
        return effect.rng.next()


# ============================================================================
# Test: Effect propagation through function calls
# ============================================================================

@compile
def _inner_use_effect() -> u64:
    """Inner function that uses effect (uses default RNG_V1)"""
    return effect.rng.next()


@compile
def test_effect_propagation() -> u64:
    """Outer function calls inner function that uses effect"""
    return _inner_use_effect()


# With override - inner function also uses override
with effect(rng=RNG_V2, suffix="prop_v2"):
    @compile(suffix="prop_v2")
    def _inner_use_effect_v2() -> u64:
        return effect.rng.next()
    
    @compile(suffix="prop_v2")
    def test_effect_propagation_v2() -> u64:
        return _inner_use_effect_v2()


# ============================================================================
# Test: Multiple effects
# ============================================================================

@compile
def _alloc_size() -> i32:
    return i32(4096)

MockAlloc = SimpleNamespace(size=_alloc_size)
effect.default(allocator=MockAlloc)


@compile
def test_multiple_effects() -> i32:
    """Test using multiple effects in one function"""
    rng_val = effect.rng.next()
    alloc_val = effect.allocator.size()
    # 42 + 4096 = 4138
    return i32(rng_val) + alloc_val


# ============================================================================
# Test: Direct assignment (non-overridable)
# ============================================================================

@compile
def _secure_next() -> u64:
    """Secure RNG: returns 888"""
    return u64(888)

SecureRNG = SimpleNamespace(next=_secure_next)

# Direct assignment - should NOT be overridable
effect.secure_rng = SecureRNG


@compile
def test_direct_assignment() -> u64:
    """Test direct assignment effect"""
    return effect.secure_rng.next()


# Try to override (should be ignored because secure_rng is direct assignment)
@compile
def _weak_next() -> u64:
    return u64(1)

WeakRNG = SimpleNamespace(next=_weak_next)

with effect(secure_rng=WeakRNG, suffix="weak"):
    @compile(suffix="weak")
    def test_direct_assignment_immune() -> u64:
        """This should still return 888, not 1 (direct assignment is immune)"""
        return effect.secure_rng.next()


# ============================================================================
# Test: Nested effect contexts
# ============================================================================

with effect(rng=RNG_V2, suffix="outer"):
    @compile(suffix="outer")
    def test_nested_outer() -> u64:
        """Uses RNG_V2 (999)"""
        return effect.rng.next()
    
    with effect(rng=RNG_V3, suffix="inner"):
        @compile(suffix="inner")
        def test_nested_inner() -> u64:
            """Uses RNG_V3 (777) - innermost context wins"""
            return effect.rng.next()


# ============================================================================
# Test: Effect flag (compile-time constant)
# ============================================================================

effect.default(MULTIPLIER=2)


@compile
def test_effect_flag() -> i32:
    """Test using effect as compile-time constant"""
    x: i32 = 10
    return x * i32(effect.MULTIPLIER)


with effect(MULTIPLIER=5, suffix="mult5"):
    @compile(suffix="mult5")
    def test_effect_flag_5() -> i32:
        x: i32 = 10
        return x * i32(effect.MULTIPLIER)


# ============================================================================
# Test: Effect with method that takes arguments
# ============================================================================

@compile
def _math_add(a: i32, b: i32) -> i32:
    return a + b

@compile
def _math_mul(a: i32, b: i32) -> i32:
    return a * b

MathAdd = SimpleNamespace(op=_math_add)
MathMul = SimpleNamespace(op=_math_mul)

effect.default(math=MathAdd)


@compile
def test_effect_with_args() -> i32:
    """Test effect method with arguments (default: add)"""
    return effect.math.op(i32(10), i32(20))


with effect(math=MathMul, suffix="mul"):
    @compile(suffix="mul")
    def test_effect_with_args_mul() -> i32:
        """Test effect method with arguments (override: mul)"""
        return effect.math.op(i32(10), i32(20))


# ============================================================================
# Main: Run all tests
# ============================================================================

if __name__ == "__main__":
    print("=== Effect System Integration Tests ===\n")
    
    # Basic effect usage
    print("--- Basic Effect Usage ---")
    result = test_effect_basic_call()
    print(f"test_effect_basic_call: {result} (expected: 42)")
    assert result == 42, f"Expected 42, got {result}"
    
    result = test_effect_call_twice()
    print(f"test_effect_call_twice: {result} (expected: 84)")
    assert result == 84, f"Expected 84, got {result}"
    
    # Effect override
    print("\n--- Effect Override ---")
    result = test_effect_override_v2()
    print(f"test_effect_override_v2: {result} (expected: 999)")
    assert result == 999, f"Expected 999, got {result}"
    
    result = test_effect_override_v3()
    print(f"test_effect_override_v3: {result} (expected: 777)")
    assert result == 777, f"Expected 777, got {result}"
    
    # Effect propagation
    print("\n--- Effect Propagation ---")
    result = test_effect_propagation()
    print(f"test_effect_propagation: {result} (expected: 42)")
    assert result == 42, f"Expected 42, got {result}"
    
    result = test_effect_propagation_v2()
    print(f"test_effect_propagation_v2: {result} (expected: 999)")
    assert result == 999, f"Expected 999, got {result}"
    
    # Multiple effects
    print("\n--- Multiple Effects ---")
    result = test_multiple_effects()
    print(f"test_multiple_effects: {result} (expected: 42 + 4096 = 4138)")
    assert result == 4138, f"Expected 4138, got {result}"
    
    # Direct assignment
    print("\n--- Direct Assignment ---")
    result = test_direct_assignment()
    print(f"test_direct_assignment: {result} (expected: 888)")
    assert result == 888, f"Expected 888, got {result}"
    
    result = test_direct_assignment_immune()
    print(f"test_direct_assignment_immune: {result} (expected: 888, immune to override)")
    assert result == 888, f"Expected 888 (immune to override), got {result}"
    
    # Nested contexts
    print("\n--- Nested Contexts ---")
    result = test_nested_outer()
    print(f"test_nested_outer: {result} (expected: 999)")
    assert result == 999, f"Expected 999, got {result}"
    
    result = test_nested_inner()
    print(f"test_nested_inner: {result} (expected: 777)")
    assert result == 777, f"Expected 777, got {result}"
    
    # Effect flags
    print("\n--- Effect Flags ---")
    result = test_effect_flag()
    print(f"test_effect_flag: {result} (expected: 20)")
    assert result == 20, f"Expected 20, got {result}"
    
    result = test_effect_flag_5()
    print(f"test_effect_flag_5: {result} (expected: 50)")
    assert result == 50, f"Expected 50, got {result}"
    
    # Effect with arguments
    print("\n--- Effect With Arguments ---")
    result = test_effect_with_args()
    print(f"test_effect_with_args: {result} (expected: 30)")
    assert result == 30, f"Expected 30, got {result}"
    
    result = test_effect_with_args_mul()
    print(f"test_effect_with_args_mul: {result} (expected: 200)")
    assert result == 200, f"Expected 200, got {result}"
    
    print("\n=== All tests passed! ===")
