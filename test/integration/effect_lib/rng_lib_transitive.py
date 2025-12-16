# -*- coding: utf-8 -*-
"""
RNG library with transitive dependencies for testing import transitive propagation.

This module has a call chain:
    get_random_sum() -> random_a() -> effect.rng.next()
                     -> random_b() -> effect.rng.next()

When importing get_random_sum with effect override, both random_a and random_b
should use the overridden effect.
"""

from pythoc import compile, effect, u64, void, ptr, static
from types import SimpleNamespace


# ============================================================
# Default RNG implementation (simple LCG)
# ============================================================

@compile
def _default_get_state() -> ptr[u64]:
    """Return pointer to static LCG state"""
    state: static[u64] = 12345
    return ptr(state)


@compile
def _default_seed(s: u64) -> void:
    """Set RNG seed"""
    _default_get_state()[0] = s


@compile
def _default_next() -> u64:
    """Generate next random number using LCG algorithm"""
    state_ptr = _default_get_state()
    state_ptr[0] = state_ptr[0] * u64(1103515245) + u64(12345)
    return state_ptr[0]


DefaultRNG = SimpleNamespace(
    seed=_default_seed,
    next=_default_next,
)


# ============================================================
# Set module default
# ============================================================

effect.default(rng=DefaultRNG)


# ============================================================
# Public API - layered functions that all use effect.rng
# ============================================================

@compile
def seed(s: u64) -> void:
    """Set RNG seed"""
    effect.rng.seed(s)


@compile
def random_a() -> u64:
    """Generate random u64 - layer A"""
    return effect.rng.next()


@compile
def random_b() -> u64:
    """Generate random u64 - layer B"""
    return effect.rng.next()


@compile
def get_random_sum() -> u64:
    """Get sum of two random numbers.
    
    This function calls random_a() and random_b(), both of which use effect.rng.
    When imported with effect override, both should use the overridden RNG.
    """
    a = random_a()
    b = random_b()
    return a + b
