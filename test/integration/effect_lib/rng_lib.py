# -*- coding: utf-8 -*-
"""
RNG library that uses effects.

This module demonstrates the effect system:
- Uses effect.rng.next() for random number generation
- Sets a default RNG implementation via effect.default()
- Callers can override with their own RNG via `with effect(rng=..., suffix=...)`
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
    # LCG: state = state * 1103515245 + 12345
    state_ptr[0] = state_ptr[0] * u64(1103515245) + u64(12345)
    return state_ptr[0]


# Bundle as effect object
DefaultRNG = SimpleNamespace(
    seed=_default_seed,
    next=_default_next,
)


# ============================================================
# Set module default
# ============================================================

effect.default(rng=DefaultRNG)


# ============================================================
# Public API - uses effect.rng
# ============================================================

@compile
def seed(s: u64) -> void:
    """Set RNG seed"""
    effect.rng.seed(s)


@compile
def random() -> u64:
    """Generate random u64 using current effect.rng"""
    return effect.rng.next()
