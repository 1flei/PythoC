# -*- coding: utf-8 -*-
"""
Custom RNG implementations for testing effect overrides.

These are alternative RNG implementations that can be used
to override the default RNG in rng_lib.
"""

from pythoc import compile, u64, void, ptr, static
from types import SimpleNamespace


# ============================================================
# Mock RNG that always returns a fixed value (for testing)
# ============================================================

@compile
def _mock_seed(s: u64) -> void:
    """Mock seed - does nothing"""
    pass


@compile
def _mock_next() -> u64:
    """Mock next - always returns 999"""
    return u64(999)


MockRNG = SimpleNamespace(
    seed=_mock_seed,
    next=_mock_next,
)


# ============================================================
# Counter RNG that returns incrementing values
# ============================================================

@compile
def _counter_get_state() -> ptr[u64]:
    """Return pointer to counter state"""
    state: static[u64] = 0
    return ptr(state)


@compile
def _counter_seed(s: u64) -> void:
    """Set counter to seed value"""
    _counter_get_state()[0] = s


@compile
def _counter_next() -> u64:
    """Return current counter and increment"""
    state_ptr = _counter_get_state()
    result = state_ptr[0]
    state_ptr[0] = state_ptr[0] + u64(1)
    return result


CounterRNG = SimpleNamespace(
    seed=_counter_seed,
    next=_counter_next,
)
