"""
Test transitive effect propagation across group boundaries.

Scenario:
    root(effect context) -> bridge(no direct effect, separate group)
        -> leaf(effect use, separate group)

The bridge group must be effect-specialized as well. Otherwise it keeps
calling the default leaf group and loses the caller's override.
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

from pythoc import compile, effect, i32, u64, void


@compile
def _default_rng_next() -> u64:
    return u64(11)


@compile
def _default_rng_seed(s: u64) -> void:
    pass


DefaultRNG = SimpleNamespace(next=_default_rng_next, seed=_default_rng_seed)


@compile
def _mock_rng_next() -> u64:
    return u64(99)


@compile
def _mock_rng_seed(s: u64) -> void:
    pass


MockRNG = SimpleNamespace(next=_mock_rng_next, seed=_mock_rng_seed)

effect.default(rng=DefaultRNG)


@compile(suffix="leaf")
def leaf_value() -> u64:
    return effect.rng.next()


@compile(suffix="bridge")
def bridge_value() -> u64:
    return leaf_value()


@compile
def test_default_chain() -> u64:
    return bridge_value()


with effect(rng=MockRNG, suffix="mock"):
    @compile
    def test_override_chain() -> u64:
        return bridge_value()


@compile
def test_both_versions() -> i32:
    default_val = test_default_chain()
    override_val = test_override_chain()
    if default_val == u64(11) and override_val == u64(99):
        return i32(1)
    return i32(0)


if __name__ == "__main__":
    print("=== Cross-group transitive effect tests ===")

    result = test_default_chain()
    print(f"test_default_chain: {result} (expected: 11)")
    assert result == 11, f"Expected 11, got {result}"

    result = test_override_chain()
    print(f"test_override_chain: {result} (expected: 99)")
    assert result == 99, f"Expected 99, got {result}"

    result = test_both_versions()
    print(f"test_both_versions: {result} (expected: 1)")
    assert result == 1, f"Expected 1, got {result}"

    print("=== All tests passed ===")
