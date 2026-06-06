# -*- coding: utf-8 -*-
"""
Simple test to verify effect propagation works correctly.

Tests the import-override pattern: importing @compile functions from a module
under ``with effect(...)`` recompiles them with the overridden effect.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pythoc import compile, effect, u64, u8, ptr, void
from pythoc.libc.stdlib import malloc as libc_malloc, free as libc_free
from pythoc.std import mem
from types import SimpleNamespace

MAGIC = 0xDEADBEEF

# =============================================================================
# Marker allocator (override implementation)
# =============================================================================

@compile
def _marker_malloc(size: u64) -> ptr[void]:
    raw_p = ptr[u8](libc_malloc(size + u64(8)))
    marker_p = ptr[u64](raw_p)
    marker_p[0] = u64(MAGIC)
    return ptr[u8](ptr[u64](raw_p) + 1)

@compile
def _marker_free(p: ptr[void]) -> void:
    raw_p = ptr[u8](ptr[u64](p) - 1)
    libc_free(raw_p)

MarkerMem = SimpleNamespace(
    malloc=_marker_malloc,
    free=_marker_free,
)

# =============================================================================
# Import with effect override (tracked version)
# =============================================================================

print("Importing with effect override...")
with effect(mem=MarkerMem, suffix="tracked"):
    from effect_prop_targets import target_alloc as target_alloc_tracked, target_free as target_free_tracked
    print(f"target_alloc_tracked: effect_suffix={getattr(target_alloc_tracked, '_func_info', None) and target_alloc_tracked._func_info.binding_state.effect_suffix}")

print("Import complete.")

# =============================================================================
# Import default version
# =============================================================================

from effect_prop_targets import target_alloc, target_free

# =============================================================================
# Check helper
# =============================================================================

@compile
def check_marker(p: ptr[u8]) -> u64:
    """Check if memory has our magic marker."""
    if p == ptr[u8](0):
        return u64(0)
    marker_p = ptr[u64](p) - 1
    return marker_p[0]

# =============================================================================
# Test functions
# =============================================================================

@compile
def test_tracked() -> u64:
    """Test tracked allocator - should have MAGIC marker"""
    p = ptr[u8](target_alloc_tracked(u64(64)))
    marker = check_marker(p)
    target_free_tracked(ptr[void](p))
    return marker

@compile
def test_default() -> u64:
    """Test default allocator - should NOT have MAGIC marker"""
    p = ptr[u8](target_alloc(u64(64)))
    marker = check_marker(p)
    target_free(ptr[void](p))
    return marker


if __name__ == '__main__':
    ok = True

    print("Testing tracked allocator...")
    result = test_tracked()
    print(f"Tracked result: {hex(result)}")
    print(f"Expected: {hex(MAGIC)}")
    if result == MAGIC:
        print("PASS: Tracked allocator was used!")
    else:
        print("FAIL: Tracked allocator was NOT used")
        ok = False
    
    print()
    print("Testing default allocator...")
    result = test_default()
    print(f"Default result: {hex(result)}")
    if result == MAGIC:
        print("FAIL: Default allocator should NOT have marker")
        ok = False
    else:
        print("PASS: Default allocator does not have marker (as expected)")

    sys.exit(0 if ok else 1)
