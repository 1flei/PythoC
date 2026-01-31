# -*- coding: utf-8 -*-
"""
Integration test for pythoc.std.mem effect-based memory library.

Tests:
1. Default libc implementation works (effect.mem.alloc/free)
2. Linear allocation tracking works (effect.mem.lalloc/lfree)
3. User can override with custom allocator via effect
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pythoc import compile, effect, u64, u8, ptr, void, linear, struct, consume, refined, assume
from pythoc.std import mem  # Sets up default mem effect
from pythoc.std.mem import MemProof  # Import MemProof type (refined linear for memory tracking)
from pythoc.libc.stdlib import malloc as libc_malloc, free as libc_free
from types import SimpleNamespace


# ============================================================================
# Test functions - all @compile must be defined before execution
# ============================================================================

@compile
def test_alloc_free_fn() -> u64:
    """Test basic alloc/free with default implementation"""
    p = effect.mem.alloc(u64(64))
    p[0] = u8(42)
    result = u64(p[0])
    effect.mem.free(p)
    return result


@compile
def test_array_alloc_fn() -> u64:
    """Test array allocation helper"""
    p = effect.mem.alloc(u64(10) * u64(8))
    p_u64 = ptr[u64](p)
    p_u64[0] = u64(100)
    p_u64[9] = u64(200)
    result = p_u64[0] + p_u64[9]
    effect.mem.free(p)
    return result


@compile
def test_lalloc_lfree_fn() -> u64:
    """Test lalloc/lfree with linear token tracking"""
    p, t = effect.mem.lalloc(u64(32))
    p[0] = u8(77)
    result = u64(p[0])
    effect.mem.lfree(p, t)
    return result


@compile
def test_lalloc_array_fn() -> u64:
    """Test linear array allocation"""
    p, t = effect.mem.lalloc(u64(5) * u64(8))
    p_u64 = ptr[u64](p)
    p_u64[0] = u64(10)
    p_u64[4] = u64(20)
    result = p_u64[0] + p_u64[4]
    effect.mem.lfree(p, t)
    return result


@compile
def lib_create_buffer(size: u64) -> ptr[u8]:
    """Library function using effect.mem"""
    return effect.mem.alloc(size)


@compile
def lib_destroy_buffer(p: ptr[u8]) -> void:
    """Library function using effect.mem"""
    effect.mem.free(p)


@compile
def test_lib_usage_fn() -> u64:
    """Test library usage"""
    buf = lib_create_buffer(u64(16))
    buf[0] = u8(88)
    result = u64(buf[0])
    lib_destroy_buffer(buf)
    return result


@compile
def lib_create_tracked(size: u64) -> struct[ptr[u8], MemProof]:
    """Library with linear-tracked memory"""
    return effect.mem.lalloc(size)


@compile
def lib_destroy_tracked(p: ptr[u8], t: MemProof) -> void:
    """Library linear free"""
    effect.mem.lfree(p, t)


@compile
def test_lib_linear_usage_fn() -> u64:
    """Test library linear usage"""
    buf, token = lib_create_tracked(u64(16))
    buf[0] = u8(55)
    result = u64(buf[0])
    lib_destroy_tracked(buf, token)
    return result


# ============================================================================
# Custom allocator for override test
# ============================================================================

@compile
def _custom_alloc(size: u64) -> ptr[u8]:
    """Custom allocator: writes magic number 0xAB at start to prove it was used"""
    p = ptr[u8](libc_malloc(size))
    p[0] = u8(0xAB)  # Write magic marker
    return p


@compile
def _custom_free(p: ptr[u8]) -> void:
    """Custom free (just wraps libc)"""
    libc_free(p)


@compile
def _custom_lalloc(size: u64) -> struct[ptr[u8], linear]:
    """Custom linear alloc"""
    return ptr[u8](libc_malloc(size)), linear()


@compile
def _custom_lfree(p: ptr[u8], t: linear) -> void:
    """Custom linear free"""
    libc_free(p)
    consume(t)


CustomMem = SimpleNamespace(
    alloc=_custom_alloc,
    free=_custom_free,
    lalloc=_custom_lalloc,
    lfree=_custom_lfree,
)


# Define test function that uses custom allocator via effect
# Effect override is done at Python level (module import time), not inside @compile
with effect(mem=CustomMem, suffix="custom"):
    @compile
    def use_custom_allocator() -> u64:
        """Use custom allocator via effect override"""
        p = effect.mem.alloc(u64(32))
        # Custom allocator writes 0xAB at start, so p[0] should be 0xAB (171)
        # If default allocator was used, p[0] would be uninitialized (not 0xAB)
        result = u64(p[0])  # Should be 171 (0xAB) if custom was used
        effect.mem.free(p)
        return result


# ============================================================================
# Test runner
# ============================================================================

def test_alloc_and_free():
    """Test basic alloc/free with default implementation"""
    result = test_alloc_free_fn()
    assert result == 42, f"Expected 42, got {result}"
    print("OK test_alloc_and_free")


def test_alloc_array():
    """Test array allocation helper"""
    result = test_array_alloc_fn()
    assert result == 300, f"Expected 300, got {result}"
    print("OK test_alloc_array")


def test_lalloc_and_lfree():
    """Test lalloc/lfree with linear token tracking"""
    result = test_lalloc_lfree_fn()
    assert result == 77, f"Expected 77, got {result}"
    print("OK test_lalloc_and_lfree")


def test_lalloc_array():
    """Test linear array allocation"""
    result = test_lalloc_array_fn()
    assert result == 30, f"Expected 30, got {result}"
    print("OK test_lalloc_array")


def test_library_uses_default():
    """Library author uses effect.mem and gets default implementation"""
    result = test_lib_usage_fn()
    assert result == 88, f"Expected 88, got {result}"
    print("OK test_library_uses_default")


def test_library_uses_linear():
    """Library with linear-tracked memory"""
    result = test_lib_linear_usage_fn()
    assert result == 55, f"Expected 55, got {result}"
    print("OK test_library_uses_linear")


def test_effect_override():
    """Test overriding mem effect with custom implementation"""
    result = use_custom_allocator()
    # Custom allocator writes 0xAB (171) at start of allocated memory
    # This proves the custom allocator was actually used, not the default
    assert result == 171, f"Expected 171 (0xAB from custom allocator), got {result}"
    print("OK test_effect_override")


if __name__ == '__main__':
    print("=== std.mem default tests ===")
    test_alloc_and_free()
    test_alloc_array()
    print()

    print("=== std.mem linear tests ===")
    test_lalloc_and_lfree()
    test_lalloc_array()
    print()

    print("=== std.mem library usage tests ===")
    test_library_uses_default()
    test_library_uses_linear()
    print()

    print("=== std.mem effect override tests ===")
    test_effect_override()
    print()

    print("All std.mem tests passed!")
