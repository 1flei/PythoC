# -*- coding: utf-8 -*-
"""
Simple test for effect.mem propagation without linear_wrap complexity.

This test verifies that effect propagation works for direct function calls.

Design:
- Counters are in a separate module to avoid being affected by effect propagation
- The counting allocator uses the counters but doesn't use effect.mem itself
- Test functions use effect.mem.malloc/free which get overridden
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pythoc import compile, effect, u64, u8, ptr, void, flush_all_pending_outputs
from pythoc.libc.stdlib import malloc as libc_malloc, free as libc_free
from pythoc.std import mem
from types import SimpleNamespace

# Import counters from separate module (won't be affected by effect propagation)
from counters_module import (
    malloc_counter, free_counter, 
    init_counters, get_malloc_count, get_free_count
)

flush = flush_all_pending_outputs

# Flush counter module first
flush()


# Counting allocator - directly calls libc, doesn't use effect
@compile
def _counting_malloc(size: u64) -> ptr[u8]:
    malloc_counter(1)
    return ptr[u8](libc_malloc(size))


@compile
def _counting_free(p: ptr[u8]) -> void:
    free_counter(1)
    libc_free(p)


CountingMem = SimpleNamespace(
    malloc=_counting_malloc,
    free=_counting_free,
)

# Flush counting allocator
flush()


# Functions that use effect.mem
@compile
def allocate_and_free_memory(size: u64) -> void:
    """Directly use effect.mem to allocate and free memory"""
    p: ptr[u8] = ptr[u8](effect.mem.malloc(size))
    effect.mem.free(p)


@compile
def allocate_memory_only(size: u64) -> ptr[u8]:
    """Allocate memory using effect.mem"""
    return ptr[u8](effect.mem.malloc(size))


@compile
def free_memory_only(p: ptr[u8]) -> void:
    """Free memory using effect.mem"""
    effect.mem.free(p)


# Flush default versions
flush()


# Create counted versions
with effect(mem=CountingMem, suffix="counted"):
    @compile
    def allocate_and_free_counted(size: u64) -> void:
        """Use counted allocator"""
        p: ptr[u8] = ptr[u8](effect.mem.malloc(size))
        effect.mem.free(p)
    
    @compile
    def allocate_counted(size: u64) -> ptr[u8]:
        """Allocate with counted allocator"""
        return ptr[u8](effect.mem.malloc(size))
    
    @compile
    def free_counted(p: ptr[u8]) -> void:
        """Free with counted allocator"""
        effect.mem.free(p)
    
    flush()


class TestSimpleEffectMem(unittest.TestCase):
    """Test effect.mem propagation with direct calls"""

    def test_counting_allocator_is_used(self):
        """Verify counted allocator increments counters"""
        init_counters()
        
        # Verify counters start at 0
        self.assertEqual(get_malloc_count(), 0)
        self.assertEqual(get_free_count(), 0)
        
        # Use counted allocator
        allocate_and_free_counted(100)
        
        # Should have 1 malloc and 1 free
        malloc_count = get_malloc_count()
        free_count = get_free_count()
        
        self.assertGreater(malloc_count, 0, 
            f"Expected malloc_count > 0, got {malloc_count}")
        self.assertGreater(free_count, 0,
            f"Expected free_count > 0, got {free_count}")
        
        print(f"OK: Counted allocator was used - malloc={malloc_count}, free={free_count}")

    def test_default_allocator_does_not_count(self):
        """Verify default allocator doesn't affect counters"""
        init_counters()
        
        # Use default allocator
        allocate_and_free_memory(100)
        
        # Counters should remain 0
        self.assertEqual(get_malloc_count(), 0)
        self.assertEqual(get_free_count(), 0)
        
        print("OK: Default allocator does not affect counters")

    def test_multiple_allocations(self):
        """Test multiple allocations are counted correctly"""
        init_counters()
        
        # Allocate 5 times
        ptrs = []
        for _ in range(5):
            p = allocate_counted(64)
            ptrs.append(p)
        
        self.assertEqual(get_malloc_count(), 5)
        
        # Free all
        for p in ptrs:
            free_counted(p)
        
        self.assertEqual(get_free_count(), 5)
        
        print("OK: Multiple allocations counted correctly")


if __name__ == '__main__':
    print("=== Testing effect.mem with simple direct calls ===")
    print()
    unittest.main(verbosity=2)
