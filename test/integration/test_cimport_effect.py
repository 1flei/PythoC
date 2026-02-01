# -*- coding: utf-8 -*-
"""
Test that bindings module uses effect.mem for memory allocation.

This test verifies that when importing c_parser with a custom mem effect,
all memory allocations go through the custom implementation.

Design:
- Create a counting allocator with function-local static counters
- Import c_parser with the counting allocator via effect override
- Parse some C code to trigger allocations
- Verify that the counting allocator was used (counters > 0)
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pythoc import compile, effect, u64, u8, ptr, void, linear, struct, consume, i64, flush_all_pending_outputs, static
from pythoc.libc.stdlib import malloc as libc_malloc, free as libc_free
from pythoc.std import mem  # Load default mem effect first
from types import SimpleNamespace


# Alias for convenience
flush = flush_all_pending_outputs


# ============================================================================
# Static counters using function-local static variables
# ============================================================================

@compile
def malloc_counter(op: i64) -> u64:
    """Static malloc counter.
    op=0: get current value
    op=1: increment and return new value
    op=-1: reset to 0
    """
    count: static[u64] = u64(0)
    if op == i64(1):
        count = count + u64(1)
    elif op == i64(-1):
        count = u64(0)
    return count


@compile
def free_counter(op: i64) -> u64:
    """Static free counter.
    op=0: get current value
    op=1: increment and return new value
    op=-1: reset to 0
    """
    count: static[u64] = u64(0)
    if op == i64(1):
        count = count + u64(1)
    elif op == i64(-1):
        count = u64(0)
    return count


@compile
def init_counters() -> void:
    """Reset allocation counters to zero"""
    malloc_counter(i64(-1))
    free_counter(i64(-1))


@compile
def get_malloc_count() -> u64:
    """Get current malloc count"""
    return malloc_counter(i64(0))


@compile
def get_free_count() -> u64:
    """Get current free count"""
    return free_counter(i64(0))


# ============================================================================
# Counting allocator implementation
# ============================================================================

@compile
def _counting_malloc(size: u64) -> ptr[u8]:
    """Counting allocator: increments counter and calls libc malloc"""
    malloc_counter(i64(1))
    return ptr[u8](libc_malloc(size))


@compile
def _counting_free(p: ptr[u8]) -> void:
    """Counting free: increments counter and calls libc free"""
    free_counter(i64(1))
    libc_free(p)


@compile
def _counting_lmalloc(size: u64) -> struct[ptr[u8], linear]:
    """Counting linear malloc"""
    malloc_counter(i64(1))
    return ptr[u8](libc_malloc(size)), linear()


@compile
def _counting_lfree(p: ptr[u8], t: linear) -> void:
    """Counting linear free"""
    free_counter(i64(1))
    libc_free(p)
    consume(t)


# Bundle as effect implementation
CountingMem = SimpleNamespace(
    malloc=_counting_malloc,
    free=_counting_free,
    lmalloc=_counting_lmalloc,
    lfree=_counting_lfree,
)


# Flush counting functions first
flush()


# ============================================================================
# Import default version FIRST and flush to compile with default mem effect
# ============================================================================
# IMPORTANT: We must import and flush the default version BEFORE importing
# the counted version, otherwise the counted version's flush will compile
# everything (including default) with the counting allocator.

from pythoc.bindings.c_parser import parse_declarations
from pythoc.bindings.c_ast import decl_free

# Flush default versions before importing counted versions
flush()


# ============================================================================
# Import c_parser with custom mem effect
# ============================================================================

# This should cause all @compile functions in c_parser (and transitively c_ast)
# to use our CountingMem allocator instead of the default libc allocator
with effect(mem=CountingMem, suffix="counted"):
    from pythoc.bindings.c_parser import parse_declarations as parse_declarations_counted
    from pythoc.bindings.c_ast import decl_free as decl_free_counted
    # Force flush inside with block to ensure compilation happens with effect context
    flush()


# ============================================================================
# Test code
# ============================================================================

TEST_C_CODE = "int add(int a, int b);\nvoid print_hello(void);\nstruct Point { int x; int y; };\ntypedef int myint;\n"


# Test function that uses counting allocator
with effect(mem=CountingMem, suffix="counted"):
    @compile
    def test_parse_counted() -> i64:
        """Parse C code using counting allocator.
        
        Returns:
            Number of declarations parsed
        """
        count: i64 = i64(0)
        
        for decl_prf, decl in parse_declarations_counted(TEST_C_CODE):
            count = count + i64(1)
            # Free the declaration
            decl_free_counted(decl_prf, decl)
        
        return count


# Test function that uses default allocator (no effect override)
@compile
def test_parse_default() -> i64:
    """Parse C code with default allocator.
    
    Returns:
        Number of declarations parsed
    """
    count: i64 = i64(0)
    
    for decl_prf, decl in parse_declarations(TEST_C_CODE):
        count = count + i64(1)
        # Free the declaration
        decl_free(decl_prf, decl)
    
    return count


# ============================================================================
# Test class
# ============================================================================

class TestCimportEffect(unittest.TestCase):
    """Test that c_parser uses effect.mem for memory allocation"""

    def test_counting_allocator_is_used(self):
        """Verify that imports with effect override use the counting allocator"""
        # Reset counters
        init_counters()
        
        # Verify counters are zero
        self.assertEqual(get_malloc_count(), 0, "malloc_count should start at 0")
        self.assertEqual(get_free_count(), 0, "free_count should start at 0")
        
        # Parse using counted version
        decl_count = test_parse_counted()
        
        # Should have parsed 4 declarations
        self.assertEqual(decl_count, 4, f"Expected 4 declarations, got {decl_count}")
        
        # Should have made allocations (malloc_count > 0)
        malloc_count = get_malloc_count()
        free_count = get_free_count()
        
        self.assertGreater(malloc_count, 0, 
            f"Expected malloc_count > 0, got {malloc_count}. "
            "This indicates that c_parser is not properly using effect.mem.")
        
        self.assertGreater(free_count, 0,
            f"Expected free_count > 0, got {free_count}. "
            "This indicates that decl_free is not properly using effect.mem.")
        
        print(f"OK: Counted allocator was used - malloc={malloc_count}, free={free_count}")

    def test_default_allocator_does_not_count(self):
        """Verify that default allocator does not affect our counters"""
        # Reset counters
        init_counters()
        
        # Parse using default version
        decl_count = test_parse_default()
        
        # Should have parsed 4 declarations
        self.assertEqual(decl_count, 4, f"Expected 4 declarations, got {decl_count}")
        
        # Counters should remain at 0 (default allocator doesn't increment them)
        malloc_count = get_malloc_count()
        free_count = get_free_count()
        
        self.assertEqual(malloc_count, 0,
            f"Expected malloc_count = 0 for default allocator, got {malloc_count}")
        self.assertEqual(free_count, 0,
            f"Expected free_count = 0 for default allocator, got {free_count}")
        
        print("OK: Default allocator does not affect our counters (as expected)")


if __name__ == '__main__':
    print("=== Testing effect.mem with c_parser ===")
    print()
    unittest.main(verbosity=2)
