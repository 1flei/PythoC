# -*- coding: utf-8 -*-
"""
Simple test to verify effect propagation works correctly.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pythoc import compile, effect, u64, u8, ptr, void, linear, struct, consume
from pythoc.libc.stdlib import malloc as libc_malloc, free as libc_free
from types import SimpleNamespace

MAGIC = 0xDEADBEEF

# Marker allocator
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

@compile
def _marker_lmalloc(size: u64) -> struct[ptr[void], linear]:
    return _marker_malloc(size), linear()

@compile
def _marker_lfree(p: ptr[void], t: linear) -> void:
    _marker_free(p)
    consume(t)

MarkerMem = SimpleNamespace(
    malloc=_marker_malloc,
    free=_marker_free,
    lmalloc=_marker_lmalloc,
    lfree=_marker_lfree,
)

# Test with effect override
print("Importing with effect override...")
with effect(mem=MarkerMem, suffix="tracked"):
    from pythoc.bindings.c_ast import ctype_alloc as ctype_alloc_tracked, ctype_free as ctype_free_tracked
    print(f"ctype_alloc_tracked._mangled_name: {getattr(ctype_alloc_tracked, '_mangled_name', 'N/A')}")
    print(f"ctype_alloc_tracked._effect_suffix: {getattr(ctype_alloc_tracked, '_effect_suffix', 'N/A')}")

print("Import complete.")

# Also import default version
from pythoc.bindings.c_ast import ctype_alloc, ctype_free

@compile
def check_marker(p: ptr[u8]) -> u64:
    """Check if memory has our magic marker."""
    if p == ptr[u8](0):
        return u64(0)
    marker_p = ptr[u64](p) - 1
    return marker_p[0]

# Test tracked version
with effect(mem=MarkerMem, suffix="tracked"):
    @compile
    def test_tracked() -> u64:
        """Test tracked allocator"""
        prf, ty = ctype_alloc_tracked()
        marker = check_marker(ptr[u8](ty))
        ctype_free_tracked(prf, ty)
        return marker

# Test default version
@compile
def test_default() -> u64:
    """Test default allocator"""
    prf, ty = ctype_alloc()
    marker = check_marker(ptr[u8](ty))
    ctype_free(prf, ty)
    return marker


if __name__ == '__main__':
    print("Testing tracked allocator...")
    result = test_tracked()
    print(f"Tracked result: {hex(result)}")
    print(f"Expected: {hex(MAGIC)}")
    if result == MAGIC:
        print("PASS: Tracked allocator was used!")
    else:
        print("FAIL: Tracked allocator was NOT used")
    
    print()
    print("Testing default allocator...")
    result = test_default()
    print(f"Default result: {hex(result)}")
    if result == MAGIC:
        print("WARNING: Default allocator coincidentally has marker")
    else:
        print("OK: Default allocator does not have marker (as expected)")
