"""
Standard Memory Effect Library

Provides effect-based memory allocation with default libc implementation.

Effect API (via effect.mem):
- effect.mem.alloc(size: u64) -> ptr[u8]                        : Allocate heap memory
- effect.mem.free(p: ptr[u8]) -> void                           : Free heap memory
- effect.mem.lalloc(size: u64) -> struct[ptr[u8], MemProof]     : Allocate with linear token
- effect.mem.lfree(p: ptr[u8], t: MemProof) -> void             : Free and consume linear token

Types:
- MemProof: refined[linear, "MemProof"] - linear token for memory tracking

Design:
    lalloc/lfree use MemProof (a refined linear type) for compile-time resource tracking.
    Using a refined type instead of raw linear allows distinguishing memory tokens
    from other linear resources (e.g., file handles).
    The token must be consumed exactly once via lfree, ensuring memory is freed.
    Forgetting to call lfree is a compile-time error.

Usage:
    # Use memory allocation via effect
    from pythoc import compile, effect
    from pythoc.std import mem  # Sets up default mem effect

    @compile
    def create_buffer(n: u64) -> ptr[u8]:
        return effect.mem.alloc(n)

    @compile
    def safe_alloc(n: u64) -> struct[ptr[u8], mem.MemProof]:
        return effect.mem.lalloc(n)

    @compile
    def safe_free(p: ptr[u8], t: mem.MemProof) -> void:
        effect.mem.lfree(p, t)
"""

from types import SimpleNamespace

from pythoc import compile, effect, u64, u8, ptr, void, linear, struct, consume, refined, assume
from pythoc.libc.stdlib import malloc, free


# ============================================================
# MemProof - refined linear type for memory tracking
# ============================================================

MemProof = refined[linear, "MemProof"]


# ============================================================
# Default implementation using libc malloc/free
# ============================================================

@compile
def _mem_alloc(size: u64) -> ptr[u8]:
    """Default heap allocator using libc malloc"""
    return ptr[u8](malloc(size))


@compile
def _mem_free(p: ptr[u8]) -> void:
    """Default heap deallocator using libc free"""
    free(p)


@compile
def _mem_lalloc(size: u64) -> struct[ptr[u8], MemProof]:
    """Default linear allocator - returns ptr + MemProof for tracking"""
    return ptr[u8](malloc(size)), assume(linear(), "MemProof")


@compile
def _mem_lfree(p: ptr[u8], t: MemProof) -> void:
    """Default linear deallocator - frees memory and consumes MemProof"""
    free(p)
    consume(t)


# Bundle as effect implementation
# The effect API uses alloc/free names, but implementation uses _mem_* functions
DefaultMem = SimpleNamespace(
    alloc=_mem_alloc,
    free=_mem_free,
    lalloc=_mem_lalloc,
    lfree=_mem_lfree,
)


# ============================================================
# Set module default (overridable by callers)
# ============================================================

effect.default(mem=DefaultMem)
