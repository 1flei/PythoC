"""
Standard Memory Effect Library

Provides effect-based memory allocation with default libc implementation.

Effect API (via effect.mem):
- effect.mem.malloc(size: u64) -> ptr[void]                        : Allocate heap memory
- effect.mem.free(p: ptr[void]) -> void                            : Free heap memory
- effect.mem.lmalloc(size: u64) -> struct[ptr[void], MemProof]     : Allocate with linear token
- effect.mem.lfree(p: ptr[void], t: MemProof) -> void              : Free and consume linear token

Types:
- MemProof: refined[linear, "MemProof"] - linear token for memory tracking

Design:
    lmalloc/lfree use MemProof (a refined linear type) for compile-time resource tracking.
    Using a refined type instead of raw linear allows distinguishing memory tokens
    from other linear resources (e.g., file handles).
    The token must be consumed exactly once via lfree, ensuring memory is freed.
    Forgetting to call lfree is a compile-time error.

Usage:
    # Use memory allocation via effect
    from pythoc import compile, effect
    from pythoc.std import mem  # Sets up default mem effect

    @compile
    def create_buffer(n: u64) -> ptr[void]:
        return effect.mem.malloc(n)

    @compile
    def safe_alloc(n: u64) -> struct[ptr[void], mem.MemProof]:
        return effect.mem.lmalloc(n)

    @compile
    def safe_free(p: ptr[void], t: mem.MemProof) -> void:
        effect.mem.lfree(p, t)
"""

from types import SimpleNamespace

from pythoc import compile, effect, u64, ptr, void, linear, struct, consume, refined, assume
from pythoc.libc.stdlib import malloc as libc_malloc, free as libc_free


# ============================================================
# MemProof - refined linear type for memory tracking
# ============================================================

MemProof = refined[linear, "MemProof"]


# ============================================================
# Default implementation using libc malloc/free
# ============================================================

@compile
def _mem_malloc(size: u64) -> ptr[void]:
    """Default heap allocator using libc malloc"""
    return libc_malloc(size)


@compile
def _mem_free(p: ptr[void]) -> void:
    """Default heap deallocator using libc free"""
    libc_free(p)


@compile
def _mem_lmalloc(size: u64) -> struct[ptr[void], MemProof]:
    """Default linear allocator - returns ptr + MemProof for tracking"""
    return libc_malloc(size), assume(linear(), "MemProof")


@compile
def _mem_lfree(p: ptr[void], t: MemProof) -> void:
    """Default linear deallocator - frees memory and consumes MemProof"""
    libc_free(p)
    consume(t)


# Bundle as effect implementation
DefaultMem = SimpleNamespace(
    malloc=_mem_malloc,
    free=_mem_free,
    lmalloc=_mem_lmalloc,
    lfree=_mem_lfree,
)


# ============================================================
# Set module default (overridable by callers)
# ============================================================

effect.default(mem=DefaultMem)
