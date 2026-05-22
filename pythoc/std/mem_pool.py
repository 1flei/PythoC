"""
Segregated pool allocator for effect.mem.

Provides PoolMem: a malloc/free implementation suitable as the runtime
memory policy (via effect.default(mem=PoolMem) in pythoc.std.runtime).

Global std/mem.py keeps LibcMem as the project-wide default.  Runtime modules
call effect.mem.malloc/free and register PoolMem per module at import time.

Design:
- Small allocations (<= POOL_SMALL_USABLE) use a fixed-size free list.
- Large allocations (<= POOL_LARGE_USABLE) use a second free list (stacks).
- Oversized requests fall through to libc malloc/free.
- Each pooled block has a 16-byte header (magic + size) before user pointer.
"""
from __future__ import annotations

from types import SimpleNamespace

from pythoc import compile, i32, i64, u64, u8, ptr, void, struct, nullptr, sizeof, static
from pythoc.builtin_entities import atomic_cas_i64, atomic_store_i64
from pythoc.libc.stdlib import malloc as libc_malloc, free as libc_free
from pythoc.libc.string import memset

POOL_HEADER_SIZE = u64(16)
POOL_SMALL_USABLE = u64(1024)
POOL_LARGE_USABLE = u64(69632)
POOL_SMALL_BLOCK = POOL_HEADER_SIZE + POOL_SMALL_USABLE
POOL_LARGE_BLOCK = POOL_HEADER_SIZE + POOL_LARGE_USABLE

POOL_MAGIC_SMALL = u64(1)
POOL_MAGIC_LARGE = u64(2)
POOL_MAGIC_LIBC = u64(3)

POOL_SMALL_CACHE_MAX = u64(512)
POOL_LARGE_CACHE_MAX = u64(64)


@compile
class MemPoolState:
    lock: i64
    small_freelist: ptr[void]
    large_freelist: ptr[void]
    small_cached: u64
    large_cached: u64


@compile
def _pool_state() -> ptr[MemPoolState]:
    state: static[ptr[MemPoolState]] = nullptr
    if state == nullptr:
        state = ptr[MemPoolState](libc_malloc(i64(sizeof(MemPoolState))))
        memset(ptr[void](state), 0, i64(sizeof(MemPoolState)))
    return state


@compile
def _pool_raw_alloc(block_size: u64) -> ptr[void]:
    return libc_malloc(i64(block_size))


@compile
def _pool_raw_free(block: ptr[void]) -> void:
    libc_free(block)


@compile
def _pool_write_header(block: ptr[void], magic: u64, size: u64) -> ptr[void]:
    hdr: ptr[u64] = ptr[u64](block)
    hdr[0] = magic
    hdr[1] = size
    return ptr[void](ptr[u8](block) + i64(POOL_HEADER_SIZE))


@compile
def _pool_header(user: ptr[void]) -> ptr[u64]:
    return ptr[u64](ptr[void](ptr[u8](user) - i64(POOL_HEADER_SIZE)))


@compile
def _pool_lock(state: ptr[MemPoolState]) -> void:
    expected: i64 = i64(0)
    while True:
        expected = i64(0)
        if atomic_cas_i64(
            ptr[i64](ptr[void](state)),
            ptr[i64](ptr[void](ptr(expected))),
            i64(1),
        ) != 0:
            return


@compile
def _pool_unlock(state: ptr[MemPoolState]) -> void:
    atomic_store_i64(ptr[i64](ptr[void](state)), i64(0))


@compile
def _pool_malloc_small(state: ptr[MemPoolState], size: u64) -> ptr[void]:
    block: ptr[void] = state.small_freelist
    if block != nullptr:
        state.small_freelist = ptr[ptr[void]](block)[0]
        state.small_cached = state.small_cached - u64(1)
    else:
        block = _pool_raw_alloc(POOL_SMALL_BLOCK)
    return _pool_write_header(block, POOL_MAGIC_SMALL, size)


@compile
def _pool_malloc_large(state: ptr[MemPoolState], size: u64) -> ptr[void]:
    block: ptr[void] = state.large_freelist
    if block != nullptr:
        state.large_freelist = ptr[ptr[void]](block)[0]
        state.large_cached = state.large_cached - u64(1)
    else:
        block = _pool_raw_alloc(POOL_LARGE_BLOCK)
    return _pool_write_header(block, POOL_MAGIC_LARGE, size)


@compile
def _pool_malloc_libc(size: u64) -> ptr[void]:
    block: ptr[void] = _pool_raw_alloc(POOL_HEADER_SIZE + size)
    return _pool_write_header(block, POOL_MAGIC_LIBC, size)


@compile
def _pool_malloc(size: u64) -> ptr[void]:
    if size == u64(0):
        return nullptr
    if size > POOL_LARGE_USABLE:
        return _pool_malloc_libc(size)
    state: ptr[MemPoolState] = _pool_state()
    _pool_lock(state)
    result: ptr[void] = nullptr
    if size <= POOL_SMALL_USABLE:
        result = _pool_malloc_small(state, size)
    else:
        result = _pool_malloc_large(state, size)
    _pool_unlock(state)
    return result


@compile
def _pool_free_small(state: ptr[MemPoolState], block: ptr[void]) -> void:
    if state.small_cached >= POOL_SMALL_CACHE_MAX:
        _pool_raw_free(block)
        return
    ptr[ptr[void]](block)[0] = state.small_freelist
    state.small_freelist = block
    state.small_cached = state.small_cached + u64(1)


@compile
def _pool_free_large(state: ptr[MemPoolState], block: ptr[void]) -> void:
    if state.large_cached >= POOL_LARGE_CACHE_MAX:
        _pool_raw_free(block)
        return
    ptr[ptr[void]](block)[0] = state.large_freelist
    state.large_freelist = block
    state.large_cached = state.large_cached + u64(1)


@compile
def _pool_free(p: ptr[void]) -> void:
    if p == nullptr:
        return
    hdr: ptr[u64] = _pool_header(p)
    magic: u64 = hdr[0]
    block: ptr[void] = ptr[void](hdr)
    if magic == POOL_MAGIC_LIBC:
        _pool_raw_free(block)
        return
    state: ptr[MemPoolState] = _pool_state()
    _pool_lock(state)
    if magic == POOL_MAGIC_SMALL:
        _pool_free_small(state, block)
    elif magic == POOL_MAGIC_LARGE:
        _pool_free_large(state, block)
    else:
        _pool_raw_free(block)
    _pool_unlock(state)


PoolMem = SimpleNamespace(
    malloc=_pool_malloc,
    free=_pool_free,
)
