"""Size-class pool allocator for effect.mem."""
from __future__ import annotations

from types import SimpleNamespace

from pythoc import (
    compile, i64, u64, u8, ptr, void, struct, nullptr, sizeof,
    array, static, thread_local,
)
from pythoc.builtin_entities import atomic_cas_i64, atomic_store_i64
from pythoc.libc.stdlib import malloc as libc_malloc, free as libc_free
from pythoc.libc.string import memset

POOL_HEADER_SIZE = u64(16)
POOL_CLASS_COUNT = 11
POOL_MAGIC_LIBC = u64(255)

POOL_LOCAL_CACHE_MAX = u64(8192)
POOL_GLOBAL_CACHE_MAX = u64(8192)
POOL_HUGE_CACHE_MAX = u64(512)


@compile
class MemPoolState:
    lock: i64
    freelists: array[ptr[void], POOL_CLASS_COUNT]
    cached: array[u64, POOL_CLASS_COUNT]


@compile
class MemPoolLocalState:
    freelists: array[ptr[void], POOL_CLASS_COUNT]
    cached: array[u64, POOL_CLASS_COUNT]


@compile
def _pool_state() -> ptr[MemPoolState]:
    state: static[ptr[MemPoolState]] = nullptr
    if state == nullptr:
        state = ptr[MemPoolState](libc_malloc(i64(sizeof(MemPoolState))))
        memset(ptr[void](state), 0, i64(sizeof(MemPoolState)))
    return state


@compile
def _pool_local_state() -> ptr[MemPoolLocalState]:
    state: thread_local[ptr[MemPoolLocalState]] = nullptr
    if state == nullptr:
        state = ptr[MemPoolLocalState](libc_malloc(i64(sizeof(MemPoolLocalState))))
        memset(ptr[void](state), 0, i64(sizeof(MemPoolLocalState)))
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
def _pool_class_index(size: u64) -> i64:
    if size <= u64(64):
        return i64(0)
    if size <= u64(128):
        return i64(1)
    if size <= u64(256):
        return i64(2)
    if size <= u64(512):
        return i64(3)
    if size <= u64(1024):
        return i64(4)
    if size <= u64(2048):
        return i64(5)
    if size <= u64(4096):
        return i64(6)
    if size <= u64(8192):
        return i64(7)
    if size <= u64(16384):
        return i64(8)
    if size <= u64(32768):
        return i64(9)
    if size <= u64(65536):
        return i64(10)
    return i64(-1)


@compile
def _pool_class_size(index: i64) -> u64:
    if index == i64(0):
        return u64(64)
    if index == i64(1):
        return u64(128)
    if index == i64(2):
        return u64(256)
    if index == i64(3):
        return u64(512)
    if index == i64(4):
        return u64(1024)
    if index == i64(5):
        return u64(2048)
    if index == i64(6):
        return u64(4096)
    if index == i64(7):
        return u64(8192)
    if index == i64(8):
        return u64(16384)
    if index == i64(9):
        return u64(32768)
    return u64(65536)


@compile
def _pool_cache_max(index: i64) -> u64:
    if index >= i64(10):
        return POOL_HUGE_CACHE_MAX
    return POOL_LOCAL_CACHE_MAX


@compile
def _pool_global_cache_max(index: i64) -> u64:
    if index >= i64(10):
        return POOL_HUGE_CACHE_MAX
    return POOL_GLOBAL_CACHE_MAX


@compile
def _pool_pop_local(local: ptr[MemPoolLocalState], index: i64) -> ptr[void]:
    block: ptr[void] = local.freelists[index]
    if block != nullptr:
        local.freelists[index] = ptr[ptr[void]](block)[0]
        local.cached[index] = local.cached[index] - u64(1)
    return block


@compile
def _pool_pop_global(state: ptr[MemPoolState], index: i64) -> ptr[void]:
    block: ptr[void] = state.freelists[index]
    if block != nullptr:
        state.freelists[index] = ptr[ptr[void]](block)[0]
        state.cached[index] = state.cached[index] - u64(1)
    return block


@compile
def _pool_push_local(local: ptr[MemPoolLocalState], index: i64, block: ptr[void]) -> i64:
    if local.cached[index] >= _pool_cache_max(index):
        return i64(0)
    ptr[ptr[void]](block)[0] = local.freelists[index]
    local.freelists[index] = block
    local.cached[index] = local.cached[index] + u64(1)
    return i64(1)


@compile
def _pool_push_global(state: ptr[MemPoolState], index: i64, block: ptr[void]) -> i64:
    if state.cached[index] >= _pool_global_cache_max(index):
        return i64(0)
    ptr[ptr[void]](block)[0] = state.freelists[index]
    state.freelists[index] = block
    state.cached[index] = state.cached[index] + u64(1)
    return i64(1)


@compile
def _pool_malloc_libc(size: u64) -> ptr[void]:
    block: ptr[void] = _pool_raw_alloc(POOL_HEADER_SIZE + size)
    return _pool_write_header(block, POOL_MAGIC_LIBC, size)


@compile
def _pool_malloc(size: u64) -> ptr[void]:
    if size == u64(0):
        return nullptr
    index: i64 = _pool_class_index(size)
    if index < i64(0):
        return _pool_malloc_libc(size)

    local: ptr[MemPoolLocalState] = _pool_local_state()
    block: ptr[void] = _pool_pop_local(local, index)
    if block != nullptr:
        return _pool_write_header(block, u64(index) + u64(1), size)

    state: ptr[MemPoolState] = _pool_state()
    _pool_lock(state)
    block = _pool_pop_global(state, index)
    _pool_unlock(state)
    if block == nullptr:
        block = _pool_raw_alloc(POOL_HEADER_SIZE + _pool_class_size(index))
    return _pool_write_header(block, u64(index) + u64(1), size)


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
    index: i64 = i64(magic - u64(1))
    if index < i64(0) or index >= i64(POOL_CLASS_COUNT):
        _pool_raw_free(block)
        return

    local: ptr[MemPoolLocalState] = _pool_local_state()
    if _pool_push_local(local, index, block) != i64(0):
        return

    state: ptr[MemPoolState] = _pool_state()
    _pool_lock(state)
    if _pool_push_global(state, index, block) == i64(0):
        _pool_unlock(state)
        _pool_raw_free(block)
        return
    _pool_unlock(state)


PoolMem = SimpleNamespace(
    malloc=_pool_malloc,
    free=_pool_free,
)
