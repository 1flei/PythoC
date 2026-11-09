"""Execute std containers with a non-libc pointer and a tracked allocator."""

import os
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pythoc import compile, effect, i32, i64, u8, u64, ptr, void, static, nullptr
from pythoc.libc.string import memcpy
from pythoc.std.mem import DefaultMem
from pythoc.std.mem_pool import PoolMem
from effect_lib.std_mem_clients import make_clients


@compile
class AllocationCounts:
    mallocs: static[i64] = 0
    reallocs: static[i64] = 0
    frees: static[i64] = 0
    live: static[i64] = 0
    errors: static[i64] = 0


@compile
def tracked_malloc(size: u64) -> ptr[void]:
    raw = ptr[u64](DefaultMem.malloc(size + u64(16)))
    raw[0] = u64(0x5043545241434B)
    raw[1] = size
    AllocationCounts.mallocs += 1
    AllocationCounts.live += 1
    return ptr[void](ptr[u8](raw) + 16)


@compile
def tracked_free(p: ptr[void]) -> void:
    if p == nullptr:
        return
    raw = ptr[u64](ptr[u8](p) - 16)
    if raw[0] != u64(0x5043545241434B):
        AllocationCounts.errors += 1
        return
    raw[0] = u64(0)
    AllocationCounts.frees += 1
    AllocationCounts.live -= 1
    DefaultMem.free(ptr[void](raw))


@compile
def tracked_realloc(p: ptr[void], size: u64) -> ptr[void]:
    AllocationCounts.reallocs += 1
    if p == nullptr:
        return tracked_malloc(size)
    if size == u64(0):
        tracked_free(p)
        return nullptr
    raw = ptr[u64](ptr[u8](p) - 16)
    old_size: u64 = raw[1]
    copied: u64 = size
    if old_size < copied:
        copied = old_size
    new_p = tracked_malloc(size)
    memcpy(new_p, p, copied)
    tracked_free(p)
    return new_p


@compile
def reset_counts() -> void:
    AllocationCounts.mallocs = 0
    AllocationCounts.reallocs = 0
    AllocationCounts.frees = 0
    AllocationCounts.live = 0
    AllocationCounts.errors = 0


@compile
def malloc_count() -> i64:
    return AllocationCounts.mallocs


@compile
def realloc_count() -> i64:
    return AllocationCounts.reallocs


@compile
def free_count() -> i64:
    return AllocationCounts.frees


@compile
def live_count() -> i64:
    return AllocationCounts.live


@compile
def error_count() -> i64:
    return AllocationCounts.errors


TrackedMem = SimpleNamespace(
    malloc=tracked_malloc, realloc=tracked_realloc, free=tracked_free,
)
default_clients = make_clients()
with effect(mem=TrackedMem, suffix='std_tracked_mem'):
    tracked_clients = make_clients()
with effect(mem=PoolMem, suffix='std_pooled_mem'):
    pooled_clients = make_clients()


class TestStdMemEffect(unittest.TestCase):
    def setUp(self):
        reset_counts()

    def assert_balanced(self, mallocs, reallocs=0):
        self.assertEqual(malloc_count(), mallocs)
        self.assertEqual(realloc_count(), reallocs)
        self.assertEqual(free_count(), mallocs)
        self.assertEqual(live_count(), 0)
        self.assertEqual(error_count(), 0)

    def test_inline_vector_needs_no_allocation(self):
        self.assertEqual(tracked_clients.small_vector(4), sum(3 * i + 1 for i in range(4)))
        self.assert_balanced(0)

    def test_vector_spill_and_repeated_realloc(self):
        self.assertEqual(tracked_clients.small_vector(65), sum(3 * i + 1 for i in range(65)))
        self.assert_balanced(5, 4)

    def test_heap_vector_uses_matching_provider(self):
        self.assertEqual(tracked_clients.heap_vector(65), sum(3 * i + 1 for i in range(65)))
        self.assert_balanced(6, 5)

    def test_set_rehash_and_destroy_use_matching_provider(self):
        self.assertEqual(tracked_clients.hash_set(128), sum(range(128)))
        self.assert_balanced(10)

    def test_map_rehash_and_destroy_use_matching_provider(self):
        self.assertEqual(tracked_clients.hash_map(128), sum(3 * i + 1 for i in range(128)))
        self.assert_balanced(15)

    def test_default_and_pool_bindings_coexist_with_tracked_binding(self):
        for clients in (default_clients, pooled_clients):
            self.assertEqual(clients.small_vector(65), sum(3 * i + 1 for i in range(65)))
            self.assertEqual(clients.heap_vector(65), sum(3 * i + 1 for i in range(65)))
            self.assertEqual(clients.hash_set(128), sum(range(128)))
            self.assertEqual(clients.hash_map(128), sum(3 * i + 1 for i in range(128)))
        self.assert_balanced(0)


if __name__ == '__main__':
    unittest.main()
