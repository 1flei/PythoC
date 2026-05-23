#!/usr/bin/env python3
"""
Test runtime memory policy: PoolMem via effect.mem.

Verifies:
- PoolMem roundtrip through effect.mem override
- Runtime modules register PoolMem per-module at import
- Runtime spawn/join still works with pooled allocations
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc import compile, effect, i32, i64, u64, ptr, void, nullptr
from pythoc.build.output_manager import flush_all_pending_outputs
from pythoc.std.mem_pool import PoolMem

from test.utils.test_utils import DeferredTestCase

from pythoc.std.runtime.api import (
    runtime_new, runtime_start, runtime_spawn, runtime_join,
    runtime_shutdown, runtime_free,
)
from pythoc.std.runtime.task import TaskHandle


flush_all_pending_outputs()


with effect(mem=PoolMem, suffix="pool"):
    @compile
    def pool_malloc_free_small() -> i32:
        p: ptr[void] = effect.mem.malloc(u64(128))
        if p == nullptr:
            return i32(0)
        effect.mem.free(p)
        return i32(1)

    @compile
    def pool_malloc_free_large() -> i32:
        p: ptr[void] = effect.mem.malloc(u64(4096))
        if p == nullptr:
            return i32(0)
        effect.mem.free(p)
        return i32(1)

    flush_all_pending_outputs()


@compile(suffix="rt_pool_entry")
def test_fn_runtime_pool_entry(arg: ptr[void]) -> ptr[void]:
    out: ptr[i64] = ptr[i64](arg)
    out[0] = i64(9001)
    return nullptr


@compile(suffix="rt_pool_spawn")
def test_fn_runtime_pool_spawn() -> i64:
    rt = runtime_new(i32(2))
    runtime_start(rt)
    value: i64 = 0
    task: TaskHandle = runtime_spawn(
        rt, test_fn_runtime_pool_entry, ptr[void](ptr(value)), u64(0)
    )
    runtime_join(rt, task)
    runtime_shutdown(rt)
    runtime_free(rt)
    return value


flush_all_pending_outputs()


class TestRuntimeMemPool(DeferredTestCase):
    """Runtime PoolMem policy tests."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        flush_all_pending_outputs()

    def test_runtime_modules_bind_pool_mem(self):
        from pythoc import effect as effect_mod
        import pythoc.std.runtime.task as task_mod
        import pythoc.std.runtime.scheduler as sched_mod
        import pythoc.std.runtime.platform as plat_mod

        for mod in (task_mod, sched_mod, plat_mod):
            name = mod.__name__
            self.assertIn('mem', effect_mod._defaults[name])
            self.assertIs(effect_mod._defaults[name]['mem'], PoolMem)

    def test_pool_malloc_free_small(self):
        self.assertEqual(pool_malloc_free_small(), 1)

    def test_pool_malloc_free_large(self):
        self.assertEqual(pool_malloc_free_large(), 1)

    def test_runtime_spawn_join_with_pool(self):
        self.assertEqual(test_fn_runtime_pool_spawn(), 9001)


if __name__ == '__main__':
    unittest.main()
