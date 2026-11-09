"""Memory providers share resize, zero-size, and failure semantics."""

import os
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pythoc import compile, i32, u8, u64, ptr, void, nullptr
from pythoc.std.mem import DefaultMem
from pythoc.std.mem_pool import PoolMem


def make_checks(provider, name):
    @compile(suffix=(name, 'null_and_zero'))
    def null_and_zero() -> i32:
        empty: ptr[void] = provider.realloc(nullptr, u64(0))
        if empty != nullptr:
            provider.free(empty)
            return i32(0)
        p: ptr[void] = provider.realloc(nullptr, u64(32))
        if p == nullptr:
            return i32(0)
        ptr[u8](p)[0] = u8(91)
        p = provider.realloc(p, u64(0))
        if p != nullptr:
            provider.free(p)
            return i32(0)
        provider.free(nullptr)
        return i32(1)

    @compile(suffix=(name, 'grow_and_shrink'))
    def grow_and_shrink() -> i32:
        p: ptr[void] = provider.malloc(u64(64))
        index: u64 = 0
        while index < u64(64):
            ptr[u8](p)[index] = u8(index + 1)
            index += 1
        grown: ptr[void] = provider.realloc(p, u64(70000))
        if grown == nullptr:
            provider.free(p)
            return i32(0)
        valid: i32 = 1
        index = 0
        while index < u64(64):
            if ptr[u8](grown)[index] != u8(index + 1):
                valid = 0
            index += 1
        ptr[u8](grown)[69999] = u8(137)
        if ptr[u8](grown)[69999] != u8(137):
            valid = 0
        shrunk: ptr[void] = provider.realloc(grown, u64(8))
        if shrunk == nullptr:
            provider.free(grown)
            return i32(0)
        index = 0
        while index < u64(8):
            if ptr[u8](shrunk)[index] != u8(index + 1):
                valid = 0
            index += 1
        provider.free(shrunk)
        return valid

    @compile(suffix=(name, 'failure_retains_old_block'))
    def failure_retains_old_block(requested: u64) -> i32:
        p: ptr[void] = provider.malloc(u64(32))
        ptr[u8](p)[0] = u8(71)
        ptr[u8](p)[31] = u8(93)
        # Impossible sizes exercise upstream failure and the pool's rejection
        # of header-size addition before it can wrap around.
        failed: ptr[void] = provider.realloc(p, requested)
        if failed != nullptr:
            provider.free(failed)
            return i32(0)
        valid: i32 = 1
        if ptr[u8](p)[0] != u8(71) or ptr[u8](p)[31] != u8(93):
            valid = 0
        ptr[u8](p)[0] = u8(11)
        if ptr[u8](p)[0] != u8(11):
            valid = 0
        provider.free(p)
        return valid

    return SimpleNamespace(
        null_and_zero=null_and_zero, grow_and_shrink=grow_and_shrink,
        failure_retains_old_block=failure_retains_old_block,
    )


providers = {
    'default': make_checks(DefaultMem, 'default'),
    'pool': make_checks(PoolMem, 'pool'),
}


class TestMemRealloc(unittest.TestCase):
    def test_null_and_zero_size(self):
        for name, checks in providers.items():
            with self.subTest(provider=name):
                self.assertEqual(checks.null_and_zero(), 1)

    def test_grow_and_shrink_preserve_prefix(self):
        for name, checks in providers.items():
            with self.subTest(provider=name):
                self.assertEqual(checks.grow_and_shrink(), 1)

    def test_failure_keeps_old_pointer_and_contents(self):
        for name, checks in providers.items():
            for requested in (1 << 63, (1 << 64) - 1):
                with self.subTest(provider=name, requested=requested):
                    self.assertEqual(checks.failure_retains_old_block(requested), 1)


if __name__ == '__main__':
    unittest.main()
