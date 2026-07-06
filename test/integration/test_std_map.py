#!/usr/bin/env python3
"""Integration tests for pythoc.std.map.FlatHashMap.

Coverage:
- basic insert/find/contains/erase across supported key types
- large-scale operations (10k+ key/value pairs)
- collision-heavy key patterns
- load-factor boundary and repeated rehash
- value mutation via returned pointer
- repeated updates of the same key
- tombstone reuse and clear-then-reuse
- deterministic stress test with mixed operations
- failure cases (unsupported key type, bad factory args)
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pythoc import (
    i8, i16, i32, i64, u8, u16, u32, u64,
    bool, ptr, compile, nullptr, array,
)
from pythoc.std.map import FlatHashMap


I8Map = FlatHashMap(i8, i64)
I16Map = FlatHashMap(i16, i64)
I32Map = FlatHashMap(i32, i64)
I64Map = FlatHashMap(i64, i64)
U8Map = FlatHashMap(u8, i64)
U16Map = FlatHashMap(u16, i64)
U32Map = FlatHashMap(u32, i64)
U64Map = FlatHashMap(u64, i64)

ALL_MAPS = [
    I8Map, I16Map, I32Map, I64Map,
    U8Map, U16Map, U32Map, U64Map,
]

STRESS_MAPS = [I32Map, I64Map, U64Map]


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def _make_basic_tester(api, base: i32):
    @compile(suffix=(api, base, "basic"))
    def run() -> i32:
        m: api
        mp: ptr[api] = ptr(m)
        api.init(mp)

        api.insert(mp, base, i64(base * 10))
        api.insert(mp, base + 1, i64((base + 1) * 10))
        api.insert(mp, base + 2, i64((base + 2) * 10))

        if not api.contains(mp, base):
            return 1
        if api.contains(mp, base + 100):
            return 2

        p = api.find(mp, base + 1)
        if p == nullptr:
            return 3
        if p[0] != i64((base + 1) * 10):
            return 4

        if api.size(mp) != 3:
            return 5

        if not api.erase(mp, base + 1):
            return 6
        if api.contains(mp, base + 1):
            return 7
        if api.size(mp) != 2:
            return 8

        api.insert(mp, base + 3, i64((base + 3) * 10))
        if not api.contains(mp, base + 3):
            return 9

        api.destroy(mp)
        return 0
    return run


def _make_update_tester(api, base: i32):
    @compile(suffix=(api, base, "update"))
    def run() -> i32:
        m: api
        mp: ptr[api] = ptr(m)
        api.init(mp)

        if not api.insert(mp, base, i64(1)):
            return 1
        if api.insert(mp, base, i64(2)):
            return 2
        if api.size(mp) != 1:
            return 3

        p = api.find(mp, base)
        if p == nullptr:
            return 4
        if p[0] != i64(2):
            return 5

        api.destroy(mp)
        return 0
    return run


# ---------------------------------------------------------------------------
# Large scale
# ---------------------------------------------------------------------------

def _make_large_scale_tester(api, count: i32):
    @compile(suffix=(api, count, "large_scale"))
    def run() -> i32:
        m: api
        mp: ptr[api] = ptr(m)
        api.init(mp)

        i: i32 = 0
        while i < count:
            key: i32 = i * 7 + 3
            api.insert(mp, key, i64(key * 11))
            i += 1

        if api.size(mp) != count:
            return 1

        i = 0
        while i < count:
            key: i32 = i * 7 + 3
            p = api.find(mp, key)
            if p == nullptr:
                return 2
            if p[0] != i64(key * 11):
                return 3
            i += 1

        # Erase every third element.
        i = 0
        while i < count:
            key: i32 = i * 7 + 3
            if i % 3 == 0:
                if not api.erase(mp, key):
                    return 4
            i += 1

        expected_size: i32 = count - (count + 2) // 3
        if api.size(mp) != expected_size:
            return 5

        i = 0
        while i < count:
            key: i32 = i * 7 + 3
            p = api.find(mp, key)
            if i % 3 == 0:
                if p != nullptr:
                    return 6
            else:
                if p == nullptr or p[0] != i64(key * 11):
                    return 7
            i += 1

        api.destroy(mp)
        return 0
    return run


# ---------------------------------------------------------------------------
# Collision-heavy keys
# ---------------------------------------------------------------------------

def _make_collision_tester(api, count: i32):
    @compile(suffix=(api, count, "collision"))
    def run() -> i32:
        m: api
        mp: ptr[api] = ptr(m)
        api.init(mp)

        i: i32 = 0
        while i < count:
            key: i32 = i * 2048
            api.insert(mp, key, i64(key))
            i += 1

        if api.size(mp) != count:
            return 1

        i = 0
        while i < count:
            key: i32 = i * 2048
            p = api.find(mp, key)
            if p == nullptr or p[0] != i64(key):
                return 2
            i += 1

        # Erase from the front.
        i = 0
        while i < count:
            key: i32 = i * 2048
            if not api.erase(mp, key):
                return 3
            i += 1

        if api.size(mp) != 0:
            return 4

        api.destroy(mp)
        return 0
    return run


# ---------------------------------------------------------------------------
# Repeated rehash / load-factor boundary
# ---------------------------------------------------------------------------

def _make_repeated_rehash_tester(api, count: i32):
    @compile(suffix=(api, count, "repeated_rehash"))
    def run() -> i32:
        m: api
        mp: ptr[api] = ptr(m)
        api.init(mp)

        i: i32 = 0
        while i < count:
            api.insert(mp, i, i64(i * 2))
            j: i32 = 0
            while j <= i:
                p = api.find(mp, j)
                if p == nullptr or p[0] != i64(j * 2):
                    return i * 1000 + j + 1
                j += 1
            i += 1

        if api.size(mp) != count:
            return -1

        api.destroy(mp)
        return 0
    return run


# ---------------------------------------------------------------------------
# Value mutation
# ---------------------------------------------------------------------------

def _make_value_mutation_tester(api, count: i32):
    @compile(suffix=(api, count, "value_mutation"))
    def run() -> i32:
        m: api
        mp: ptr[api] = ptr(m)
        api.init(mp)

        i: i32 = 0
        while i < count:
            api.insert(mp, i, i64(i))
            i += 1

        # Mutate values in place through the pointer returned by find.
        i = 0
        while i < count:
            p = api.find(mp, i)
            if p == nullptr:
                return 1
            p[0] = i64(i * 100)
            i += 1

        # Verify mutations persisted.
        i = 0
        while i < count:
            p = api.find(mp, i)
            if p == nullptr or p[0] != i64(i * 100):
                return 2
            i += 1

        api.destroy(mp)
        return 0
    return run


# ---------------------------------------------------------------------------
# Repeated updates of the same key
# ---------------------------------------------------------------------------

def _make_repeated_update_tester(api, count: i32):
    @compile(suffix=(api, count, "repeated_update"))
    def run() -> i32:
        m: api
        mp: ptr[api] = ptr(m)
        api.init(mp)

        i: i32 = 0
        while i < count:
            api.insert(mp, 42, i64(i))
            p = api.find(mp, 42)
            if p == nullptr or p[0] != i64(i):
                return 1
            i += 1

        if api.size(mp) != 1:
            return 2

        api.destroy(mp)
        return 0
    return run


# ---------------------------------------------------------------------------
# Tombstone reuse and clear-then-reuse
# ---------------------------------------------------------------------------

def _make_tombstone_tester(api, base: i32):
    @compile(suffix=(api, base, "tombstone"))
    def run() -> i32:
        m: api
        mp: ptr[api] = ptr(m)
        api.init(mp)

        i: i32 = 0
        while i < 32:
            api.insert(mp, base + i, i64(i))
            i += 1

        api.erase(mp, base + 5)
        api.erase(mp, base + 10)
        api.erase(mp, base + 15)

        api.insert(mp, base + 5, i64(500))
        api.insert(mp, base + 40, i64(4000))

        if api.size(mp) != 31:
            return 1

        p = api.find(mp, base + 5)
        if p == nullptr or p[0] != i64(500):
            return 2
        if api.contains(mp, base + 10):
            return 3
        p = api.find(mp, base + 40)
        if p == nullptr or p[0] != i64(4000):
            return 4

        api.destroy(mp)
        return 0
    return run


def _make_clear_reuse_tester(api, count: i32):
    @compile(suffix=(api, count, "clear_reuse"))
    def run() -> i32:
        m: api
        mp: ptr[api] = ptr(m)
        api.init(mp)

        i: i32 = 0
        while i < count:
            api.insert(mp, i, i64(i))
            i += 1
        api.clear(mp)

        if api.size(mp) != 0:
            return 1
        if api.contains(mp, 0):
            return 2

        i = 0
        while i < count:
            api.insert(mp, i + 1000, i64(i + 1000))
            i += 1

        if api.size(mp) != count:
            return 3

        i = 0
        while i < count:
            p = api.find(mp, i + 1000)
            if p == nullptr or p[0] != i64(i + 1000):
                return 4
            i += 1

        api.destroy(mp)
        return 0
    return run


# ---------------------------------------------------------------------------
# Deterministic stress test
# ---------------------------------------------------------------------------

def _make_stress_tester(api, steps: i32, seed: i32):
    @compile(suffix=(api, steps, seed, "stress"))
    def run() -> i32:
        m: api
        mp: ptr[api] = ptr(m)
        api.init(mp)

        N: i32 = 1024
        present: array[i8, 1024]
        values: array[i64, 1024]
        i: i32 = 0
        while i < N:
            present[i] = 0
            values[i] = i64(0)
            i += 1

        state: u64 = u64(seed)
        step: i32 = 0
        while step < steps:
            state = state * u64(1103515245) + u64(12345)
            r: u64 = state & u64(0x7FFFFFFF)

            op: i32 = i32(r & u64(3))   # 0/1 insert, 2 erase, 3 contains
            key_idx: i32 = i32((r >> 2) & u64(1023))

            if op == 0 or op == 1:
                new_value: i64 = i64(step)
                inserted: bool = api.insert(mp, key_idx, new_value)
                if present[key_idx] == 1:
                    if inserted:
                        return step * 10 + 1
                else:
                    if not inserted:
                        return step * 10 + 2
                    present[key_idx] = 1
                values[key_idx] = new_value
            elif op == 2:
                erased: bool = api.erase(mp, key_idx)
                if present[key_idx] == 1:
                    if not erased:
                        return step * 10 + 3
                    present[key_idx] = 0
                else:
                    if erased:
                        return step * 10 + 4
            else:
                has: bool = api.contains(mp, key_idx)
                expected: i8 = present[key_idx]
                if expected == 1 and not has:
                    return step * 10 + 5
                if expected == 0 and has:
                    return step * 10 + 6

            step += 1

        i = 0
        while i < N:
            p = api.find(mp, i)
            if present[i] == 1:
                if p == nullptr or p[0] != values[i]:
                    return 9000000 + i
            else:
                if p != nullptr:
                    return 9100000 + i
            i += 1

        api.destroy(mp)
        return 0
    return run


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def _make_empty_ops_tester(api):
    @compile(suffix=(api, "empty_ops"))
    def run() -> i32:
        m: api
        mp: ptr[api] = ptr(m)
        api.init(mp)

        if api.size(mp) != 0:
            return 1
        if api.capacity(mp) != 0:
            return 2
        if api.contains(mp, 42):
            return 3
        if api.find(mp, 42) != nullptr:
            return 4
        if api.erase(mp, 42):
            return 5

        api.destroy(mp)
        return 0
    return run


def _make_multiple_instances_tester(api, count: i32):
    @compile(suffix=(api, count, "multi_instance"))
    def run() -> i32:
        a: api
        b: api
        ap: ptr[api] = ptr(a)
        bp: ptr[api] = ptr(b)
        api.init(ap)
        api.init(bp)

        i: i32 = 0
        while i < count:
            api.insert(ap, i, i64(i))
            api.insert(bp, i + 10000, i64(i + 10000))
            i += 1

        if api.size(ap) != count:
            return 1
        if api.size(bp) != count:
            return 2

        i = 0
        while i < count:
            pa = api.find(ap, i)
            pb = api.find(bp, i + 10000)
            if pa == nullptr or pa[0] != i64(i):
                return 3
            if pb == nullptr or pb[0] != i64(i + 10000):
                return 4
            if api.contains(ap, i + 10000):
                return 5
            if api.contains(bp, i):
                return 6
            i += 1

        api.destroy(ap)
        api.destroy(bp)
        return 0
    return run


# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------

def test_basic():
    for api in ALL_MAPS:
        result = _make_basic_tester(api, 10)()
        assert result == 0, f"basic failed for {api.__name__}: {result}"
    print("OK test_basic")


def test_update():
    for api in ALL_MAPS:
        result = _make_update_tester(api, 20)()
        assert result == 0, f"update failed for {api.__name__}: {result}"
    print("OK test_update")


def test_large_scale():
    for api in STRESS_MAPS:
        result = _make_large_scale_tester(api, 10000)()
        assert result == 0, f"large_scale failed for {api.__name__}: {result}"
    print("OK test_large_scale")


def test_collision_heavy():
    for api in STRESS_MAPS:
        result = _make_collision_tester(api, 64)()
        assert result == 0, f"collision failed for {api.__name__}: {result}"
    print("OK test_collision_heavy")


def test_repeated_rehash():
    for api in STRESS_MAPS:
        result = _make_repeated_rehash_tester(api, 500)()
        assert result == 0, f"repeated_rehash failed for {api.__name__}: {result}"
    print("OK test_repeated_rehash")


def test_value_mutation():
    for api in STRESS_MAPS:
        result = _make_value_mutation_tester(api, 1000)()
        assert result == 0, f"value_mutation failed for {api.__name__}: {result}"
    print("OK test_value_mutation")


def test_repeated_update():
    for api in STRESS_MAPS:
        result = _make_repeated_update_tester(api, 1000)()
        assert result == 0, f"repeated_update failed for {api.__name__}: {result}"
    print("OK test_repeated_update")


def test_tombstone():
    for api in ALL_MAPS:
        result = _make_tombstone_tester(api, 0)()
        assert result == 0, f"tombstone failed for {api.__name__}: {result}"
    print("OK test_tombstone")


def test_clear_reuse():
    for api in STRESS_MAPS:
        result = _make_clear_reuse_tester(api, 1000)()
        assert result == 0, f"clear_reuse failed for {api.__name__}: {result}"
    print("OK test_clear_reuse")


def test_stress():
    for api in STRESS_MAPS:
        result = _make_stress_tester(api, 5000, 12345)()
        assert result == 0, f"stress failed for {api.__name__}: {result}"
    print("OK test_stress")


def test_empty_ops():
    for api in ALL_MAPS:
        result = _make_empty_ops_tester(api)()
        assert result == 0, f"empty_ops failed for {api.__name__}: {result}"
    print("OK test_empty_ops")


def test_multiple_instances():
    for api in STRESS_MAPS:
        result = _make_multiple_instances_tester(api, 1000)()
        assert result == 0, f"multiple_instances failed for {api.__name__}: {result}"
    print("OK test_multiple_instances")


# ---------------------------------------------------------------------------
# Failure tests
# ---------------------------------------------------------------------------

def test_unsupported_key_type():
    from pythoc import f64
    try:
        FlatHashMap(f64, i64)
    except TypeError as e:
        if "no default hash" in str(e):
            print("OK test_unsupported_key_type")
            return
    raise AssertionError("FlatHashMap(f64, i64) should raise TypeError")


def test_bad_factory_args():
    try:
        FlatHashMap(42, i64)
    except (TypeError, AttributeError):
        pass
    else:
        raise AssertionError("FlatHashMap(42, i64) should raise an error")

    try:
        FlatHashMap(i32, i64, size_type="u64")
    except TypeError as e:
        if "size_type must be a PythoC integer type" not in str(e):
            raise AssertionError(f"unexpected error message: {e}")
    else:
        raise AssertionError("FlatHashMap(i32, i64, size_type=\"u64\") should raise TypeError")

    print("OK test_bad_factory_args")


if __name__ == '__main__':
    test_basic()
    test_update()
    test_empty_ops()
    test_tombstone()
    test_large_scale()
    test_collision_heavy()
    test_repeated_rehash()
    test_value_mutation()
    test_repeated_update()
    test_clear_reuse()
    test_multiple_instances()
    test_stress()
    test_unsupported_key_type()
    test_bad_factory_args()
    print("All FlatHashMap tests passed!")
