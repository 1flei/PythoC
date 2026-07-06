#!/usr/bin/env python3
"""Integration tests for pythoc.std.set.FlatHashSet.

Coverage:
- basic operations across all supported scalar integer key types
- large-scale insert/erase/lookup (10k+ elements)
- collision-heavy key patterns
- load-factor boundary and repeated rehash
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
from pythoc.std.set import FlatHashSet


# Factory instances for the scalar key types we support out of the box.
I8Set = FlatHashSet(i8)
I16Set = FlatHashSet(i16)
I32Set = FlatHashSet(i32)
I64Set = FlatHashSet(i64)
U8Set = FlatHashSet(u8)
U16Set = FlatHashSet(u16)
U32Set = FlatHashSet(u32)
U64Set = FlatHashSet(u64)

ALL_SETS = [
    I8Set, I16Set, I32Set, I64Set,
    U8Set, U16Set, U32Set, U64Set,
]

# Representative sets used for expensive stress tests to keep runtime reasonable.
STRESS_SETS = [I32Set, I64Set, U64Set]


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def _make_basic_tester(api, base: i32):
    @compile(suffix=(api, base, "basic"))
    def run() -> i32:
        s: api
        sp: ptr[api] = ptr(s)
        api.init(sp)

        api.insert(sp, base)
        api.insert(sp, base + 1)
        api.insert(sp, base + 2)

        if not api.contains(sp, base):
            return 1
        if not api.contains(sp, base + 1):
            return 2
        if not api.contains(sp, base + 2):
            return 3
        if api.contains(sp, base + 100):
            return 4

        if api.size(sp) != 3:
            return 5

        if not api.erase(sp, base + 1):
            return 6
        if api.contains(sp, base + 1):
            return 7
        if api.size(sp) != 2:
            return 8

        api.insert(sp, base + 3)
        if not api.contains(sp, base + 3):
            return 9

        api.destroy(sp)
        return 0
    return run


def _make_duplicate_tester(api, base: i32):
    @compile(suffix=(api, base, "duplicate"))
    def run() -> i32:
        s: api
        sp: ptr[api] = ptr(s)
        api.init(sp)

        if not api.insert(sp, base):
            return 1
        if api.insert(sp, base):
            return 2
        if api.size(sp) != 1:
            return 3

        api.destroy(sp)
        return 0
    return run


def _make_find_tester(api, base: i32):
    @compile(suffix=(api, base, "find"))
    def run() -> i32:
        s: api
        sp: ptr[api] = ptr(s)
        api.init(sp)

        api.insert(sp, base)

        p = api.find(sp, base)
        if p == nullptr:
            return 1
        if p[0] != base:
            return 2

        p = api.find(sp, base + 1)
        if p != nullptr:
            return 3

        api.destroy(sp)
        return 0
    return run


# ---------------------------------------------------------------------------
# Large scale
# ---------------------------------------------------------------------------

def _make_large_scale_tester(api, count: i32):
    @compile(suffix=(api, count, "large_scale"))
    def run() -> i32:
        s: api
        sp: ptr[api] = ptr(s)
        api.init(sp)

        i: i32 = 0
        while i < count:
            key: i32 = i * 7 + 3
            api.insert(sp, key)
            i += 1

        if api.size(sp) != count:
            return 1

        i = 0
        while i < count:
            key: i32 = i * 7 + 3
            if not api.contains(sp, key):
                return 2
            i += 1

        # Erase every third element.
        i = 0
        while i < count:
            key: i32 = i * 7 + 3
            if i % 3 == 0:
                if not api.erase(sp, key):
                    return 3
            i += 1

        expected_size: i32 = count - (count + 2) // 3
        if api.size(sp) != expected_size:
            return 4

        i = 0
        while i < count:
            key: i32 = i * 7 + 3
            if i % 3 == 0:
                if api.contains(sp, key):
                    return 5
            else:
                if not api.contains(sp, key):
                    return 6
            i += 1

        api.destroy(sp)
        return 0
    return run


# ---------------------------------------------------------------------------
# Collision-heavy keys
#
# With identity hash, keys 0, 2048, 4096, ... share h2=0 and idx=0 for any
# power-of-two capacity <= 2048, forcing long probe sequences.
# ---------------------------------------------------------------------------

def _make_collision_tester(api, count: i32):
    @compile(suffix=(api, count, "collision"))
    def run() -> i32:
        s: api
        sp: ptr[api] = ptr(s)
        api.init(sp)

        i: i32 = 0
        while i < count:
            key: i32 = i * 2048
            api.insert(sp, key)
            i += 1

        if api.size(sp) != count:
            return 1

        i = 0
        while i < count:
            key: i32 = i * 2048
            if not api.contains(sp, key):
                return 2
            i += 1

        # Erase from the front (worst-case probe walk).
        i = 0
        while i < count:
            key: i32 = i * 2048
            if not api.erase(sp, key):
                return 3
            i += 1

        if api.size(sp) != 0:
            return 4

        api.destroy(sp)
        return 0
    return run


# ---------------------------------------------------------------------------
# Repeated rehash / load-factor boundary
# ---------------------------------------------------------------------------

def _make_repeated_rehash_tester(api, count: i32):
    @compile(suffix=(api, count, "repeated_rehash"))
    def run() -> i32:
        s: api
        sp: ptr[api] = ptr(s)
        api.init(sp)

        i: i32 = 0
        while i < count:
            api.insert(sp, i)
            # Verify after every insertion that all previous keys are still reachable.
            j: i32 = 0
            while j <= i:
                if not api.contains(sp, j):
                    return i * 1000 + j + 1
                j += 1
            i += 1

        if api.size(sp) != count:
            return -1

        api.destroy(sp)
        return 0
    return run


# ---------------------------------------------------------------------------
# Tombstone reuse and clear-then-reuse
# ---------------------------------------------------------------------------

def _make_tombstone_tester(api, base: i32):
    @compile(suffix=(api, base, "tombstone"))
    def run() -> i32:
        s: api
        sp: ptr[api] = ptr(s)
        api.init(sp)

        # Fill enough to allocate a control table.
        i: i32 = 0
        while i < 32:
            api.insert(sp, base + i)
            i += 1

        api.erase(sp, base + 5)
        api.erase(sp, base + 10)
        api.erase(sp, base + 15)

        # Re-insert (should reuse tombstones) and a new key.
        api.insert(sp, base + 5)
        api.insert(sp, base + 40)

        if api.size(sp) != 31:
            return 1
        if not api.contains(sp, base + 5):
            return 2
        if api.contains(sp, base + 10):
            return 3
        if not api.contains(sp, base + 40):
            return 4

        api.destroy(sp)
        return 0
    return run


def _make_clear_reuse_tester(api, count: i32):
    @compile(suffix=(api, count, "clear_reuse"))
    def run() -> i32:
        s: api
        sp: ptr[api] = ptr(s)
        api.init(sp)

        i: i32 = 0
        while i < count:
            api.insert(sp, i)
            i += 1
        api.clear(sp)

        if api.size(sp) != 0:
            return 1
        if api.contains(sp, 0):
            return 2

        # Reuse the cleared set.
        i = 0
        while i < count:
            api.insert(sp, i + 1000)
            i += 1

        if api.size(sp) != count:
            return 3

        i = 0
        while i < count:
            if not api.contains(sp, i + 1000):
                return 4
            i += 1

        api.destroy(sp)
        return 0
    return run


# ---------------------------------------------------------------------------
# Deterministic stress test
#
# Uses a tiny LCG to drive a mixed sequence of inserts/erases/contains and
# checks against a reference bitset kept in a fixed-size array.
# ---------------------------------------------------------------------------

def _make_stress_tester(api, steps: i32, seed: i32):
    @compile(suffix=(api, steps, seed, "stress"))
    def run() -> i32:
        s: api
        sp: ptr[api] = ptr(s)
        api.init(sp)

        # Reference state: 1 means "should be present".  N is a power of two
        # so we can use bit masks for the pseudo-random index.
        N: i32 = 1024
        present: array[i8, 1024]
        i: i32 = 0
        while i < N:
            present[i] = 0
            i += 1

        state: u64 = u64(seed)
        step: i32 = 0
        while step < steps:
            # LCG (glibc constants).  Use bit masks instead of u64 % to avoid
            # PythoC lowering '%' to floating-point remainder.
            state = state * u64(1103515245) + u64(12345)
            r: u64 = state & u64(0x7FFFFFFF)

            op: i32 = i32(r & u64(3))   # 0/1 insert, 2 erase, 3 contains
            key_idx: i32 = i32((r >> 2) & u64(1023))

            if op == 0:
                inserted: bool = api.insert(sp, key_idx)
                if present[key_idx] == 1:
                    if inserted:
                        return step * 10 + 1
                else:
                    if not inserted:
                        return step * 10 + 2
                    present[key_idx] = 1
            elif op == 1:
                erased: bool = api.erase(sp, key_idx)
                if present[key_idx] == 1:
                    if not erased:
                        return step * 10 + 3
                    present[key_idx] = 0
                else:
                    if erased:
                        return step * 10 + 4
            else:
                has: bool = api.contains(sp, key_idx)
                expected: i8 = present[key_idx]
                if expected == 1 and not has:
                    return step * 10 + 5
                if expected == 0 and has:
                    return step * 10 + 6

            step += 1

        # Final consistency check.
        i = 0
        while i < N:
            has: bool = api.contains(sp, i)
            if present[i] == 1 and not has:
                return 9000000 + i
            if present[i] == 0 and has:
                return 9100000 + i
            i += 1

        api.destroy(sp)
        return 0
    return run


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def _make_empty_ops_tester(api):
    @compile(suffix=(api, "empty_ops"))
    def run() -> i32:
        s: api
        sp: ptr[api] = ptr(s)
        api.init(sp)

        if api.size(sp) != 0:
            return 1
        if api.capacity(sp) != 0:
            return 2
        if api.contains(sp, 42):
            return 3
        if api.find(sp, 42) != nullptr:
            return 4
        if api.erase(sp, 42):
            return 5

        api.destroy(sp)
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
            api.insert(ap, i)
            api.insert(bp, i + 10000)
            i += 1

        if api.size(ap) != count:
            return 1
        if api.size(bp) != count:
            return 2

        i = 0
        while i < count:
            if not api.contains(ap, i):
                return 3
            if api.contains(ap, i + 10000):
                return 4
            if not api.contains(bp, i + 10000):
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
    for api in ALL_SETS:
        result = _make_basic_tester(api, 10)()
        assert result == 0, f"basic failed for {api.__name__}: {result}"
    print("OK test_basic")


def test_duplicate():
    for api in ALL_SETS:
        result = _make_duplicate_tester(api, 20)()
        assert result == 0, f"duplicate failed for {api.__name__}: {result}"
    print("OK test_duplicate")


def test_find():
    for api in ALL_SETS:
        result = _make_find_tester(api, 40)()
        assert result == 0, f"find failed for {api.__name__}: {result}"
    print("OK test_find")


def test_large_scale():
    for api in STRESS_SETS:
        result = _make_large_scale_tester(api, 10000)()
        assert result == 0, f"large_scale failed for {api.__name__}: {result}"
    print("OK test_large_scale")


def test_collision_heavy():
    for api in STRESS_SETS:
        result = _make_collision_tester(api, 64)()
        assert result == 0, f"collision failed for {api.__name__}: {result}"
    print("OK test_collision_heavy")


def test_repeated_rehash():
    for api in STRESS_SETS:
        result = _make_repeated_rehash_tester(api, 500)()
        assert result == 0, f"repeated_rehash failed for {api.__name__}: {result}"
    print("OK test_repeated_rehash")


def test_tombstone():
    for api in ALL_SETS:
        result = _make_tombstone_tester(api, 0)()
        assert result == 0, f"tombstone failed for {api.__name__}: {result}"
    print("OK test_tombstone")


def test_clear_reuse():
    for api in STRESS_SETS:
        result = _make_clear_reuse_tester(api, 1000)()
        assert result == 0, f"clear_reuse failed for {api.__name__}: {result}"
    print("OK test_clear_reuse")


def test_stress():
    for api in STRESS_SETS:
        result = _make_stress_tester(api, 5000, 12345)()
        assert result == 0, f"stress failed for {api.__name__}: {result}"
    print("OK test_stress")


def test_empty_ops():
    for api in ALL_SETS:
        result = _make_empty_ops_tester(api)()
        assert result == 0, f"empty_ops failed for {api.__name__}: {result}"
    print("OK test_empty_ops")


def test_multiple_instances():
    for api in STRESS_SETS:
        result = _make_multiple_instances_tester(api, 1000)()
        assert result == 0, f"multiple_instances failed for {api.__name__}: {result}"
    print("OK test_multiple_instances")


# ---------------------------------------------------------------------------
# Failure tests
# ---------------------------------------------------------------------------

def test_unsupported_key_type():
    from pythoc import f64
    try:
        FlatHashSet(f64)
    except TypeError as e:
        if "no default hash" in str(e):
            print("OK test_unsupported_key_type")
            return
    raise AssertionError("FlatHashSet(f64) should raise TypeError")


def test_bad_factory_args():
    try:
        FlatHashSet(42)
    except (TypeError, AttributeError):
        pass
    else:
        raise AssertionError("FlatHashSet(42) should raise an error")

    try:
        FlatHashSet(i32, size_type="u64")
    except (TypeError, AttributeError):
        pass
    else:
        raise AssertionError("FlatHashSet(i32, size_type=\"u64\") should raise an error")

    print("OK test_bad_factory_args")


if __name__ == '__main__':
    test_basic()
    test_duplicate()
    test_find()
    test_empty_ops()
    test_tombstone()
    test_large_scale()
    test_collision_heavy()
    test_repeated_rehash()
    test_clear_reuse()
    test_multiple_instances()
    test_stress()
    test_unsupported_key_type()
    test_bad_factory_args()
    print("All FlatHashSet tests passed!")
