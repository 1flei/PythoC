from __future__ import annotations
from pythoc import *
from pythoc.libc.stdlib import malloc, free
from pythoc.libc.string import memset, memcpy


# Control byte constants (signed i8 representation).
_CTRL_EMPTY = i8(-128)      # 0x80: never touched
_CTRL_DELETED = i8(-2)      # 0xFE: tombstone

# Maximum load factor numerator/denominator: 7/8.
_LOAD_NUM = 7
_LOAD_DEN = 8

# Minimum non-zero capacity.
_MIN_CAPACITY = 16


# Constants from Abseil's MixingHashState.  kStaticRandomData[0] is used as a
# fixed seed; the multiplier kMul is chosen so that the 128-bit product spreads
# entropy across all output bits.
_HASH_K_MUL = u64(0x79d5f9e0de1e8cf5)
_HASH_SEED = u64(0x243f6a8885a308d3)


@compile
def _mix64(a: u64, b: u64) -> u64:
    """Return high64(a*b) ^ low64(a*b) without using a 128-bit type."""
    a_lo: u64 = a & u64(0xFFFFFFFF)
    a_hi: u64 = a >> 32
    b_lo: u64 = b & u64(0xFFFFFFFF)
    b_hi: u64 = b >> 32

    lo: u64 = a_lo * b_lo
    mid1: u64 = a_lo * b_hi
    mid2: u64 = a_hi * b_lo
    hi: u64 = a_hi * b_hi

    mid: u64 = mid1 + mid2
    mid_carry: u64 = u64(0)
    if mid < mid1:
        mid_carry = u64(1)

    mid_lo_shifted: u64 = (mid & u64(0xFFFFFFFF)) << 32
    low64: u64 = lo + mid_lo_shifted
    carry: u64 = (mid >> 32) + (mid_carry << 32)
    if low64 < lo:
        carry += 1
    high64: u64 = hi + carry

    return high64 ^ low64


@compile
def _scalar_hash(T: param, key: T) -> u64:
    """Abseil-style hash for scalar integer keys.

    Mirrors ``absl::Hash``'s IntegralFastPath: cast to unsigned 64-bit and mix
    with a seed and the fixed multiplier ``kMul``.
    """
    v: u64 = u64(key)
    return _mix64(_HASH_SEED ^ v, _HASH_K_MUL)


@compile
def _identity_eq(T: param, a: T, b: T) -> bool:
    return a == b


_SCALAR_INTEGER_TYPES = [i8, i16, i32, i64, u8, u16, u32, u64]

_DEFAULT_HASH = {t: _scalar_hash(t) for t in _SCALAR_INTEGER_TYPES}
_DEFAULT_EQ = {t: _identity_eq(t) for t in _SCALAR_INTEGER_TYPES}


def FlatHashSet(key_type, size_type=u64):
    """Factory producing a flat hash set specialized for ``key_type``.

    The returned object is the compiled ``_FlatHashSet`` class itself, with
    compiled helper methods declared directly inside the class body.  Callers
    write::

        IntSet = FlatHashSet(i32)
        s: IntSet
        sp = ptr(s)
        IntSet.init(sp)
        IntSet.insert(sp, 42)

    Instance-level attribute access resolves struct fields and class-level
    access resolves class attributes (methods), so a field and a method may
    share the same name without ambiguity.
    """
    if key_type not in _DEFAULT_HASH:
        raise TypeError(
            f"FlatHashSet: no default hash for key type {key_type}. "
            f"Supported: {list(_DEFAULT_HASH.keys())}"
        )
    if not (hasattr(size_type, '_is_integer') and size_type._is_integer):
        raise TypeError(
            f"FlatHashSet: size_type must be a PythoC integer type, got {size_type}"
        )

    hash_fn = _DEFAULT_HASH[key_type]
    eq_fn = _DEFAULT_EQ[key_type]
    type_suffix = (key_type, size_type)

    @compile(suffix=type_suffix)
    class _FlatHashSet:
        size: size_type
        capacity: size_type
        growth_left: size_type
        ctrl: ptr[i8]
        slots: ptr[key_type]

        def init(s: ptr[_FlatHashSet]) -> None:
            s.size = 0
            s.capacity = 0
            s.growth_left = 0
            s.ctrl = nullptr
            s.slots = nullptr

        def destroy(s: ptr[_FlatHashSet]) -> None:
            if s.capacity > 0:
                free(s.ctrl)
                free(s.slots)
            s.size = 0
            s.capacity = 0
            s.growth_left = 0
            s.ctrl = nullptr
            s.slots = nullptr

        def size(s: ptr[_FlatHashSet]) -> size_type:
            return s.size

        def capacity(s: ptr[_FlatHashSet]) -> size_type:
            return s.capacity

        def clear(s: ptr[_FlatHashSet]) -> None:
            if s.capacity > 0:
                memset(s.ctrl, -128, s.capacity)
            s.size = 0
            s.growth_left = s.capacity * _LOAD_NUM / _LOAD_DEN

        def find(s: ptr[_FlatHashSet], key: key_type) -> ptr[key_type]:
            if s.capacity == 0:
                return nullptr

            h: u64 = hash_fn(key)
            h2: i8 = i8(u8(h & 0x7F))
            idx: size_type = size_type(h >> 7) % s.capacity
            cap: size_type = s.capacity

            while True:
                c: i8 = s.ctrl[idx]
                if c == _CTRL_EMPTY:
                    return nullptr
                if c == h2:
                    slot: ptr[key_type] = ptr(s.slots[idx])
                    if eq_fn(slot[0], key):
                        return slot
                idx = (idx + 1) % cap

        def contains(s: ptr[_FlatHashSet], key: key_type) -> bool:
            return _FlatHashSet.find(s, key) != nullptr

        def _insert_no_grow(s: ptr[_FlatHashSet], key: key_type) -> bool:
            h: u64 = hash_fn(key)
            h2: i8 = i8(u8(h & 0x7F))
            idx: size_type = size_type(h >> 7) % s.capacity
            cap: size_type = s.capacity
            first_deleted: size_type = cap

            while True:
                c: i8 = s.ctrl[idx]
                if c == _CTRL_EMPTY:
                    if first_deleted < cap:
                        idx = first_deleted
                    s.ctrl[idx] = h2
                    s.slots[idx] = key
                    s.size += 1
                    s.growth_left -= 1
                    return True
                if c == _CTRL_DELETED:
                    if first_deleted == cap:
                        first_deleted = idx
                elif c == h2:
                    slot: ptr[key_type] = ptr(s.slots[idx])
                    if eq_fn(slot[0], key):
                        return False
                idx = (idx + 1) % cap

        def _rehash(s: ptr[_FlatHashSet], newcapacity: size_type) -> None:
            old_ctrl: ptr[i8] = s.ctrl
            old_slots: ptr[key_type] = s.slots
            old_cap: size_type = s.capacity

            new_ctrl: ptr[i8] = malloc(newcapacity)
            memset(new_ctrl, -128, newcapacity)
            new_slots: ptr[key_type] = malloc(newcapacity * sizeof(key_type))

            s.ctrl = new_ctrl
            s.slots = new_slots
            s.capacity = newcapacity
            s.size = 0
            s.growth_left = newcapacity * _LOAD_NUM / _LOAD_DEN

            if old_cap > 0:
                i: size_type = 0
                while i < old_cap:
                    c: i8 = old_ctrl[i]
                    if c != _CTRL_EMPTY and c != _CTRL_DELETED:
                        _FlatHashSet._insert_no_grow(s, old_slots[i])
                    i += 1
                free(old_ctrl)
                free(old_slots)

        def insert(s: ptr[_FlatHashSet], key: key_type) -> bool:
            if s.growth_left == 0:
                new_cap: size_type = _MIN_CAPACITY
                if s.capacity > 0:
                    new_cap = s.capacity * 2
                _FlatHashSet._rehash(s, new_cap)
            return _FlatHashSet._insert_no_grow(s, key)

        def erase(s: ptr[_FlatHashSet], key: key_type) -> bool:
            if s.capacity == 0:
                return False

            h: u64 = hash_fn(key)
            h2: i8 = i8(u8(h & 0x7F))
            idx: size_type = size_type(h >> 7) % s.capacity
            cap: size_type = s.capacity

            while True:
                c: i8 = s.ctrl[idx]
                if c == _CTRL_EMPTY:
                    return False
                if c == h2:
                    slot: ptr[key_type] = ptr(s.slots[idx])
                    if eq_fn(slot[0], key):
                        s.ctrl[idx] = _CTRL_DELETED
                        s.size -= 1
                        return True
                idx = (idx + 1) % cap

    return _FlatHashSet
