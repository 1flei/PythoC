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


# Re-use the identity hash/equality wrappers from std.set so both containers
# share the same compiled helpers for scalar integer keys.
from pythoc.std.set import (
    _DEFAULT_HASH as _DEFAULT_HASH,
    _DEFAULT_EQ as _DEFAULT_EQ,
)


def FlatHashMap(key_type, value_type, size_type=u64):
    """Factory producing a flat hash map specialized for ``key_type``.

    The returned object is the compiled ``_FlatHashMap`` class itself, with
    compiled helper methods declared directly inside the class body.  Callers
    write::

        IntMap = FlatHashMap(i32, i64)
        m: IntMap
        mp = ptr(m)
        IntMap.init(mp)
        IntMap.insert(mp, 1, 2)

    Instance-level attribute access resolves struct fields and class-level
    access resolves class attributes (methods), so a field and a method may
    share the same name without ambiguity.
    """
    if key_type not in _DEFAULT_HASH:
        raise TypeError(
            f"FlatHashMap: no default hash for key type {key_type}. "
            f"Supported: {list(_DEFAULT_HASH.keys())}"
        )
    if not (hasattr(size_type, '_is_integer') and size_type._is_integer):
        raise TypeError(
            f"FlatHashMap: size_type must be a PythoC integer type, got {size_type}"
        )

    hash_fn = _DEFAULT_HASH[key_type]
    eq_fn = _DEFAULT_EQ[key_type]
    type_suffix = (key_type, value_type, size_type)

    @compile(suffix=type_suffix)
    class _FlatHashMap:
        size: size_type
        capacity: size_type
        growth_left: size_type
        ctrl: ptr[i8]
        keys: ptr[key_type]
        values: ptr[value_type]

        def init(m: ptr[_FlatHashMap]) -> None:
            m.size = 0
            m.capacity = 0
            m.growth_left = 0
            m.ctrl = nullptr
            m.keys = nullptr
            m.values = nullptr

        def destroy(m: ptr[_FlatHashMap]) -> None:
            if m.capacity > 0:
                free(m.ctrl)
                free(m.keys)
                free(m.values)
            m.size = 0
            m.capacity = 0
            m.growth_left = 0
            m.ctrl = nullptr
            m.keys = nullptr
            m.values = nullptr

        def size(m: ptr[_FlatHashMap]) -> size_type:
            return m.size

        def capacity(m: ptr[_FlatHashMap]) -> size_type:
            return m.capacity

        def clear(m: ptr[_FlatHashMap]) -> None:
            if m.capacity > 0:
                memset(m.ctrl, -128, m.capacity)
            m.size = 0
            m.growth_left = m.capacity * _LOAD_NUM / _LOAD_DEN

        def find(m: ptr[_FlatHashMap], key: key_type) -> ptr[value_type]:
            if m.capacity == 0:
                return nullptr

            h: u64 = hash_fn(key)
            h2: i8 = i8(u8(h & 0x7F))
            idx: size_type = size_type(h >> 7) % m.capacity
            cap: size_type = m.capacity

            while True:
                c: i8 = m.ctrl[idx]
                if c == _CTRL_EMPTY:
                    return nullptr
                if c == h2:
                    slot_key: ptr[key_type] = ptr(m.keys[idx])
                    if eq_fn(slot_key[0], key):
                        return ptr(m.values[idx])
                idx = (idx + 1) % cap

        def contains(m: ptr[_FlatHashMap], key: key_type) -> bool:
            return _FlatHashMap.find(m, key) != nullptr

        def _insert_no_grow(m: ptr[_FlatHashMap], key: key_type, value: value_type) -> bool:
            h: u64 = hash_fn(key)
            h2: i8 = i8(u8(h & 0x7F))
            idx: size_type = size_type(h >> 7) % m.capacity
            cap: size_type = m.capacity
            first_deleted: size_type = cap

            while True:
                c: i8 = m.ctrl[idx]
                if c == _CTRL_EMPTY:
                    if first_deleted < cap:
                        idx = first_deleted
                    m.ctrl[idx] = h2
                    m.keys[idx] = key
                    m.values[idx] = value
                    m.size += 1
                    m.growth_left -= 1
                    return True
                if c == _CTRL_DELETED:
                    if first_deleted == cap:
                        first_deleted = idx
                elif c == h2:
                    slot_key: ptr[key_type] = ptr(m.keys[idx])
                    if eq_fn(slot_key[0], key):
                        m.values[idx] = value
                        return False
                idx = (idx + 1) % cap

        def _rehash(m: ptr[_FlatHashMap], newcapacity: size_type) -> None:
            old_ctrl: ptr[i8] = m.ctrl
            old_keys: ptr[key_type] = m.keys
            old_values: ptr[value_type] = m.values
            old_cap: size_type = m.capacity

            new_ctrl: ptr[i8] = malloc(newcapacity)
            memset(new_ctrl, -128, newcapacity)
            new_keys: ptr[key_type] = malloc(newcapacity * sizeof(key_type))
            new_values: ptr[value_type] = malloc(newcapacity * sizeof(value_type))

            m.ctrl = new_ctrl
            m.keys = new_keys
            m.values = new_values
            m.capacity = newcapacity
            m.size = 0
            m.growth_left = newcapacity * _LOAD_NUM / _LOAD_DEN

            if old_cap > 0:
                i: size_type = 0
                while i < old_cap:
                    c: i8 = old_ctrl[i]
                    if c != _CTRL_EMPTY and c != _CTRL_DELETED:
                        _FlatHashMap._insert_no_grow(m, old_keys[i], old_values[i])
                    i += 1
                free(old_ctrl)
                free(old_keys)
                free(old_values)

        def insert(m: ptr[_FlatHashMap], key: key_type, value: value_type) -> bool:
            if m.growth_left == 0:
                new_cap: size_type = _MIN_CAPACITY
                if m.capacity > 0:
                    new_cap = m.capacity * 2
                _FlatHashMap._rehash(m, new_cap)
            return _FlatHashMap._insert_no_grow(m, key, value)

        def erase(m: ptr[_FlatHashMap], key: key_type) -> bool:
            if m.capacity == 0:
                return False

            h: u64 = hash_fn(key)
            h2: i8 = i8(u8(h & 0x7F))
            idx: size_type = size_type(h >> 7) % m.capacity
            cap: size_type = m.capacity

            while True:
                c: i8 = m.ctrl[idx]
                if c == _CTRL_EMPTY:
                    return False
                if c == h2:
                    slot_key: ptr[key_type] = ptr(m.keys[idx])
                    if eq_fn(slot_key[0], key):
                        m.ctrl[idx] = _CTRL_DELETED
                        m.size -= 1
                        return True
                idx = (idx + 1) % cap

    return _FlatHashMap
