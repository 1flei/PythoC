"""Flat hash maps with a shared probing and slot-state implementation."""

from __future__ import annotations

from pythoc import (
    compile, effect, ptr, u64, i8, bool, void, nullptr, sizeof, struct, refine,
)
from pythoc.libc.string import memset
from pythoc.schema_protocol import is_linear_schema_type
from pythoc.std.mem import DefaultMem
from pythoc.std.hash import key_functions
from pythoc.std._hash_table import HashTable, EMPTY
from pythoc.std.refinement import nonnull


effect.default(mem=DefaultMem)


def FlatHashMap(key_type, value_type, size_type=u64, *, hash_fn=None, eq_fn=None):
    """Return a map type with fixed compile-time hash/equality operations.

    Elements are copyable values. Returned element pointers are invalidated by
    rehash, clear and destroy; this interface does not establish a lifetime borrow.
    """
    hash_fn, eq_fn = key_functions(key_type, hash_fn, eq_fn)
    if not (getattr(size_type, '_is_integer', False)
            and not getattr(size_type, '_is_signed', True)):
        raise TypeError('FlatHashMap size_type must be an unsigned integer type')
    if is_linear_schema_type(key_type) or is_linear_schema_type(value_type):
        raise TypeError('FlatHashMap requires copyable keys and values')
    Table = HashTable(key_type, size_type, hash_fn, eq_fn)
    Item = struct['key': key_type, 'value': value_type]
    ValuePointer = nonnull(ptr[value_type])

    @compile(suffix=(key_type, value_type, size_type, hash_fn, eq_fn))
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
                effect.mem.free(ptr[void](m.ctrl))
                effect.mem.free(ptr[void](m.keys))
                effect.mem.free(ptr[void](m.values))
            _FlatHashMap.init(m)

        def size(m: ptr[_FlatHashMap]) -> size_type:
            return m.size

        def capacity(m: ptr[_FlatHashMap]) -> size_type:
            return m.capacity

        def clear(m: ptr[_FlatHashMap]) -> None:
            if m.capacity > 0:
                memset(m.ctrl, EMPTY, m.capacity)
            m.size = 0
            m.growth_left = Table.growth_limit(m.capacity)

        def find(m: ptr[_FlatHashMap], key: key_type) -> ptr[value_type]:
            for position in Table.find(m.ctrl, m.keys, m.capacity, key):
                return ptr(m.values[position[2]])
            return nullptr

        def contains(m: ptr[_FlatHashMap], key: key_type) -> bool:
            return _FlatHashMap.find(m, key) != nullptr

        def at(m: ptr[_FlatHashMap], key: key_type) -> ValuePointer:
            """Yield a present value pointer, valid until an invalidating mutation."""
            for present in refine(_FlatHashMap.find(m, key), ValuePointer.predicate):
                yield present

        def _insert_no_grow(m: ptr[_FlatHashMap], key: key_type, value: value_type) -> bool:
            position = Table.insert(m.ctrl, m.keys, m.capacity,
                                    ptr(m.size), ptr(m.growth_left), key)
            if position.index < m.capacity:
                m.values[position.index] = value
            return position.inserted

        def _rehash(m: ptr[_FlatHashMap], capacity: size_type) -> None:
            old_ctrl = m.ctrl
            old_keys = m.keys
            old_values = m.values
            old_capacity = m.capacity
            m.ctrl = ptr[i8](effect.mem.malloc(u64(capacity)))
            memset(m.ctrl, EMPTY, capacity)
            m.keys = ptr[key_type](effect.mem.malloc(u64(capacity * sizeof(key_type))))
            m.values = ptr[value_type](effect.mem.malloc(u64(capacity * sizeof(value_type))))
            m.capacity = capacity
            m.size = 0
            m.growth_left = Table.growth_limit(capacity)
            for position in Table.slots(old_ctrl, old_capacity):
                index: size_type = position[2]
                _FlatHashMap._insert_no_grow(m, old_keys[index], old_values[index])
            if old_capacity > 0:
                effect.mem.free(ptr[void](old_ctrl))
                effect.mem.free(ptr[void](old_keys))
                effect.mem.free(ptr[void](old_values))

        def insert(m: ptr[_FlatHashMap], key: key_type, value: value_type) -> bool:
            if m.growth_left == 0:
                _FlatHashMap._rehash(m, Table.next_capacity(m.capacity))
            return _FlatHashMap._insert_no_grow(m, key, value)

        def erase(m: ptr[_FlatHashMap], key: key_type) -> bool:
            return Table.erase(m.ctrl, m.keys, m.capacity, ptr(m.size), key)

        def items(m: ptr[_FlatHashMap]) -> Item:
            for position in Table.slots(m.ctrl, m.capacity):
                index: size_type = position[2]
                yield m.keys[index], m.values[index]

    return _FlatHashMap
