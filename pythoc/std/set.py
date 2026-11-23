"""Flat hash sets with a shared probing and slot-state implementation."""

from __future__ import annotations

import sys

from pythoc import (
    compile, effect, ptr, u64, i8, bool, const, void, nullptr, sizeof, refine,
)
from pythoc.libc.string import memset
from pythoc.logger import logger
from pythoc.schema_protocol import is_linear_schema_type
from pythoc.source import SourceLocation
from pythoc.std.mem import DefaultMem
from pythoc.std.hash import _key_functions
from pythoc.std._hash_table import HashTable, EMPTY
from pythoc.std.refinement import nonnull


effect.default(mem=DefaultMem)


def FlatHashSet(key_type, size_type=u64, *, hash_fn=None, eq_fn=None):
    """Return a set type with fixed compile-time hash/equality operations."""
    node = logger.current_location() or SourceLocation.from_frame(sys._getframe(1))
    hash_fn, eq_fn = _key_functions(key_type, hash_fn, eq_fn, node)
    if not (getattr(size_type, '_is_integer', False)
            and not getattr(size_type, '_is_signed', True)):
        logger.error('FlatHashSet size_type must be an unsigned integer type',
                     node=node, exc_type=TypeError)
    if is_linear_schema_type(key_type):
        logger.error('FlatHashSet requires copyable keys', node=node, exc_type=TypeError)
    Table = HashTable(key_type, size_type, hash_fn, eq_fn)
    KeyPointer = ptr[const[key_type]]
    PresentKey = nonnull(KeyPointer)

    @compile(suffix=(key_type, size_type, hash_fn, eq_fn))
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
                effect.mem.free(ptr[void](s.ctrl))
                effect.mem.free(ptr[void](s.slots))
            _FlatHashSet.init(s)

        def size(s: ptr[_FlatHashSet]) -> size_type:
            return s.size

        def capacity(s: ptr[_FlatHashSet]) -> size_type:
            return s.capacity

        def clear(s: ptr[_FlatHashSet]) -> None:
            if s.capacity > 0:
                memset(s.ctrl, EMPTY, s.capacity)
            s.size = 0
            s.growth_left = Table.growth_limit(s.capacity)

        def find(s: ptr[_FlatHashSet], key: key_type) -> KeyPointer:
            for position in Table.find(s.ctrl, s.slots, s.capacity, key):
                return ptr(s.slots[position[2]])
            return nullptr

        def contains(s: ptr[_FlatHashSet], key: key_type) -> bool:
            return _FlatHashSet.find(s, key) != nullptr

        def at(s: ptr[_FlatHashSet], key: key_type) -> PresentKey:
            """Yield a read-only key pointer, valid until an invalidating mutation."""
            for present in refine(_FlatHashSet.find(s, key), PresentKey.predicate):
                yield present

        def _insert_no_grow(s: ptr[_FlatHashSet], key: key_type) -> bool:
            position = Table.insert(s.ctrl, s.slots, s.capacity,
                                    ptr(s.size), ptr(s.growth_left), key)
            return position.inserted

        def _rehash(s: ptr[_FlatHashSet], capacity: size_type) -> None:
            old_ctrl = s.ctrl
            old_slots = s.slots
            old_capacity = s.capacity
            s.ctrl = ptr[i8](effect.mem.malloc(u64(capacity)))
            memset(s.ctrl, EMPTY, capacity)
            s.slots = ptr[key_type](effect.mem.malloc(u64(capacity * sizeof(key_type))))
            s.capacity = capacity
            s.size = 0
            s.growth_left = Table.growth_limit(capacity)
            for position in Table.slots(old_ctrl, old_capacity):
                index: size_type = position[2]
                _FlatHashSet._insert_no_grow(s, old_slots[index])
            if old_capacity > 0:
                effect.mem.free(ptr[void](old_ctrl))
                effect.mem.free(ptr[void](old_slots))

        def insert(s: ptr[_FlatHashSet], key: key_type) -> bool:
            if s.growth_left == 0:
                _FlatHashSet._rehash(s, Table.next_capacity(s.capacity))
            return _FlatHashSet._insert_no_grow(s, key)

        def erase(s: ptr[_FlatHashSet], key: key_type) -> bool:
            return Table.erase(s.ctrl, s.slots, s.capacity, ptr(s.size), key)

        def values(s: ptr[_FlatHashSet]) -> key_type:
            for position in Table.slots(s.ctrl, s.capacity):
                yield s.slots[position[2]]

    return _FlatHashSet
