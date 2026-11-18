"""Shared control-byte, probing and slot-state rules for flat hash tables."""

from pythoc import compile, ptr, i8, u8, u64, bool, struct, refined, assume


EMPTY = i8(-128)
DELETED = i8(-2)


def HashTable(key_type, size_type, hash_fn, eq_fn):
    """Generate operations over caller-owned control bytes and typed keys.

    Occupied slots are used immediately by container methods. Their predicate
    describes current table state and does not grant a borrow across mutation.
    """
    suffix = ('hash_table', key_type, size_type, hash_fn, eq_fn)
    Probe = struct['index': size_type, 'found': bool, 'tag': i8]
    Insert = struct['index': size_type, 'inserted': bool]

    @compile(suffix=(suffix, 'occupied'))
    def occupied(ctrl: ptr[i8], capacity: size_type, index: size_type) -> bool:
        if index >= capacity:
            return False
        return ctrl[index] >= 0

    Occupied = refined[occupied]

    @compile(suffix=suffix)
    class _HashTable:
        def growth_limit(capacity: size_type) -> size_type:
            return capacity - capacity / 8

        def next_capacity(capacity: size_type) -> size_type:
            if capacity == 0:
                return size_type(16)
            return capacity * 2

        def probe(ctrl: ptr[i8], keys: ptr[key_type], capacity: size_type,
                  key: key_type) -> Probe:
            if capacity == 0:
                return capacity, False, i8(0)
            hash_value: u64 = hash_fn(key)
            tag: i8 = i8(u8(hash_value & 0x7F))
            index: size_type = size_type(hash_value >> 7) % capacity
            deleted: size_type = capacity
            visited: size_type = 0
            while visited < capacity:
                control: i8 = ctrl[index]
                if control == EMPTY:
                    if deleted < capacity:
                        index = deleted
                    return index, False, tag
                if control == DELETED:
                    if deleted == capacity:
                        deleted = index
                elif control == tag:
                    if eq_fn(keys[index], key):
                        return index, True, tag
                index = (index + 1) % capacity
                visited += 1
            return deleted, False, tag

        def find(ctrl: ptr[i8], keys: ptr[key_type], capacity: size_type,
                 key: key_type) -> Occupied:
            position = _HashTable.probe(ctrl, keys, capacity, key)
            if position.found:
                yield assume(ctrl, capacity, position.index, occupied)

        def insert(ctrl: ptr[i8], keys: ptr[key_type], capacity: size_type,
                   size: ptr[size_type], growth_left: ptr[size_type],
                   key: key_type) -> Insert:
            position = _HashTable.probe(ctrl, keys, capacity, key)
            index: size_type = position.index
            if position.found or index == capacity:
                return index, False
            if ctrl[index] == EMPTY:
                growth_left[0] -= 1
            ctrl[index] = position.tag
            keys[index] = key
            size[0] += 1
            return index, True

        def erase(ctrl: ptr[i8], keys: ptr[key_type], capacity: size_type,
                  size: ptr[size_type], key: key_type) -> bool:
            for position in _HashTable.find(ctrl, keys, capacity, key):
                ctrl[position[2]] = DELETED
                size[0] -= 1
                return True
            return False

        def slots(ctrl: ptr[i8], capacity: size_type) -> Occupied:
            index: size_type = 0
            while index < capacity:
                if occupied(ctrl, capacity, index):
                    yield assume(ctrl, capacity, index, occupied)
                index += 1

    return _HashTable
