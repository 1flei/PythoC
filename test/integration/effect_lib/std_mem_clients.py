"""Container clients instantiated under the caller's memory binding."""

from types import SimpleNamespace

from pythoc import compile, i32, i64, ptr
from pythoc.std.map import FlatHashMap
from pythoc.std.set import FlatHashSet
from pythoc.std.vector import Vector


def make_clients():
    SmallVector = Vector(i32, 4)
    HeapVector = Vector(i32, 0)
    IntSet = FlatHashSet(i32)
    IntMap = FlatHashMap(i32, i64)

    @compile
    def small_vector(count: i32) -> i64:
        v: SmallVector
        SmallVector.init(ptr(v))
        index: i32 = 0
        while index < count:
            SmallVector.push_back(ptr(v), index * 3 + 1)
            index += 1
        total: i64 = 0
        index = 0
        while index < count:
            total += i64(SmallVector.get(ptr(v), index))
            index += 1
        SmallVector.destroy(ptr(v))
        return total

    @compile
    def heap_vector(count: i32) -> i64:
        v: HeapVector
        HeapVector.init(ptr(v))
        index: i32 = 0
        while index < count:
            HeapVector.push_back(ptr(v), index * 3 + 1)
            index += 1
        total: i64 = 0
        index = 0
        while index < count:
            total += i64(HeapVector.get(ptr(v), index))
            index += 1
        HeapVector.destroy(ptr(v))
        return total

    @compile
    def hash_set(count: i32) -> i64:
        s: IntSet
        IntSet.init(ptr(s))
        index: i32 = 0
        while index < count:
            IntSet.insert(ptr(s), index)
            index += 1
        total: i64 = 0
        index = 0
        while index < count:
            if IntSet.contains(ptr(s), index):
                total += i64(index)
            index += 1
        IntSet.clear(ptr(s))
        IntSet.insert(ptr(s), 7)
        if not IntSet.erase(ptr(s), 7):
            total = -1
        IntSet.destroy(ptr(s))
        return total

    @compile
    def hash_map(count: i32) -> i64:
        m: IntMap
        IntMap.init(ptr(m))
        index: i32 = 0
        while index < count:
            IntMap.insert(ptr(m), index, i64(index * 3 + 1))
            index += 1
        total: i64 = 0
        index = 0
        while index < count:
            total += IntMap.find(ptr(m), index)[0]
            index += 1
        IntMap.clear(ptr(m))
        IntMap.insert(ptr(m), 7, i64(11))
        if not IntMap.erase(ptr(m), 7):
            total = -1
        IntMap.destroy(ptr(m))
        return total

    return SimpleNamespace(
        small_vector=small_vector, heap_vector=heap_vector,
        hash_set=hash_set, hash_map=hash_map,
    )
