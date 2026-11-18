"""Type-owned std APIs, checked access and shared custom-key hash tables."""

import unittest

from pythoc import compile, i32, u64, bool, ptr, refine, static, linear
from pythoc.std.linearize import linearize
from pythoc.std.map import FlatHashMap
from pythoc.std.refinement import positive
from pythoc.std.result import Option, Result
from pythoc.std.set import FlatHashSet
from pythoc.std.vector import Vector
from pythoc.type_descriptor import same_type


Positive = positive(i32)
O = Option(i32)
R = Result(i32, i32)
V = Vector(i32, 2)


@compile
class Key:
    value: i32


@compile
class Owned:
    token: linear


@compile
def colliding_hash(key: Key) -> u64:
    return u64(0)


@compile
def key_equal(left: Key, right: Key) -> bool:
    return left.value == right.value


M = FlatHashMap(Key, i32, hash_fn=colliding_hash, eq_fn=key_equal)
S = FlatHashSet(Key, hash_fn=colliding_hash, eq_fn=key_equal)


@compile
class Counts:
    released: static[i32] = 0


@compile
def acquire(value: i32) -> i32:
    return value


@compile
def release(value: i32) -> None:
    Counts.released += value


@compile
def other_release(value: i32) -> None:
    Counts.released += value * 10


Proof = linearize(acquire, release, name='SharedDisplayName')
OtherProof = linearize(acquire, other_release, name='SharedDisplayName')


@compile
def proof_methods() -> i32:
    Counts.released = 0
    value, proof = Proof.acquire(3)
    Proof.release(value, proof)
    other, token = OtherProof.acquire(4)
    OtherProof.release(other, token)
    return Counts.released


@compile
def refinement_method(value: i32) -> i32:
    for checked in refine(value, Positive.predicate):
        return checked
    return -1


@compile
def vector_access() -> i32:
    vector: V
    vp = ptr(vector)
    V.init(vp)
    V.push_back(vp, 3)
    V.push_back(vp, 4)
    V.push_back(vp, 5)
    total: i32 = 0
    for value in V.values(vp):
        total += value
    for element in V.at(vp, 1):
        element[0] = 9
    for element in V.at(vp, 3):
        total += 1000
    for element in V.at(vp, i32(-1)):
        total += 1000
    if V.truncate(vp, 4):
        total += 1000
    if not V.truncate(vp, 2):
        total += 1000
    for value in V.values(vp):
        total += value * 10
    V.destroy(vp)
    return total


@compile
def hash_access() -> i32:
    mapping: M
    values: S
    mp, sp = ptr(mapping), ptr(values)
    M.init(mp)
    S.init(sp)
    index: i32 = 0
    while index < 50:
        key: Key = (index,)
        M.insert(mp, key, index * 2)
        S.insert(sp, key)
        index += 1
    removed: Key = (17,)
    M.erase(mp, removed)
    S.erase(sp, removed)
    total: i32 = 0
    for value in M.at(mp, removed):
        total += 10000
    for found_key in S.at(sp, removed):
        total += 10000
    for item in M.items(mp):
        total += item.value
    for entry in S.values(sp):
        total += entry.value
    replacement: Key = (99,)
    M.insert(mp, replacement, 7)
    S.insert(sp, replacement)
    for value in M.at(mp, replacement):
        value[0] = 11
        total += value[0]
    M.clear(mp)
    S.clear(sp)
    for item in M.items(mp):
        total += 10000
    for entry in S.values(sp):
        total += 10000
    M.destroy(mp)
    S.destroy(sp)
    return total


@compile
def tombstone_budget() -> i32:
    values: S
    sp = ptr(values)
    S.init(sp)
    key: Key = (3,)
    S.insert(sp, key)
    S.erase(sp, key)
    before: i32 = i32(values.growth_left)
    S.insert(sp, key)
    after: i32 = i32(values.growth_left)
    S.destroy(sp)
    return before * 100 + after


class StdTypeInterfacesTest(unittest.TestCase):
    def test_factory_returns_the_annotatable_type(self):
        self.assertTrue(O.is_enum_type())
        self.assertTrue(R.is_enum_type())
        self.assertTrue(callable(Positive.predicate))
        self.assertFalse(hasattr(O, 'type'))

    def test_refinement_predicate_is_a_type_method(self):
        self.assertEqual(refinement_method(3), 3)
        self.assertEqual(refinement_method(0), -1)

    def test_resource_pair_defines_proof_identity(self):
        self.assertFalse(same_type(Proof, OtherProof))
        self.assertEqual(proof_methods(), 43)

    def test_checked_vector_access_and_iteration(self):
        self.assertEqual(vector_access(), 132)

    def test_custom_keys_rehash_lookup_iteration_and_clear(self):
        self.assertEqual(hash_access(), (1225 - 17) * 3 + 11)

    def test_tombstone_reuse_preserves_growth_budget(self):
        self.assertEqual(tombstone_budget(), 1313)

    def test_custom_key_functions_are_validated_at_initialization(self):
        with self.assertRaisesRegex(TypeError, 'together'):
            FlatHashSet(Key, hash_fn=colliding_hash)
        with self.assertRaisesRegex(TypeError, 'incompatible signatures'):
            FlatHashMap(Key, i32, hash_fn=key_equal, eq_fn=key_equal)

    def test_unimplemented_element_ownership_is_rejected_at_initialization(self):
        with self.assertRaisesRegex(TypeError, 'copyable'):
            Vector(Owned)
        with self.assertRaisesRegex(TypeError, 'copyable'):
            FlatHashMap(i32, Owned)


if __name__ == '__main__':
    unittest.main()
