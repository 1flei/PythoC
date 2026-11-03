"""Projection and decay preserve aggregate proofs through readonly views."""

import unittest

from pythoc import array, const, i32, ptr, refined, struct, volatile
from pythoc.ir_helpers import is_const, is_volatile, propagate_qualifiers
from pythoc.type_descriptor import array_decay_type, can_convert, same_type


class ProofAggregateViewsTest(unittest.TestCase):
    def test_record_projection_is_readonly_without_freezing_the_whole_binding(self):
        record = struct['count': i32]
        proof = refined[record, 'positive_count']
        self.assertTrue(same_type(propagate_qualifiers(proof, i32), const[i32]))
        self.assertFalse(is_const(proof))
        self.assertTrue(can_convert(proof, record))

    def test_qualifiers_survive_the_proof_projection(self):
        record = struct['count': i32]
        proof = const[refined[volatile[record], 'qualified']]
        field = propagate_qualifiers(proof, i32)
        self.assertTrue(is_const(field))
        self.assertTrue(is_volatile(field))

    def test_array_decay_cannot_remove_the_readonly_proof_view(self):
        proof = refined[array[i32, 2], 'array_checked']
        self.assertTrue(same_type(array_decay_type(proof), ptr[const[i32]]))
        self.assertTrue(can_convert(proof, ptr[const[i32]]))
        self.assertFalse(can_convert(proof, ptr[i32]))
        self.assertTrue(can_convert(proof, ptr[i32], mode='explicit'))

    def test_pointer_proof_does_not_constrain_pointee_storage(self):
        proof = refined[ptr[i32], 'nonnull']
        self.assertTrue(same_type(propagate_qualifiers(proof, i32), i32))


if __name__ == '__main__':
    unittest.main()
