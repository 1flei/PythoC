"""
Unit tests for forward reference handling.
"""

import unittest

from llvmlite import ir

from pythoc import i32, i64, struct
from pythoc.decorators.structs import compile_dynamic_class
from pythoc.decorators import clear_registry
from pythoc.forward_ref import clear_forward_ref_state, is_type_defined, mark_type_defined


class TestForwardRefState(unittest.TestCase):
    """Test forward reference state management."""

    def setUp(self):
        clear_forward_ref_state()

    def tearDown(self):
        clear_forward_ref_state()

    def test_clear_registry_clears_forward_ref_state(self):
        """clear_registry should reset defined forward-ref types."""
        mark_type_defined("Node", object())
        self.assertTrue(is_type_defined("Node"))

        clear_registry()

        self.assertFalse(is_type_defined("Node"))

    def test_field_expression_waits_for_all_referenced_types(self):
        class PairHolder:
            pair: 'struct[LaterA, LaterB]'

        holder = compile_dynamic_class(PairHolder)
        mark_type_defined('LaterA', i32)
        with self.assertRaisesRegex(NameError, 'LaterB'):
            holder.get_llvm_type(ir.Context())
        mark_type_defined('LaterB', i64)
        lowered = holder.get_llvm_type(ir.Context())
        self.assertEqual(lowered.elements[0].elements,
                         (ir.IntType(32), ir.IntType(64)))


if __name__ == "__main__":
    unittest.main()
