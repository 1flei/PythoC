"""
Unit tests for forward reference handling.
"""

import unittest
from unittest.mock import patch

from pythoc.builtin_entities.composite_base import CompositeType
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

    def test_forward_ref_callbacks_bind_each_referenced_type(self):
        """Each callback should update the matching unresolved name."""
        callbacks = {}
        seen_namespaces = []

        class DummyUnifiedType:
            _field_types = [None]

        class DummyTarget:
            _struct_fields = [("field", "tuple[A, B]")]

        class FakeResolver:
            def __init__(self, user_globals):
                self.user_globals = user_globals

            def parse_annotation(self, _annotation):
                seen_namespaces.append(dict(self.user_globals))
                return ("parsed", dict(self.user_globals))

        def capture_callback(type_name, callback):
            callbacks[type_name] = callback

        with patch(
            "pythoc.forward_ref.register_forward_ref_callback",
            side_effect=capture_callback,
        ), patch("pythoc.type_resolver.TypeResolver", FakeResolver):
            CompositeType._setup_forward_ref_callbacks(
                DummyUnifiedType,
                DummyTarget,
                ["tuple[A, B]"],
                {},
            )

        self.assertIn("A", callbacks)
        self.assertIn("B", callbacks)

        callbacks["A"]("resolved_a")
        self.assertEqual(seen_namespaces[-1]["A"], "resolved_a")
        self.assertNotIn("B", seen_namespaces[-1])

        callbacks["B"]("resolved_b")
        self.assertEqual(seen_namespaces[-1]["B"], "resolved_b")
        self.assertNotIn("A", seen_namespaces[-1])


if __name__ == "__main__":
    unittest.main()
