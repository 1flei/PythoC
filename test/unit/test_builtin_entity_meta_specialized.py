"""
Unit tests for BuiltinEntityMeta pending-registration behavior.

Specialized type products (ptr[T], const[T], array[T, N], struct[...])
are created dynamically on every subscript evaluation; they must not
accumulate in the metaclass pending-registration list.
"""

import unittest

from pythoc.builtin_entities import array, const, i32, f64, ptr, struct
from pythoc.builtin_entities.base import BuiltinEntityMeta
from pythoc.registry import get_unified_registry


def _pending_count():
    return len(getattr(BuiltinEntityMeta, '_pending_registrations', []))


class TestSpecializedTypesNotRegistered(unittest.TestCase):
    """Evaluating type subscripts must not grow the pending list."""

    def test_ptr_specialization_does_not_grow_pending(self):
        before = _pending_count()
        for _ in range(5):
            ptr[i32]
        self.assertEqual(_pending_count(), before)

    def test_const_specialization_does_not_grow_pending(self):
        before = _pending_count()
        for _ in range(5):
            const[i32]
        self.assertEqual(_pending_count(), before)

    def test_array_specialization_does_not_grow_pending(self):
        before = _pending_count()
        for _ in range(5):
            array[i32, 4]
        self.assertEqual(_pending_count(), before)

    def test_struct_specialization_does_not_grow_pending(self):
        before = _pending_count()
        for _ in range(5):
            struct[i32, f64]
        self.assertEqual(_pending_count(), before)

    def test_specialized_class_carries_marker(self):
        specialized = ptr[i32]
        self.assertTrue(getattr(specialized, '_pc_specialized', False))

    def test_import_time_entities_still_registered(self):
        registry = get_unified_registry()
        self.assertIs(registry.get_builtin_entity("i32"), i32)
        self.assertIs(registry.get_builtin_entity("ptr"), ptr)
        self.assertIs(registry.get_builtin_entity("const"), const)
        self.assertIs(registry.get_builtin_entity("array"), array)
        self.assertIs(registry.get_builtin_entity("struct"), struct)


if __name__ == "__main__":
    unittest.main()
