"""
Unit tests for qualified struct keys in the unified registry.
"""

import unittest

from pythoc.registry import StructInfo, UnifiedCompilationRegistry


def _make_class(module, name):
    """Create a class object with a controlled module/qualname."""
    return type(name, (), {'__module__': module})


class TestQualifiedStructKeys(unittest.TestCase):
    """Structs with the same bare name from different modules coexist."""

    def setUp(self):
        self.registry = UnifiedCompilationRegistry()

    def test_same_bare_name_different_modules(self):
        cls_a = _make_class("mod_a", "Node")
        cls_b = _make_class("mod_b", "Node")

        info_a = self.registry.register_struct_from_fields(
            "Node", [("x", "i32")], python_class=cls_a)
        info_b = self.registry.register_struct_from_fields(
            "Node", [("y", "f64"), ("z", "f64")], python_class=cls_b)

        self.assertIsNot(info_a, info_b)
        self.assertEqual(info_a.qualified_name, "mod_a.Node")
        self.assertEqual(info_b.qualified_name, "mod_b.Node")

        # Qualified lookups hit the correct entries
        self.assertIs(self.registry.get_struct("mod_a.Node"), info_a)
        self.assertIs(self.registry.get_struct("mod_b.Node"), info_b)
        self.assertEqual(
            self.registry.get_struct("mod_a.Node").get_field_names(), ["x"])
        self.assertEqual(
            self.registry.get_struct("mod_b.Node").get_field_names(), ["y", "z"])

        # Bare-name fallback still resolves (insertion order decides ties)
        self.assertIs(self.registry.get_struct("Node"), info_a)
        self.assertTrue(self.registry.has_struct("Node"))
        self.assertTrue(self.registry.has_struct("mod_b.Node"))

    def test_in_place_update_for_same_class(self):
        cls_a = _make_class("mod_a", "Node")

        info_a = self.registry.register_struct_from_fields(
            "Node", [("x", "i32")], python_class=cls_a)
        updated = self.registry.register_struct_from_fields(
            "Node", [("x", "i32"), ("y", "i32")], python_class=cls_a)

        # Same entry updated in place, no duplicate created
        self.assertIs(updated, info_a)
        self.assertEqual(info_a.get_field_names(), ["x", "y"])
        self.assertEqual(len(self.registry.list_structs()), 1)

    def test_bare_key_without_python_class(self):
        info = self.registry.register_struct_from_fields(
            "Plain", [("x", "i32")])

        self.assertEqual(info.qualified_name, "Plain")
        self.assertIs(self.registry.get_struct("Plain"), info)
        self.assertTrue(self.registry.has_struct("Plain"))

    def test_register_struct_sets_qualified_name(self):
        cls_a = _make_class("mod_a", "Point")
        info = StructInfo(name="Point", fields=[("x", "i32")], python_class=cls_a)
        self.registry.register_struct(info)

        self.assertEqual(info.qualified_name, "mod_a.Point")
        self.assertIs(self.registry.get_struct("mod_a.Point"), info)
        self.assertIs(self.registry.get_struct("Point"), info)

    def test_missing_struct(self):
        self.assertIsNone(self.registry.get_struct("Nope"))
        self.assertFalse(self.registry.has_struct("Nope"))


if __name__ == "__main__":
    unittest.main()
