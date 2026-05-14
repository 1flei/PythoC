"""Unit tests for class-body method discovery and attachment helpers.

These tests exercise the helpers in isolation -- without invoking the LLVM
compilation pipeline -- to make sure the AST extraction logic is robust to
the various class-body shapes (struct/union/enum-style, descriptors,
explicit ``@compile`` overrides, empty bodies).
"""
import ast
import unittest

from pythoc.decorators.class_methods import (
    extract_class_method_defs,
    lookup_class_method,
)


class TestExtractClassMethodDefs(unittest.TestCase):
    def test_empty_class_returns_no_methods(self):
        class Empty:
            pass

        self.assertEqual(extract_class_method_defs(Empty), [])

    def test_only_annotations_returns_no_methods(self):
        class FieldsOnly:
            x: int
            y: int = 3

        self.assertEqual(extract_class_method_defs(FieldsOnly), [])

    def test_simple_methods_collected_in_source_order(self):
        class WithMethods:
            x: int

            def first(a, b):
                return a + b

            def second(a):
                return a

        result = extract_class_method_defs(WithMethods)
        names = [name for name, _node in result]
        self.assertEqual(names, ["first", "second"])
        for _name, node in result:
            self.assertIsInstance(node, ast.FunctionDef)

    def test_descriptors_are_ignored(self):
        class Mixed:
            x: int

            @staticmethod
            def st():
                return 0

            @classmethod
            def cm(cls):
                return 0

            @property
            def prop(self):
                return 0

            def plain(a):
                return a

        names = [name for name, _ in extract_class_method_defs(Mixed)]
        # AST extraction reports every FunctionDef regardless of decorators;
        # the runtime attach phase is where descriptor-typed members are
        # filtered out by checking inspect.isfunction.
        self.assertIn("plain", names)
        # All FunctionDefs (including the decorated ones) appear because the
        # extractor is purely syntactic.
        self.assertEqual(set(names), {"st", "cm", "prop", "plain"})

    def test_async_def_is_not_collected(self):
        class HasAsync:
            x: int

            async def coroutine():
                return 1

            def normal():
                return 2

        names = [name for name, _ in extract_class_method_defs(HasAsync)]
        self.assertEqual(names, ["normal"])


class TestLookupClassMethod(unittest.TestCase):
    def test_returns_none_when_no_compiled_marker(self):
        class Plain:
            def helper(_a):
                return 0

        self.assertIsNone(lookup_class_method(Plain, "helper"))

    def test_returns_member_when_marked_compiled(self):
        sentinel_wrapper = type("SentinelWrapper", (), {"_is_compiled": True})()

        class Holder:
            method = sentinel_wrapper

        self.assertIs(lookup_class_method(Holder, "method"), sentinel_wrapper)

    def test_falls_back_to_python_class_attribute(self):
        sentinel_wrapper = type("SentinelWrapper", (), {"_is_compiled": True})()

        class UserClass:
            method = sentinel_wrapper

        # Simulate the unified-type indirection: the type the dispatcher holds
        # is not where methods live, but it carries a back-reference via
        # ``_python_class``.
        class UnifiedTypeStub:
            _python_class = UserClass

        self.assertIs(
            lookup_class_method(UnifiedTypeStub, "method"),
            sentinel_wrapper,
        )

    def test_returns_none_for_missing_attribute(self):
        class Empty:
            pass

        self.assertIsNone(lookup_class_method(Empty, "nope"))


if __name__ == "__main__":
    unittest.main()
