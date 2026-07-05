#!/usr/bin/env python3
"""Unit tests for the ``param`` compile-time parameter type."""

import unittest

from pythoc import param
from pythoc.builtin_entities.param import param as param_cls


class TestParamType(unittest.TestCase):
    """Basic tests for the param builtin entity."""

    def test_exported_from_pythoc(self):
        self.assertIs(param, param_cls)

    def test_is_param_type(self):
        self.assertTrue(param.is_param_type())
        self.assertTrue(getattr(param, '_is_param', False))

    def test_cannot_be_used_as_runtime_type(self):
        # ``param`` is only legal as a function parameter annotation; anywhere
        # else it should be rejected by the type resolver.
        self.assertFalse(param.can_be_type())

    def test_cannot_be_called(self):
        self.assertFalse(param.can_be_called())

    def test_no_llvm_type(self):
        self.assertIsNone(param.get_llvm_type())

    def test_no_size(self):
        self.assertIsNone(param.get_size_bytes())

    def test_name(self):
        self.assertEqual(param.get_name(), 'param')


if __name__ == "__main__":
    unittest.main()
