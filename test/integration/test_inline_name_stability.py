#!/usr/bin/env python3
"""
The ``inline`` decorator must survive the inline-machinery subpackage import.

Compiling any function whose body contains ``yield`` lazily imports the
inline-machinery subpackage.  When that subpackage was named
``pythoc.inline``, Python's import machinery set the ``inline`` attribute
on the pythoc package to the *module object*, silently overwriting the
``inline`` decorator exported from pythoc.decorators -- any later
``from pythoc import inline`` then crashed with
``TypeError: 'module' object is not callable``.

The subpackage now lives at ``pythoc._inline``, so the two names never
collide.  This test fails on the old layout: importing the subpackage
there rebinds the decorator attribute to a module.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pythoc
from pythoc import compile, i32


@compile(suffix="inl_name_probe")
def _gen_probe() -> i32:
    """A yield body: flushing this module's compiled functions exercises
    the same lazy subpackage import this test triggers directly."""
    yield 1


class TestInlineNameStability(unittest.TestCase):
    def test_decorator_survives_subpackage_import(self):
        import pythoc._inline  # noqa: F401  (what yield compilation imports)
        from pythoc import inline
        self.assertTrue(callable(inline))
        self.assertNotIsInstance(inline, type(sys))

    def test_package_attribute_is_not_rebound(self):
        import pythoc._inline  # noqa: F401
        self.assertNotIsInstance(pythoc.inline, type(sys))

    def test_decorator_usable_after_import(self):
        import pythoc._inline  # noqa: F401
        from pythoc import inline

        @inline
        def helper(a: i32) -> i32:
            return a + i32(1)

        self.assertIsNotNone(helper)


if __name__ == "__main__":
    unittest.main()
