"""Deferred callable runtime: decoration never builds the native runtime.

The PythoCCallable type lives in a compiled extension, so building it
needs a C toolchain.  AOT-only flows (compile_to_executable etc.) never
call their wrappers and must not pay that cost.  Decoration therefore
loads only an already-built runtime; otherwise the wrapper is a Python
facade that builds the runtime on the first call and then shares its
attribute state with the native object.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from pythoc import compile, i64
from pythoc import python_call


class TestDeferredCallable(unittest.TestCase):
    def test_decoration_does_not_build_runtime(self):
        real_ensure = python_call.ensure_callable_extension
        with mock.patch.object(
            python_call, '_cached_extension', return_value=None,
        ), mock.patch.object(
            python_call, 'ensure_callable_extension',
            side_effect=lambda: real_ensure(),
        ) as ensure:
            @compile(suffix='deferred_f')
            def f(x: i64) -> i64:
                return x + 1

            self.assertIsInstance(f, python_call._DeferredCallable)
            self.assertEqual(f.__name__, 'f')
            self.assertFalse(f.is_fast_bound())
            ensure.assert_not_called()

            self.assertEqual(f(41), 42)
            # The build happened only after the first call began.
            ensure.assert_called()
            self.assertTrue(f.is_fast_bound())

    def test_attributes_survive_bootstrap(self):
        with mock.patch.object(
            python_call, '_cached_extension', return_value=None,
        ):
            @compile(suffix='deferred_g')
            def g(x: i64) -> i64:
                return x * 2

            g.marker = 'before-call'
            self.assertEqual(g(21), 42)
            # The facade and the native object share one attribute dict.
            self.assertEqual(g.marker, 'before-call')
            native = g.__dict__['_native_self']
            self.assertIsNotNone(native)
            g.after = 1
            self.assertEqual(native.after, 1)

    def test_build_failure_surfaces_at_call(self):
        with mock.patch.object(
            python_call, '_cached_extension', return_value=None,
        ), mock.patch.object(
            python_call, 'ensure_callable_extension',
            side_effect=RuntimeError('no toolchain'),
        ):
            @compile(suffix='deferred_h')
            def h(x: i64) -> i64:
                return x

            with self.assertRaisesRegex(RuntimeError, 'no toolchain'):
                h(1)

    def test_cached_runtime_still_gives_native_wrapper(self):
        python_call.ensure_callable_extension()

        @compile(suffix='deferred_k')
        def k(x: i64) -> i64:
            return x + 2

        self.assertNotIsInstance(k, python_call._DeferredCallable)
        self.assertEqual(k(40), 42)
        self.assertTrue(k.is_fast_bound())


if __name__ == '__main__':
    unittest.main()
