"""Deferred Python callables preserve state until runtime creation succeeds."""

from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

from pythoc.python_call import _DeferredCallable


class TestDeferredCallable(unittest.TestCase):
    def test_concurrent_bootstrap_publishes_one_native_callable(self):
        wrapper = _DeferredCallable(lambda: 42)
        entered = threading.Event()
        proceed = threading.Event()
        second_started = threading.Event()
        second_finished = threading.Event()
        results = []
        errors = []
        native = SimpleNamespace()

        def ensure():
            entered.set()
            if not proceed.wait(5):
                raise RuntimeError("runtime build wait timed out")
            return object()

        def invoke(second=False):
            try:
                if second:
                    second_started.set()
                results.append(wrapper._bootstrap())
            except BaseException as error:
                errors.append(error)
            finally:
                if second:
                    second_finished.set()

        first = threading.Thread(target=invoke)
        second = threading.Thread(target=invoke, args=(True,))
        with patch("pythoc.python_call.ensure_callable_extension",
                   side_effect=ensure) as build, patch(
                       "pythoc.python_call._native_callable",
                       return_value=native) as create:
            try:
                first.start()
                self.assertTrue(entered.wait(5))
                second.start()
                self.assertTrue(second_started.wait(5))
                self.assertFalse(second_finished.wait(0.1))
            finally:
                proceed.set()
                first.join(5)
                if second.ident is not None:
                    second.join(5)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(results, [native, native])
        build.assert_called_once()
        create.assert_called_once()

    def test_failed_bootstrap_can_retry_without_losing_compile_metadata(self):
        def function(value: int) -> int:
            return value

        wrapper = _DeferredCallable(function)
        wrapper._binding = object()
        native = SimpleNamespace()
        error = RuntimeError("runtime build failed")
        extension = object()
        with patch("pythoc.python_call.ensure_callable_extension",
                   side_effect=[error, extension]), patch(
                       "pythoc.python_call._native_callable",
                       return_value=native) as create:
            with self.assertRaisesRegex(RuntimeError, "runtime build failed"):
                wrapper._bootstrap()
            self.assertIs(wrapper._bootstrap(), native)
            self.assertIs(wrapper._bootstrap(), native)
        create.assert_called_once_with(extension, function)
        self.assertIs(wrapper.__dict__, native.__dict__)
        self.assertIs(native._binding, wrapper._binding)
        self.assertEqual(native.__name__, "function")


if __name__ == "__main__":
    unittest.main()
