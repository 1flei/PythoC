"""
Unit tests for native executor caching behavior.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pythoc.native_executor import MultiSOExecutor
from pythoc.utils.link_utils import get_shared_lib_extension


class DummyWrapper:
    """Simple wrapper object for executor tests."""


class TestMultiSOExecutor(unittest.TestCase):
    """Test executor cache and relink behavior."""

    def _make_wrapper(self, func_name="foo"):
        wrapper = DummyWrapper()
        wrapper._state = SimpleNamespace(
            source_file="/tmp/test_module.py",
            so_file=f"/tmp/test_module{get_shared_lib_extension()}",
            compiler=object(),
            original_name=func_name,
            actual_func_name=func_name,
        )
        return wrapper

    def test_execute_function_uses_shared_library_cache_key(self):
        """execute_function should reuse wrappers cached by get_function."""
        executor = MultiSOExecutor()
        wrapper = self._make_wrapper()
        so_file = wrapper._state.so_file

        executor.loaded_libs[so_file] = object()
        executor.function_cache[f"{so_file}:foo"] = "cached_wrapper"

        with patch(
            "pythoc.build.flush_all_pending_outputs",
            side_effect=AssertionError("cache miss"),
        ):
            result = executor.execute_function(wrapper)

        self.assertEqual(result, "cached_wrapper")

    def test_relink_clears_cached_functions_for_same_library(self):
        """Re-linking should invalidate every cached function from that .so."""
        executor = MultiSOExecutor()
        wrapper = self._make_wrapper(func_name="fresh_func")
        so_file = wrapper._state.so_file
        obj_file = so_file.replace(get_shared_lib_extension(), ".o")

        executor.loaded_libs[so_file] = object()
        executor.function_cache[f"{so_file}:stale_func"] = "stale_wrapper"

        def fake_exists(path):
            return path in {obj_file}

        with patch(
            "pythoc.build.flush_all_pending_outputs",
            return_value=None,
        ), patch.object(
            executor,
            "_get_dependencies",
            return_value=[],
        ), patch(
            "pythoc.native_executor.BuildCache.check_so_needs_relink",
            return_value=True,
        ), patch(
            "pythoc.native_executor.os.path.exists",
            side_effect=fake_exists,
        ), patch.object(
            executor,
            "compile_source_to_so",
            return_value=so_file,
        ), patch.object(
            executor,
            "load_library_with_dependencies",
            return_value=None,
        ), patch.object(
            executor,
            "get_function",
            return_value="fresh_wrapper",
        ):
            result = executor.execute_function(wrapper)

        self.assertEqual(result, "fresh_wrapper")
        self.assertNotIn(f"{so_file}:stale_func", executor.function_cache)


if __name__ == "__main__":
    unittest.main()
