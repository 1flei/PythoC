"""
Unit tests for native executor caching behavior.
"""

import json
import os
import tempfile
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

    def test_darwin_explicit_link_libraries_include_dependencies(self):
        """Darwin links dependent DSOs directly to avoid system symbol clashes."""
        executor = MultiSOExecutor()
        dependencies = [
            ("/tmp/dependency.py", "/tmp/dependency.so"),
            ("/tmp/self.py", "/tmp/main.so"),
        ]

        with patch("pythoc.native_executor.sys.platform", "darwin"), patch.object(
            executor,
            "_get_persisted_link_libraries",
            return_value=["pthread", "/tmp/dependency.so"],
        ), patch(
            "pythoc.native_executor.os.path.exists",
            return_value=True,
        ):
            libs, task_deps = executor._dependency_link_plan(
                "/tmp/main.o",
                "/tmp/main.so",
                dependencies,
            )

        self.assertEqual(libs, ["/tmp/dependency.so", "pthread"])
        self.assertEqual(task_deps, ())

    def test_darwin_link_plan_orders_pending_dependencies(self):
        """Darwin waits for pending dependent DSOs before linking consumers."""
        executor = MultiSOExecutor()

        with patch("pythoc.native_executor.sys.platform", "darwin"), patch.object(
            executor,
            "_get_persisted_link_libraries",
            return_value=[],
        ):
            libs, task_deps = executor._dependency_link_plan(
                "/tmp/main.o",
                "/tmp/main.so",
                [("/tmp/dependency.py", "/tmp/dependency.so")],
                pending_task_ids={
                    "/tmp/main.so": "link-main",
                    "/tmp/dependency.so": "link-dependency",
                },
                pending_dependency_graph={
                    "/tmp/main.so": ["/tmp/dependency.so"],
                    "/tmp/dependency.so": [],
                },
            )

        self.assertEqual(libs, ["/tmp/dependency.so"])
        self.assertEqual(task_deps, ("link-dependency",))

    def test_darwin_link_plan_skips_cycle_edges(self):
        """Darwin keeps dynamic lookup on cycle edges to bootstrap clean builds."""
        executor = MultiSOExecutor()

        with patch("pythoc.native_executor.sys.platform", "darwin"), patch.object(
            executor,
            "_get_persisted_link_libraries",
            return_value=[],
        ):
            libs, task_deps = executor._dependency_link_plan(
                "/tmp/a.o",
                "/tmp/a.so",
                [("/tmp/b.py", "/tmp/b.so")],
                pending_task_ids={
                    "/tmp/a.so": "link-a",
                    "/tmp/b.so": "link-b",
                },
                pending_dependency_graph={
                    "/tmp/a.so": ["/tmp/b.so"],
                    "/tmp/b.so": ["/tmp/a.so"],
                },
            )

        self.assertEqual(libs, [])
        self.assertEqual(task_deps, ())

    def test_linux_keeps_dynamic_dependency_loading(self):
        """Linux keeps the existing runtime dependency loading behavior."""
        executor = MultiSOExecutor()

        with patch("pythoc.native_executor.sys.platform", "linux"):
            libs, task_deps = executor._dependency_link_plan(
                "/tmp/main.o",
                "/tmp/main.so",
                [("/tmp/dependency.py", "/tmp/dependency.so")],
            )

        self.assertIsNone(libs)
        self.assertEqual(task_deps, ())

    def test_source_embed_dependency_is_not_link_dependency(self):
        """source_embed deps are compile-time edges and must not be linked."""
        executor = MultiSOExecutor()

        with tempfile.TemporaryDirectory() as tmpdir:
            main_source = os.path.join(tmpdir, "main.py")
            embed_source = os.path.join(tmpdir, "embed.py")
            callee_source = os.path.join(tmpdir, "callee.py")

            main_so = os.path.join(tmpdir, f"main{get_shared_lib_extension()}")
            main_obj = main_so.replace(get_shared_lib_extension(), ".o")
            deps_file = main_obj.replace(".o", ".deps")

            # Files outside cwd are placed under build/external/ by
            # _derive_so_file_from_group_key, so compute the expected path.
            callee_so_expected = os.path.join(
                "build", "external", f"callee{get_shared_lib_extension()}"
            )

            deps_data = {
                "version": 11,
                "source_mtime": 0.0,
                "link_objects": [],
                "link_libraries": [],
                "group_keys": [
                    [main_source, None, None, None],
                    [embed_source, None, None, None],
                    [callee_source, None, None, None],
                ],
                "main_group_idx": 0,
                "group_dependencies": [
                    {"dependency_type": "source_embed", "target_group_idx": 1},
                    {"dependency_type": "function_call", "target_group_idx": 2},
                ],
            }
            with open(deps_file, "w") as f:
                json.dump(deps_data, f)

            dependencies = executor._get_library_dependencies(main_source, main_so)

        self.assertEqual(dependencies, [(callee_source, callee_so_expected)])

    def test_windows_stub_mode_ignores_source_embed_target(self):
        """On Windows, source_embed targets must not appear in the link command."""
        executor = MultiSOExecutor()

        with tempfile.TemporaryDirectory() as tmpdir:
            main_source = os.path.join(tmpdir, "main.py")
            embed_source = os.path.join(tmpdir, "embed.py")
            main_so = os.path.join(tmpdir, "main.dll")
            main_obj = main_so.replace(".dll", ".o")
            deps_file = main_obj.replace(".o", ".deps")
            embed_so = os.path.join(tmpdir, "embed.dll")

            deps_data = {
                "version": 11,
                "source_mtime": 0.0,
                "link_objects": [],
                "link_libraries": [],
                "group_keys": [
                    [main_source, None, None, None],
                    [embed_source, None, None, None],
                ],
                "main_group_idx": 0,
                "group_dependencies": [
                    {"dependency_type": "source_embed", "target_group_idx": 1},
                ],
            }
            with open(deps_file, "w") as f:
                json.dump(deps_data, f)

            with patch("pythoc.native_executor.sys.platform", "win32"):
                libs, task_deps = executor._dependency_link_plan(
                    main_obj,
                    main_so,
                    executor._get_library_dependencies(main_source, main_so),
                )

        self.assertEqual(libs, [])
        self.assertEqual(task_deps, ())


if __name__ == "__main__":
    unittest.main()
