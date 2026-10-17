"""Link closure, target isolation, and frozen dependency tests."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import pythoc
from pythoc.artifact import (
    ArtifactKind, ArtifactPlan, LinkPlan, LinkScope, plan_artifact_tasks,
)
from pythoc.artifact.link_plan import is_internal_group
from pythoc.build.deps import GroupDependency, GroupDeps, GroupKey
from pythoc.utils.build_utils import compile_to_executable, link_executable


class TestLinkClosure(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = ("/src/root.py", None, None, None)
        self.child = ("/src/child.py", None, None, None)
        self.groups = {}
        self.deps = {}
        self.manager = SimpleNamespace(get_all_groups=lambda: self.groups)
        self.tracker = SimpleNamespace(
            derive_obj_file_from_group_key=lambda key: None,
            get_deps=lambda key, **kwargs: self.deps.get(key),
            get_deps_for_group=lambda key: self.deps.get(key),
        )

    def object_file(self, name):
        path = Path(self.directory.name) / name
        path.touch()
        return str(path)

    def group(self, key, name, *, targets=(), objects=(), libraries=()):
        obj = self.object_file(name)
        self.groups[key] = {"obj_file": obj}
        self.deps[key] = GroupDeps(
            group_key=GroupKey.from_tuple(key),
            group_dependencies=[
                GroupDependency(GroupKey.from_tuple(target))
                for target in targets
            ],
            link_objects=list(objects), link_libraries=list(libraries),
        )
        return obj

    def plan(self, roots, **kwargs):
        return LinkPlan.from_group_roots(
            roots, flush=False, output_manager=self.manager,
            dependency_tracker=self.tracker, **kwargs,
        )

    def test_cycle_objects_and_libraries_are_deduplicated(self):
        extra = self.object_file("extra.o")
        root = self.group(
            self.root, "root.o", targets=[self.child],
            objects=[extra], libraries=["m"],
        )
        child = self.group(
            self.child, "child.o", targets=[self.root],
            objects=[extra], libraries=["m", "c"],
        )
        plan = self.plan([self.root, self.child])
        self.assertEqual(plan.obj_files, (root, extra, child))
        self.assertEqual(plan.link_libraries, ("m", "c"))

    def test_source_embed_does_not_require_a_link_object(self):
        obj = self.group(self.root, "root.o")
        self.deps[self.root].group_dependencies.append(
            GroupDependency(GroupKey.from_tuple(self.child), "source_embed")
        )
        self.assertEqual(self.plan([self.root]).obj_files, (obj,))

    def test_missing_required_dependency_is_not_silently_omitted(self):
        self.group(self.root, "root.o", targets=[self.child])
        with self.assertRaises(FileNotFoundError):
            self.plan([self.root])

    def test_missing_explicit_link_object_is_rejected(self):
        self.group(self.root, "root.o", objects=["/missing/required.o"])
        with self.assertRaisesRegex(FileNotFoundError, "required.o"):
            self.plan([self.root])

    def test_missing_dependency_metadata_is_rejected(self):
        self.group(self.root, "root.o")
        del self.deps[self.root]
        with self.assertRaisesRegex(RuntimeError, "dependency metadata"):
            self.plan([self.root])

    def test_only_host_adapters_are_internal(self):
        package = Path(pythoc.__file__).parent
        for filename in ("python_entry.py", "callable_type.py"):
            for suffix in ("", ".__parametric__.entry.i32"):
                key = (str(package / (filename + suffix)), None, None, None)
                self.assertTrue(is_internal_group(key))
        for path in (package / "c_helpers.py", package / "libc/errno.py",
                     Path("/src/python_entry.py")):
            self.assertFalse(is_internal_group((str(path), None, None, None)))

    def test_adapter_policy_only_applies_to_standalone_targets(self):
        adapter = (
            str(Path(pythoc.__file__).parent / "python_entry.py"),
            None, None, None,
        )
        obj = self.group(adapter, "adapter.o")
        self.assertEqual(self.plan([adapter]).obj_files, (obj,))
        with self.assertRaisesRegex(RuntimeError, "host Python adapter"):
            self.plan([adapter], standalone=True)

    def test_all_groups_collects_native_support_transitively(self):
        helper = (
            str(Path(pythoc.__file__).parent / "c_helpers.py"),
            None, None, None,
        )
        obj = self.group(helper, "helper.o", targets=[self.child])
        child = self.group(self.child, "child.o", libraries=["m"])
        adapter = (
            str(Path(pythoc.__file__).parent / "python_entry.py"),
            None, None, None,
        )
        self.group(adapter, "adapter.o")
        with patch("pythoc.build.output_manager.get_output_manager",
                   return_value=self.manager), patch(
                       "pythoc.build.deps.get_dependency_tracker",
                       return_value=self.tracker), patch(
                           "pythoc.registry.get_unified_registry") as registry:
            registry.return_value.get_link_objects.return_value = []
            registry.return_value.get_link_libraries.return_value = []
            plan = LinkPlan.from_all_groups(flush=False)
        self.assertEqual(plan.obj_files, (obj, child))
        self.assertEqual(plan.link_libraries, ("m",))
        self.assertIs(plan.scope, LinkScope.TRANSITIVE)

    def test_explicit_program_objects_and_libraries_are_frozen(self):
        root = self.group(self.root, "root.o")
        extra = self.object_file("imported-c.o")
        objects = [extra]
        libraries = ["needed"]
        plan = self.plan([self.root], link_objects=objects,
                         link_libraries=libraries)
        objects.clear()
        libraries.clear()
        self.assertEqual(plan.obj_files, (extra, root))
        self.assertEqual(plan.link_libraries, ("needed",))


class TestFrozenLinkInputs(unittest.TestCase):
    def test_python_entry_steps_serialize_shared_codegen_state(self):
        from pythoc.python_adapter import _compile_adapter_library

        def adapter_path(output, specs):
            return specs[0]["adapter"] + ".adapter.o"

        with patch("pythoc.python_adapter._adapter_object_path",
                   side_effect=adapter_path), patch(
                       "pythoc.artifact.build_artifact") as build:
            _compile_adapter_library(
                [{"adapter": "first"}, {"adapter": "second"}],
                "adapter.so", (), (),
            )
        steps = build.call_args.args[0].steps
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0].resources, ('python-entry-codegen',))
        self.assertEqual(steps[1].resources, steps[0].resources)

    def test_explicit_empty_inputs_never_read_registry_at_execution(self):
        plan = ArtifactPlan(
            kind=ArtifactKind.EXECUTABLE,
            link=LinkPlan(roots=(), obj_files=("root.o",)),
            output_path="main",
        )
        with patch("pythoc.utils.link_utils.try_link_with_linkers") as link:
            task = plan_artifact_tasks(plan)[0]
            task.run()
        self.assertEqual(link.call_args.kwargs["link_libraries"], [])
        self.assertEqual(link.call_args.kwargs["link_objects"], [])

    def test_compile_to_executable_preserves_full_link_plan(self):
        plan = LinkPlan(roots=(), obj_files=("root.o", "helper.o"),
                        link_libraries=("needed",))
        with patch.object(LinkPlan, "from_all_groups", return_value=plan), \
                patch("pythoc.artifact.build_artifact") as build:
            build.return_value.path = "main"
            self.assertEqual(compile_to_executable("main", "source.py"),
                             "main")
        self.assertIs(build.call_args.args[0].link, plan)

    def test_manual_executable_snapshots_registry_when_planning(self):
        with patch("pythoc.registry.get_unified_registry") as registry, \
                patch("pythoc.artifact.build_artifact") as build:
            registry.return_value.get_link_libraries.return_value = ["needed"]
            registry.return_value.get_link_objects.return_value = ["extra.o"]
            link_executable(["root.o"], "main")
            plan = build.call_args.args[0]
            registry.return_value.get_link_libraries.return_value = ["other"]
            registry.return_value.get_link_objects.return_value = ["other.o"]
            with patch("pythoc.utils.link_utils.try_link_with_linkers") as link:
                task = plan_artifact_tasks(plan)[0]
                task.run()
        self.assertEqual(link.call_args.kwargs["link_libraries"], ["needed"])
        self.assertEqual(link.call_args.kwargs["link_objects"], ["extra.o"])
        self.assertIn("needed", task.inputs)
        self.assertIn("extra.o", task.inputs)


if __name__ == "__main__":
    unittest.main()
