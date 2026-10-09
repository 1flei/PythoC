import ctypes
import os
import sys
import tempfile
import unittest

from pythoc.artifact import (
    ArtifactKind,
    ArtifactPhase,
    ArtifactPlan,
    ArtifactStep,
    LinkPlan,
    LinkScope,
    LoadedArtifact,
    NativeArtifact,
    plan_artifact_tasks,
)
from pythoc.build.deps import GroupDependency, GroupDeps, GroupKey


class _OutputManager:
    def __init__(self, groups):
        self._groups = groups

    def get_all_groups(self):
        return self._groups


class _DependencyTracker:
    def __init__(self, dependencies):
        self._dependencies = dependencies

    def derive_obj_file_from_group_key(self, group_key):
        return None

    def get_deps(self, group_key, obj_file=None):
        return self._dependencies.get(group_key)

    def get_deps_for_group(self, group_key):
        return self._dependencies.get(group_key)


class TestLinkPlan(unittest.TestCase):
    def test_own_group_and_transitive_scopes_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root_obj = self._touch(directory, "root.o")
            child_obj = self._touch(directory, "child.o")
            external_obj = self._touch(directory, "external.o")
            root = ("/src/root.py", None, None, None)
            child = ("/src/child.py", None, None, None)
            groups = {
                root: {"obj_file": root_obj},
                child: {"obj_file": child_obj},
            }
            dependencies = {
                root: GroupDeps(
                    group_key=GroupKey.from_tuple(root),
                    group_dependencies=[
                        GroupDependency(GroupKey.from_tuple(child))
                    ],
                    link_objects=[external_obj],
                    link_libraries=["m"],
                ),
                child: GroupDeps(group_key=GroupKey.from_tuple(child)),
            }
            manager = _OutputManager(groups)
            tracker = _DependencyTracker(dependencies)

            own = LinkPlan.from_group_roots(
                [root],
                scope=LinkScope.OWN_GROUP,
                flush=False,
                output_manager=manager,
                dependency_tracker=tracker,
            )
            transitive = LinkPlan.from_group_roots(
                [root],
                flush=False,
                output_manager=manager,
                dependency_tracker=tracker,
            )

            self.assertEqual(own.obj_files, (root_obj,))
            self.assertEqual(own.link_libraries, ())
            self.assertEqual(
                transitive.obj_files,
                (root_obj, external_obj, child_obj),
            )
            self.assertEqual(transitive.link_libraries, ("m",))

    @staticmethod
    def _touch(directory, name):
        path = os.path.join(directory, name)
        with open(path, "wb") as handle:
            handle.write(name.encode("ascii"))
        return path


class TestArtifactPlan(unittest.TestCase):
    def test_steps_are_ordered_around_link(self):
        with tempfile.TemporaryDirectory() as directory:
            obj = TestLinkPlan._touch(directory, "input.o")
            output = os.path.join(directory, "output.so")

            def cache_check():
                return True

            before = ArtifactStep(
                id="before",
                kind="generate",
                phase=ArtifactPhase.PRE_LINK,
                run=lambda: None,
                outputs=(os.path.join(directory, "entry.o"),),
                cache_check=cache_check,
            )
            after = ArtifactStep(
                id="after",
                kind="publish",
                phase=ArtifactPhase.POST_LINK,
                run=lambda: None,
                outputs=(os.path.join(directory, "manifest.json"),),
            )
            plan = ArtifactPlan(
                kind=ArtifactKind.SHARED_LIBRARY,
                link=LinkPlan(roots=(), obj_files=(obj,)),
                output_path=output,
                steps=(before, after),
            )

            tasks = plan_artifact_tasks(plan)

            self.assertEqual([task.id for task in tasks], [
                "before",
                "link-shared:" + os.path.abspath(output),
                "after",
            ])
            self.assertEqual(tasks[1].deps, ("before",))
            self.assertEqual(tasks[2].deps, (tasks[1].id,))
            self.assertIs(tasks[0].cache_check, cache_check)

    def test_native_artifact_identity_is_file_content(self):
        with tempfile.TemporaryDirectory() as directory:
            output = TestLinkPlan._touch(directory, "artifact.so")
            plan = ArtifactPlan(
                kind=ArtifactKind.SHARED_LIBRARY,
                link=LinkPlan(roots=(), obj_files=(output,)),
                output_path=output,
            )

            first = NativeArtifact.from_plan(plan)
            second = NativeArtifact.from_plan(plan)

            self.assertEqual(first.identity, second.identity)
            self.assertEqual(len(first.identity), 64)

    def test_loaded_artifact_resolves_process_symbol(self):
        artifact = NativeArtifact(
            kind=ArtifactKind.SHARED_LIBRARY,
            identity="test",
            path="current-process",
            link=LinkPlan(roots=(), obj_files=()),
        )
        if sys.platform == 'win32':
            handle = ctypes.CDLL('msvcrt.dll')
        else:
            handle = ctypes.CDLL(None)
        loaded = LoadedArtifact(artifact=artifact, handle=handle)

        self.assertNotEqual(loaded.resolve("malloc"), 0)


if __name__ == "__main__":
    unittest.main()
