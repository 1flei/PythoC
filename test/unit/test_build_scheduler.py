import threading
import time
import unittest

from pythoc.build.scheduler import BuildScheduler, BuildSchedulerError, BuildTask


class TestBuildScheduler(unittest.TestCase):
    def test_runs_tasks_after_dependencies(self):
        events = []

        tasks = [
            BuildTask(id="a", kind="test", run=lambda: events.append("a")),
            BuildTask(
                id="b",
                kind="test",
                deps=("a",),
                run=lambda: events.append("b"),
            ),
        ]

        BuildScheduler(max_workers=2).run(tasks)

        self.assertEqual(events, ["a", "b"])

    def test_runs_independent_ready_tasks_in_parallel(self):
        barrier = threading.Barrier(2, timeout=2.0)

        def wait_at_barrier():
            barrier.wait()

        tasks = [
            BuildTask(id="a", kind="test", run=wait_at_barrier),
            BuildTask(id="b", kind="test", run=wait_at_barrier),
        ]

        BuildScheduler(max_workers=2).run(tasks)

    def test_serializes_tasks_with_same_output(self):
        active = 0
        max_active = 0
        guard = threading.Lock()

        def run():
            nonlocal active, max_active
            with guard:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with guard:
                active -= 1

        tasks = [
            BuildTask(id="a", kind="test", outputs=("same.o",), run=run),
            BuildTask(id="b", kind="test", outputs=("same.o",), run=run),
        ]

        BuildScheduler(max_workers=2).run(tasks)

        self.assertEqual(max_active, 1)

    def test_serializes_tasks_with_same_resource(self):
        active = 0
        max_active = 0
        guard = threading.Lock()

        def run():
            nonlocal active, max_active
            with guard:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with guard:
                active -= 1

        tasks = [
            BuildTask(id="a", kind="test", resources=("global-registry",), run=run),
            BuildTask(id="b", kind="test", resources=("global-registry",), run=run),
        ]

        BuildScheduler(max_workers=2).run(tasks)

        self.assertEqual(max_active, 1)

    def test_failure_blocks_downstream_tasks(self):
        def fail():
            raise RuntimeError("boom")

        tasks = [
            BuildTask(id="a", kind="test", run=fail),
            BuildTask(id="b", kind="test", deps=("a",), run=lambda: None),
        ]

        with self.assertRaises(BuildSchedulerError) as cm:
            BuildScheduler(max_workers=2).run(tasks)

        self.assertIn("a", cm.exception.failures)
        self.assertIn("b", cm.exception.blocked)

    def test_cache_check_skips_run(self):
        did_run = False

        def run():
            nonlocal did_run
            did_run = True

        result = BuildScheduler(max_workers=1).run([
            BuildTask(
                id="cached",
                kind="test",
                run=run,
                cache_check=lambda: True,
            )
        ])

        self.assertFalse(did_run)
        self.assertTrue(result["cached"].skipped)

    def test_success_callback_can_add_dynamic_tasks(self):
        events = []

        def commit(result):
            events.append(f"commit:{result.task_id}")
            return [
                BuildTask(
                    id="b",
                    kind="test",
                    deps=("a",),
                    run=lambda: events.append("run:b"),
                )
            ]

        BuildScheduler(max_workers=2).run([
            BuildTask(
                id="a",
                kind="test",
                run=lambda: events.append("run:a"),
                on_success=commit,
            )
        ])

        self.assertEqual(events, ["run:a", "commit:a", "run:b"])

    def test_success_callback_failure_blocks_downstream_tasks(self):
        def fail_commit(_result):
            raise RuntimeError("commit boom")

        tasks = [
            BuildTask(
                id="a",
                kind="test",
                run=lambda: None,
                on_success=fail_commit,
            ),
            BuildTask(id="b", kind="test", deps=("a",), run=lambda: None),
        ]

        with self.assertRaises(BuildSchedulerError) as cm:
            BuildScheduler(max_workers=1).run(tasks)

        self.assertIn("a", cm.exception.failures)
        self.assertIn("b", cm.exception.blocked)


if __name__ == "__main__":
    unittest.main()
