"""Small DAG scheduler for build artifacts.

The scheduler coordinates in-process task parallelism. Cross-process safety
is still handled by the existing file locks around artifact publication.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple


TaskFn = Callable[[], object]
CacheFn = Callable[[], bool]
CommitFn = Callable[["TaskResult"], Optional[Iterable["BuildTask"]]]


@dataclass
class TaskResult:
    """Execution result for one task."""

    task_id: str
    kind: str
    skipped: bool = False
    value: object = None


@dataclass(frozen=True)
class BuildTask:
    """A single artifact-producing task in the build DAG.

    ``outputs`` serialize artifact publication for shared paths. ``resources``
    serialize named non-file side effects such as global registries or queues.
    ``on_success`` runs on the scheduler thread after ``run`` completes and may
    return more tasks, allowing dynamic build graph expansion with serialized
    state commits.
    """

    id: str
    kind: str
    run: TaskFn
    deps: Sequence[str] = field(default_factory=tuple)
    inputs: Sequence[str] = field(default_factory=tuple)
    outputs: Sequence[str] = field(default_factory=tuple)
    resources: Sequence[str] = field(default_factory=tuple)
    cache_check: Optional[CacheFn] = None
    on_success: Optional[CommitFn] = None


class BuildSchedulerError(RuntimeError):
    """Raised when one or more scheduled tasks fail."""

    def __init__(self, failures: Dict[str, BaseException], blocked: Iterable[str] = ()):
        self.failures = failures
        self.blocked = tuple(blocked)
        failure_text = "; ".join(f"{task}: {exc}" for task, exc in failures.items())
        if self.blocked:
            failure_text += f"; blocked: {', '.join(self.blocked)}"
        super().__init__(failure_text or "build scheduler failed")


class BuildScheduler:
    """Run build tasks in topological order with per-output serialization."""

    def __init__(self, max_workers: Optional[int] = None):
        if max_workers is None:
            max_workers = int(os.environ.get("PC_BUILD_WORKERS", "0") or "0")
        if max_workers <= 0:
            max_workers = os.cpu_count() or 1
        self.max_workers = max(1, max_workers)
        self._output_locks: Dict[str, threading.Lock] = {}
        self._output_locks_guard = threading.Lock()
        # Track resources/outputs held by *running* tasks so the dispatch
        # loop can skip blocked tasks instead of wasting worker slots.
        self._busy_resources: Set[str] = set()
        self._busy_outputs: Set[str] = set()

    def _can_dispatch(self, task: BuildTask) -> bool:
        """Return True if no running task holds this task's resources/outputs."""
        for r in task.resources:
            if r in self._busy_resources:
                return False
        for o in task.outputs:
            if o and os.path.abspath(o) in self._busy_outputs:
                return False
        return True

    def _mark_busy(self, task: BuildTask):
        for r in task.resources:
            self._busy_resources.add(r)
        for o in task.outputs:
            if o:
                self._busy_outputs.add(os.path.abspath(o))

    def _mark_free(self, task: BuildTask):
        for r in task.resources:
            self._busy_resources.discard(r)
        for o in task.outputs:
            if o:
                self._busy_outputs.discard(os.path.abspath(o))

    def run(self, tasks: Sequence[BuildTask]) -> Dict[str, TaskResult]:
        task_map: Dict[str, BuildTask] = {}
        remaining_deps: Dict[str, Set[str]] = {}
        dependents: Dict[str, Set[str]] = {}
        ready: List[str] = []
        running = {}
        completed: Set[str] = set()
        failed: Dict[str, BaseException] = {}
        results: Dict[str, TaskResult] = {}

        def add_tasks(new_tasks: Iterable[BuildTask]):
            batch = list(new_tasks or ())
            if not batch:
                return

            for task in batch:
                if task.id in task_map:
                    raise ValueError(f"Duplicate build task id: {task.id}")
                task_map[task.id] = task
                dependents.setdefault(task.id, set())

            known_ids = set(task_map)
            for task in batch:
                missing = [dep for dep in task.deps if dep not in known_ids]
                if missing:
                    raise ValueError(
                        f"Task {task.id} depends on unknown task {missing[0]}"
                    )

                deps = {
                    dep
                    for dep in task.deps
                    if dep not in completed and dep not in failed
                }
                remaining_deps[task.id] = deps
                for dep in task.deps:
                    if dep not in completed and dep not in failed:
                        dependents.setdefault(dep, set()).add(task.id)

                if not deps and not any(dep in failed for dep in task.deps):
                    ready.append(task.id)

        def release_dependents(task_id: str):
            for dependent in dependents.get(task_id, ()):
                if dependent in completed or dependent in failed:
                    continue
                remaining_deps[dependent].discard(task_id)
                if not remaining_deps[dependent]:
                    if any(dep in failed for dep in task_map[dependent].deps):
                        continue
                    ready.append(dependent)

        add_tasks(tasks)
        if not task_map:
            return {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            while ready or running:
                # Dispatch loop: scan ready queue for tasks whose resources/
                # outputs are not held by any running task.  Tasks that can't
                # dispatch yet are skipped (left in ready) so we don't waste
                # worker slots on blocked threads.
                dispatched_any = True
                while dispatched_any and ready and len(running) < self.max_workers:
                    dispatched_any = False
                    skipped: List[str] = []
                    while ready and len(running) < self.max_workers:
                        task_id = ready.pop(0)
                        if task_id in completed or task_id in failed:
                            continue
                        task = task_map[task_id]
                        if not self._can_dispatch(task):
                            skipped.append(task_id)
                            continue
                        self._mark_busy(task)
                        future = pool.submit(self._run_task, task)
                        running[future] = task_id
                        dispatched_any = True
                    ready.extend(skipped)

                if not running:
                    break

                done, _ = wait(running.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    task_id = running.pop(future)
                    task = task_map[task_id]
                    self._mark_free(task)
                    try:
                        result = future.result()
                        extra_tasks = ()
                        if task.on_success is not None:
                            extra_tasks = task.on_success(result) or ()
                        results[task_id] = result
                        completed.add(task_id)
                        add_tasks(extra_tasks)
                    except BaseException as exc:  # pragma: no cover - re-raised below
                        failed[task_id] = exc
                        completed.add(task_id)

                    release_dependents(task_id)

                # Do NOT abort the entire batch when one task fails.  Tasks
                # whose dependencies have failed are naturally excluded from
                # the ready queue by release_dependents/add_tasks.  Independent
                # tasks (e.g. compile for a different group) must be allowed to
                # finish so their artifacts are published.  All failures are
                # collected and raised together after the loop exits.

        if failed:
            blocked = sorted(set(task_map) - set(results) - set(failed))
            raise BuildSchedulerError(failed, blocked)

        if len(results) != len(task_map):
            missing = sorted(set(task_map) - set(results))
            raise BuildSchedulerError(
                {"<scheduler>": RuntimeError("cycle or unsatisfied dependency")},
                missing,
            )

        return results

    def _validate_tasks(self, tasks: Sequence[BuildTask]) -> Dict[str, BuildTask]:
        task_map: Dict[str, BuildTask] = {}
        for task in tasks:
            if task.id in task_map:
                raise ValueError(f"Duplicate build task id: {task.id}")
            task_map[task.id] = task

        for task in tasks:
            for dep in task.deps:
                if dep not in task_map:
                    raise ValueError(f"Task {task.id} depends on unknown task {dep}")
        return task_map

    def _run_task(self, task: BuildTask) -> TaskResult:
        locks = self._locks_for_task(task)
        for lock in locks:
            lock.acquire()
        try:
            if task.cache_check is not None and task.cache_check():
                return TaskResult(task_id=task.id, kind=task.kind, skipped=True)
            value = task.run()
            return TaskResult(task_id=task.id, kind=task.kind, value=value)
        finally:
            for lock in reversed(locks):
                lock.release()

    def _locks_for_task(self, task: BuildTask) -> List[threading.Lock]:
        lock_keys = {
            f"output:{os.path.abspath(output)}"
            for output in task.outputs
            if output
        }
        lock_keys.update(
            f"resource:{resource}"
            for resource in task.resources
            if resource
        )
        locks: List[threading.Lock] = []
        with self._output_locks_guard:
            for lock_key in sorted(lock_keys):
                lock = self._output_locks.get(lock_key)
                if lock is None:
                    lock = threading.Lock()
                    self._output_locks[lock_key] = lock
                locks.append(lock)
        return locks
