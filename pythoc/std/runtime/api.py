"""
Runtime: Top-level N:M runtime with public API.

Owns the Scheduler, starts/stops worker threads.
Provides the user-facing spawn/join/yield API.
Integrates with PythoC's effect system for dependency injection.

Usage:
    from pythoc.std.runtime import api
    from pythoc import effect
    from pythoc.std import runtime  # registers effect.executor

    @compile
    def my_work(arg: ptr[void]) -> ptr[void]:
        # ... do work ...
        effect.executor.yield_now()  # cooperative yield
        return result

    @compile
    def main() -> i32:
        rt: ptr[Runtime] = runtime_new(i32(4))  # 4 worker threads
        runtime_start(rt)

        h: TaskHandle = runtime_spawn(rt, my_work, my_arg, u64(0))
        result: ptr[void] = runtime_join(rt, h)

        runtime_shutdown(rt)
        runtime_free(rt)
        return i32(0)
"""
from __future__ import annotations

from .policy import bind_mem
bind_mem()

from pythoc import (
    compile, effect, i32, i64, u64, u8, ptr, void, struct, nullptr, sizeof,
    func,
)
from pythoc.libc.string import memset

from .platform import (
    thread_create, thread_join,
    atomic_store_i64, atomic_load_i32,
    SpinLock, spinlock_lock, spinlock_unlock,
)
from .scheduler import (
    Scheduler, Worker,
    sched_init, sched_destroy, sched_spawn, sched_spawn_local, sched_notify_all,
    sched_yield, sched_join, sched_current_worker,
    worker_loop,
)
from .task import (
    Task, TaskHandle, task_destroy, task_handle_new, task_handle_consume,
    TASK_FINISHED,
)
from .coroutine import DEFAULT_STACK_SIZE


# ============================================================
# Runtime struct: the top-level object users interact with
# ============================================================

@compile
class Runtime:
    sched: Scheduler       # embedded scheduler (not a pointer - single alloc)
    num_workers: i32       # number of worker OS threads
    started: i32           # 1 if workers are running


# ============================================================
# Runtime lifecycle
# ============================================================

@compile
def runtime_new(num_workers: i32) -> ptr[Runtime]:
    """Create a new runtime with M worker threads.

    Workers are not started yet.  Call runtime_start() to begin.
    """
    rt: ptr[Runtime] = ptr[Runtime](effect.mem.malloc(u64(sizeof(Runtime))))
    memset(ptr[void](rt), 0, i64(sizeof(Runtime)))

    rt.num_workers = num_workers
    rt.started = i32(0)

    sched_init(
        ptr[Scheduler](ptr[void](ptr(rt.sched))),
        num_workers
    )

    return rt


@compile
def runtime_start(rt: ptr[Runtime]) -> void:
    """Start all worker threads.  The runtime begins processing tasks."""
    sched: ptr[Scheduler] = ptr[Scheduler](ptr[void](ptr(rt.sched)))
    num: i32 = rt.num_workers

    i: i32 = 0
    while i < num:
        w: ptr[Worker] = ptr[Worker](
            ptr[void](
                ptr[u8](ptr[void](sched.workers)) + i64(i) * i64(sizeof(Worker))
            )
        )

        # Create OS thread running worker_loop.
        w.thread = thread_create(ptr[void](worker_loop), ptr[void](w))
        i = i + 1

    rt.started = i32(1)


@compile
def runtime_shutdown(rt: ptr[Runtime]) -> void:
    """Signal shutdown and wait for all workers to finish.

    All queued tasks will be drained before workers exit.
    """
    sched: ptr[Scheduler] = ptr[Scheduler](ptr[void](ptr(rt.sched)))

    # Set shutdown flag
    atomic_store_i64(
        ptr[i64](ptr[void](ptr(sched.shutdown))),
        i64(1)
    )

    # Wake all parked workers so they see the flag
    sched_notify_all(sched)

    # Join all worker threads
    num: i32 = rt.num_workers
    i: i32 = 0
    while i < num:
        w: ptr[Worker] = ptr[Worker](
            ptr[void](
                ptr[u8](ptr[void](sched.workers)) + i64(i) * i64(sizeof(Worker))
            )
        )
        thread_join(w.thread)
        i = i + 1

    rt.started = i32(0)


@compile
def runtime_free(rt: ptr[Runtime]) -> void:
    """Free runtime resources.  Must be called after shutdown."""
    sched_destroy(ptr[Scheduler](ptr[void](ptr(rt.sched))))
    effect.mem.free(ptr[void](rt))


# ============================================================
# Raw task API: internal scheduler/runtime ownership
# ============================================================

@compile
def runtime_spawn_raw(
    rt: ptr[Runtime],
    entry: func[ptr[void], ptr[void]],
    arg: ptr[void],
    stack_size: u64
) -> ptr[Task]:
    """Spawn a task and return the raw scheduler pointer."""
    sched: ptr[Scheduler] = ptr[Scheduler](ptr[void](ptr(rt.sched)))

    # Fast path: if we're on a worker, spawn locally (no global lock)
    worker: ptr[Worker] = sched_current_worker(sched)
    if worker != nullptr:
        return sched_spawn_local(sched, worker, entry, arg, stack_size)

    # Slow path: external spawn (goes to global queue)
    return sched_spawn(sched, entry, arg, stack_size)


@compile
def runtime_join_raw(rt: ptr[Runtime], task: ptr[Task]) -> ptr[void]:
    """Wait for a raw task to finish and return its result."""
    sched: ptr[Scheduler] = ptr[Scheduler](ptr[void](ptr(rt.sched)))
    worker: ptr[Worker] = sched_current_worker(sched)
    if worker != nullptr:
        return sched_join(worker, task)

    while atomic_load_i32(ptr[i32](ptr[void](ptr(task.state)))) != TASK_FINISHED:
        pass
    result: ptr[void] = task.result
    return result


@compile
def runtime_detach_raw(rt: ptr[Runtime], task: ptr[Task]) -> void:
    """Detach raw task ownership without waiting."""
    spinlock_lock(ptr[SpinLock](ptr[void](ptr(task.lock))))
    if atomic_load_i32(ptr[i32](ptr[void](ptr(task.state)))) == TASK_FINISHED:
        spinlock_unlock(ptr[SpinLock](ptr[void](ptr(task.lock))))
        task_destroy(task)
        return
    task.detached = i32(1)
    spinlock_unlock(ptr[SpinLock](ptr[void](ptr(task.lock))))


# ============================================================
# Public API: spawn / join / yield / detach
# ============================================================

@compile
def runtime_spawn(
    rt: ptr[Runtime],
    entry: func[ptr[void], ptr[void]],
    arg: ptr[void],
    stack_size: u64
) -> TaskHandle:
    """Spawn a new task and return a linear ownership handle."""
    return task_handle_new(runtime_spawn_raw(rt, entry, arg, stack_size))


@compile
def runtime_join(rt: ptr[Runtime], handle: TaskHandle) -> ptr[void]:
    """Consume a task handle, wait for completion, and free task resources."""
    task: ptr[Task] = task_handle_consume(handle)
    result: ptr[void] = runtime_join_raw(rt, task)
    task_destroy(task)
    return result


@compile
def runtime_detach(rt: ptr[Runtime], handle: TaskHandle) -> void:
    """Consume a task handle and let the scheduler free it on completion."""
    task: ptr[Task] = task_handle_consume(handle)
    runtime_detach_raw(rt, task)


# ============================================================
# yield_now: cooperative yield from within a task
#
# This must be called from code running inside a task.
# It suspends the current task and returns control to the scheduler.
# ============================================================

@compile
def yield_now_from_worker(w: ptr[Worker]) -> void:
    """Yield the current task.  Must be called from within a task."""
    sched_yield(w)


@compile
def runtime_current_worker(rt: ptr[Runtime]) -> ptr[Worker]:
    sched: ptr[Scheduler] = ptr[Scheduler](ptr[void](ptr(rt.sched)))
    return sched_current_worker(sched)


@compile
def runtime_yield_now(rt: ptr[Runtime]) -> void:
    worker: ptr[Worker] = runtime_current_worker(rt)
    if worker != nullptr:
        sched_yield(worker)
