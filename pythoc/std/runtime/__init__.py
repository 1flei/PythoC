"""
PythoC N:M Async Runtime (pythoc.std.runtime)

Production-grade N:M coroutine scheduler: N lightweight coroutines
multiplexed onto M OS threads via work-stealing.

Design philosophy (PythoC-ic):
- C-level runtime: stackful coroutines via platform context switch
- Zero hidden allocation: all memory explicit
- Effect-injectable: executor is an effect, overridable
- Linear ownership: tasks are linear resources, must be joined/detached
- Explicit control flow: no implicit suspension, no magic

Architecture:
    +-----------+     +-----------+     +-----------+
    | Worker 0  |     | Worker 1  |     | Worker M  |
    | [deque]   |     | [deque]   |     | [deque]   |
    +-----+-----+     +-----+-----+     +-----+-----+
          |                 |                 |
          +---------+-------+---------+-------+
                    |                 |
              [global queue]    [steal from others]

Core concepts:
- Coroutine: stackful execution context (own stack, switchable)
- Task: coroutine + metadata (state, result, join handle)
- Scheduler: per-worker local deques + global injection queue
- Worker: OS thread running scheduler loop
- Runtime: owns workers + global state

Layers:
    platform.py       OS primitives (pthread, ucontext, atomics)
    coroutine.py      Stackful coroutine (alloc/init/switch/free)
    task.py           Task struct + intrusive queue
    deque.py          Chase-Lev work-stealing deque
    scheduler.py      N:M scheduler + worker loop
    api.py            Runtime lifecycle (new/start/spawn/join/shutdown)
    channel.py        Bounded MPMC channel (generic factory)
    executor_effect.py  Effect system integration

Usage:
    from pythoc import compile, i32, ptr, void
    from pythoc.std.runtime.api import (
        Runtime, runtime_new, runtime_start,
        runtime_spawn, runtime_join, runtime_join_and_free,
        runtime_shutdown, runtime_free,
    )
    from pythoc.std.runtime.channel import Channel

    # Create a typed channel
    Ch = Channel(i32, capacity=128)

    @compile
    def producer(arg: ptr[void]) -> ptr[void]:
        ch: ptr[Ch.type] = ptr[Ch.type](arg)
        i: i32 = 0
        while i < 100:
            Ch.try_send(ch, i)
            i = i + 1
        return ptr[void](nullptr)

    @compile
    def main() -> i32:
        rt: ptr[Runtime] = runtime_new(i32(4))
        runtime_start(rt)

        ch: ptr[Ch.type] = Ch.create()

        t1: ptr[Task] = runtime_spawn(rt, producer, ptr[void](ch), u64(0))
        runtime_join_and_free(rt, t1)

        Ch.destroy(ch)
        runtime_shutdown(rt)
        runtime_free(rt)
        return i32(0)

Effect-based usage:
    from pythoc import compile, effect
    from pythoc.std.runtime import executor_effect  # registers effect.executor

    @compile
    def my_task(arg: ptr[void]) -> ptr[void]:
        effect.executor.yield_now()
        return arg

    @compile
    def main() -> i32:
        task = effect.executor.spawn(my_task, nullptr)
        result = effect.executor.join(task)
        return i32(0)
"""

# Public re-exports
from .coroutine import (
    Coroutine, coro_alloc, coro_init, coro_switch, coro_free,
    DEFAULT_STACK_SIZE,
    CORO_READY, CORO_RUNNING, CORO_SUSPENDED, CORO_DONE,
)
from .task import (
    Task, TaskHandle, TaskQueue,
    task_create, task_destroy, task_mark_finished, task_set_result,
    taskq_init, taskq_push, taskq_pop, taskq_is_empty,
    TASK_PENDING, TASK_RUNNING, TASK_BLOCKED, TASK_FINISHED,
)
from .deque import (
    WSDeque, wsdeque_init, wsdeque_push, wsdeque_pop, wsdeque_steal,
)
from .scheduler import (
    Scheduler, Worker,
    sched_init, sched_destroy, sched_spawn, sched_spawn_local,
    sched_yield, sched_join,
)
from .api import (
    Runtime, runtime_new, runtime_start, runtime_shutdown, runtime_free,
    runtime_spawn, runtime_join, runtime_join_and_free, runtime_detach,
    runtime_current_worker, runtime_yield_now,
)
from .channel import Channel
