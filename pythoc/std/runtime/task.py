"""
Task: Coroutine + scheduling metadata (cross-platform).

A Task is the user-visible unit of concurrent work.
It wraps a Coroutine with:
- Unique ID for identification
- Result storage for join semantics
- State for the scheduler
- Join waiter (another task waiting on this one)

Design:
    Tasks are allocated from the heap.  A TaskHandle (linear) is the
    user's proof of ownership: you MUST either join() or detach() it.
    This prevents fire-and-forget resource leaks.

    The scheduler operates on Task pointers directly.
    No indirection, no virtual dispatch.
"""
from __future__ import annotations

from .policy import bind_mem
bind_mem()

from pythoc import (
    compile, effect, i32, i64, u64, ptr, void, struct, nullptr, sizeof, linear,
    consume, func, refined, assume,
)
from pythoc.libc.string import memset

from .coroutine import (
    Coroutine, coro_alloc, coro_init, coro_switch, coro_free,
    CORO_READY, CORO_DONE, DEFAULT_STACK_SIZE,
)
from .platform import (
    SpinLock, spinlock_init, spinlock_lock, spinlock_unlock,
    atomic_fetch_add_i64, atomic_store_i32,
)


# ============================================================
# Task states (scheduler-level, distinct from coroutine state)
# ============================================================

TASK_PENDING   = i32(0)   # Created, in the queue, not yet polled
TASK_RUNNING   = i32(1)   # Currently executing on a worker
TASK_BLOCKED   = i32(2)   # Waiting on something (channel, join, etc.)
TASK_FINISHED  = i32(3)   # Scheduler completed task cleanup, safe to join/free
TASK_WOKEN     = i32(4)   # Wake requested before the task fully suspended
TASK_BLOCKING  = i32(5)   # On a wait queue, switching back to scheduler
TASK_FINISHING = i32(6)   # Entry returned, worker still owns completion cleanup


# ============================================================
# Task struct
# ============================================================

@compile
class Task:
    id: u64                  # unique task identifier
    coro: ptr[Coroutine]     # underlying coroutine (owns the stack)
    scheduler_coro: ptr[Coroutine]  # scheduler context to return to
    state: i32               # TASK_PENDING | TASK_RUNNING | TASK_BLOCKED | TASK_FINISHED
    result: ptr[void]        # result pointer (set by entry fn, read by joiner)
    entry_fn: func[ptr[void], ptr[void]]  # user's function
    entry_arg: ptr[void]     # argument passed to user's function
    joiner: ptr[Task]        # task waiting to join this one (nullable)
    detached: i32            # 1 if no joiner will consume the task
    queued: i32              # 1 while linked through TaskQueue.next
    lock: SpinLock           # protects state transitions
    next: ptr[Task]          # intrusive linked list for queues


# ============================================================
# TaskHandle: linear proof of task ownership
#
# You get a TaskHandle from spawn().
# You MUST consume it via join() or detach().
# Compiler enforces this at compile time — no leak possible.
# ============================================================

TaskProof = refined[linear, "runtime_task"]


@compile
class TaskHandle:
    task: ptr[Task]         # the task this handle refers to
    _proof: TaskProof       # linear token: must be consumed


@compile
def task_handle_new(task: ptr[Task]) -> TaskHandle:
    handle: TaskHandle
    handle.task = task
    handle._proof = assume(linear(), "runtime_task")
    return handle


@compile
def task_handle_consume(handle: TaskHandle) -> ptr[Task]:
    task: ptr[Task] = handle.task
    consume(handle._proof)
    return task


# ============================================================
# Global task ID counter (atomic)
# ============================================================

@compile
class TaskIdGen:
    counter: i64


@compile
def _next_task_id(gen: ptr[TaskIdGen]) -> u64:
    """Generate next unique task ID (atomic increment)."""
    return u64(atomic_fetch_add_i64(
        ptr[i64](ptr[void](gen)),
        i64(1)
    ))


# ============================================================
# Task lifecycle
# ============================================================

@compile
def task_set_result(task: ptr[Task], result: ptr[void]) -> void:
    """Set task result (called from within the task's entry function)."""
    task.result = result


@compile
def task_mark_finished(task: ptr[Task]) -> void:
    """Mark that the entry returned; the worker publishes FINISHED later."""
    spinlock_lock(ptr[SpinLock](ptr[void](ptr(task.lock))))
    atomic_store_i32(ptr[i32](ptr[void](ptr(task.state))), TASK_FINISHING)
    task.coro.state = CORO_DONE
    spinlock_unlock(ptr[SpinLock](ptr[void](ptr(task.lock))))


@compile
def task_entry_trampoline(task_arg: ptr[void]) -> ptr[void]:
    task: ptr[Task] = ptr[Task](task_arg)
    result: ptr[void] = task.entry_fn(task.entry_arg)
    task_set_result(task, result)
    task_mark_finished(task)
    coro_switch(task.coro, task.scheduler_coro)
    return result


@compile
def task_create(
    id_gen: ptr[TaskIdGen],
    entry: func[ptr[void], ptr[void]],
    arg: ptr[void],
    stack_size: u64
) -> ptr[Task]:
    """Allocate and initialize a new task.

    The task is in TASK_PENDING state.  It will begin executing
    `entry(arg)` when a worker picks it up.

    Returns:
        ptr[Task] — caller must eventually free via task_destroy()
    """
    task: ptr[Task] = ptr[Task](effect.mem.malloc(u64(sizeof(Task))))
    memset(ptr[void](task), 0, i64(sizeof(Task)))

    task.id = _next_task_id(id_gen)
    task.state = TASK_PENDING
    task.entry_fn = entry
    task.entry_arg = arg
    task.result = nullptr
    task.joiner = nullptr
    task.next = nullptr
    task.detached = i32(0)
    task.queued = i32(0)
    task.scheduler_coro = nullptr

    spinlock_init(ptr[SpinLock](ptr[void](ptr(task.lock))))

    # Allocate coroutine with its own stack
    coro: ptr[Coroutine] = coro_alloc(stack_size)
    task.coro = coro
    coro_init(coro, ptr[void](task_entry_trampoline), ptr[void](task))

    return task


@compile
def task_destroy(task: ptr[Task]) -> void:
    """Free task and its coroutine.  Only call after task is FINISHED."""
    if task.coro != nullptr:
        coro_free(task.coro)
    effect.mem.free(ptr[void](task))


# ============================================================
# Task queue: intrusive singly-linked FIFO
#
# Used for the global injection queue and per-worker overflow.
# Lock-protected for multi-producer access.
# ============================================================

@compile
class TaskQueue:
    head: ptr[Task]
    tail: ptr[Task]
    count: u64
    lock: SpinLock


@compile
def taskq_init(q: ptr[TaskQueue]) -> void:
    """Initialize an empty task queue."""
    q.head = nullptr
    q.tail = nullptr
    q.count = u64(0)
    spinlock_init(ptr[SpinLock](ptr[void](ptr(q.lock))))


@compile
def taskq_push(q: ptr[TaskQueue], task: ptr[Task]) -> void:
    """Push task to tail of queue (thread-safe)."""
    spinlock_lock(ptr[SpinLock](ptr[void](ptr(q.lock))))
    spinlock_lock(ptr[SpinLock](ptr[void](ptr(task.lock))))
    if task.queued != i32(0):
        spinlock_unlock(ptr[SpinLock](ptr[void](ptr(task.lock))))
        spinlock_unlock(ptr[SpinLock](ptr[void](ptr(q.lock))))
        return
    task.queued = i32(1)
    spinlock_unlock(ptr[SpinLock](ptr[void](ptr(task.lock))))

    task.next = nullptr
    if q.tail == nullptr:
        q.head = task
        q.tail = task
    else:
        q.tail.next = task
        q.tail = task
    q.count = q.count + u64(1)

    spinlock_unlock(ptr[SpinLock](ptr[void](ptr(q.lock))))


@compile
def taskq_pop(q: ptr[TaskQueue]) -> ptr[Task]:
    """Pop task from head of queue (thread-safe).  Returns nullptr if empty."""
    spinlock_lock(ptr[SpinLock](ptr[void](ptr(q.lock))))

    task: ptr[Task] = q.head
    if task != nullptr:
        q.head = task.next
        if q.head == nullptr:
            q.tail = nullptr
        task.next = nullptr
        spinlock_lock(ptr[SpinLock](ptr[void](ptr(task.lock))))
        task.queued = i32(0)
        spinlock_unlock(ptr[SpinLock](ptr[void](ptr(task.lock))))
        q.count = q.count - u64(1)

    spinlock_unlock(ptr[SpinLock](ptr[void](ptr(q.lock))))
    return task


@compile
def taskq_is_empty(q: ptr[TaskQueue]) -> i32:
    """Check if queue is empty."""
    spinlock_lock(ptr[SpinLock](ptr[void](ptr(q.lock))))
    if q.head == nullptr:
        spinlock_unlock(ptr[SpinLock](ptr[void](ptr(q.lock))))
        return i32(1)
    spinlock_unlock(ptr[SpinLock](ptr[void](ptr(q.lock))))
    return i32(0)
