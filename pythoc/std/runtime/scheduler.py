"""
Scheduler: Work-stealing N:M scheduler (cross-platform).

The scheduler multiplexes N tasks onto M worker OS threads.
Each worker:
1. Pops from its local deque (fast, no contention)
2. If empty, checks the global queue
3. If empty, steals from a random other worker
4. If nothing found, parks (condition variable)

Design:
    The Scheduler is a value type (struct) holding all shared state.
    Workers are OS threads running the worker_loop function.
    There is no "scheduler thread" — scheduling is distributed.

    All threading primitives use the portable abstraction from platform.py:
    - ThreadHandle, thread_create, thread_join
    - Mutex, mutex_init/lock/unlock/destroy
    - CondVar, condvar_init/wait/signal/broadcast/destroy
    These compile to pthread on Linux/macOS, Win32 on Windows.
"""
from __future__ import annotations

from .policy import bind_mem
bind_mem()

from pythoc import (
    compile, effect, i32, i64, u64, u8, ptr, void, struct, nullptr, sizeof, func,
)
from pythoc.libc.string import memset

from .platform import (
    ThreadHandle, Mutex, CondVar,
    thread_create, thread_join, thread_current, thread_equal,
    mutex_init, mutex_lock, mutex_unlock, mutex_destroy,
    condvar_init, condvar_wait, condvar_signal, condvar_broadcast, condvar_destroy,
    SpinLock, spinlock_init, spinlock_lock, spinlock_unlock,
    atomic_load_i64, atomic_store_i64, atomic_fetch_add_i64, atomic_cas_i64,
    atomic_load_i32, atomic_store_i32,
)
from .coroutine import (
    Coroutine, coro_alloc, coro_init, coro_switch, coro_free,
    CORO_READY, CORO_RUNNING, CORO_SUSPENDED, CORO_DONE,
    DEFAULT_STACK_SIZE,
)
from .task import (
    Task, TaskQueue, TaskIdGen,
    task_create, task_destroy, task_mark_finished,
    taskq_init, taskq_push, taskq_pop, taskq_is_empty,
    TASK_PENDING, TASK_RUNNING, TASK_BLOCKED, TASK_FINISHED, TASK_WOKEN,
    TASK_BLOCKING,
    _next_task_id,
)
from .deque import (
    WSDeque, wsdeque_init, wsdeque_push, wsdeque_pop, wsdeque_steal, wsdeque_size,
)


# ============================================================
# Worker: one OS thread in the pool
# ============================================================

MAX_WORKERS = 128  # max supported worker threads

@compile
class Worker:
    id: i32                     # worker index [0, M)
    thread: ThreadHandle        # OS thread handle (portable)
    local_deque: WSDeque        # local work-stealing deque
    scheduler: ptr[void]        # back-pointer to Scheduler
    current_task: ptr[Task]     # task currently executing (or nullptr)
    scheduler_coro: Coroutine   # "scheduler context" for this worker
    is_parked: i32              # 1 if parked waiting for work
    should_stop: i32            # 1 if shutdown requested


# ============================================================
# Scheduler: global shared state
# ============================================================

@compile
class Scheduler:
    workers: ptr[Worker]         # array of M workers
    num_workers: i32             # M (number of OS threads)
    global_queue: TaskQueue      # injection queue for new tasks
    id_gen: TaskIdGen            # atomic task ID generator
    active_tasks: i64            # atomic count of alive tasks

    # Parking lot: workers park here when idle
    park_mutex: Mutex
    park_cond: CondVar

    # Shutdown flag
    shutdown: i64                # atomic: 0 = running, 1 = shutting down


# ============================================================
# Scheduler lifecycle
# ============================================================

@compile
def sched_init(sched: ptr[Scheduler], num_workers: i32) -> void:
    """Initialize scheduler with M worker slots.  Does NOT start threads."""
    memset(ptr[void](sched), 0, i64(sizeof(Scheduler)))

    sched.num_workers = num_workers
    sched.shutdown = i64(0)
    sched.active_tasks = i64(0)

    # Allocate worker array
    worker_bytes: i64 = i64(num_workers) * i64(sizeof(Worker))
    sched.workers = ptr[Worker](effect.mem.malloc(u64(worker_bytes)))
    memset(ptr[void](sched.workers), 0, worker_bytes)

    # Initialize global queue
    taskq_init(ptr[TaskQueue](ptr[void](ptr(sched.global_queue))))

    # Initialize ID generator
    sched.id_gen.counter = i64(1)

    # Initialize parking lot (portable)
    mutex_init(ptr[Mutex](ptr[void](ptr(sched.park_mutex))))
    condvar_init(ptr[CondVar](ptr[void](ptr(sched.park_cond))))

    # Initialize each worker
    i: i32 = 0
    while i < num_workers:
        w: ptr[Worker] = ptr[Worker](
            ptr[void](ptr[u8](ptr[void](sched.workers)) + i64(i) * i64(sizeof(Worker)))
        )
        w.id = i
        w.scheduler = ptr[void](sched)
        w.current_task = nullptr
        w.is_parked = i32(0)
        w.should_stop = i32(0)
        wsdeque_init(ptr[WSDeque](ptr[void](ptr(w.local_deque))))
        i = i + 1


@compile
def sched_destroy(sched: ptr[Scheduler]) -> void:
    """Destroy scheduler.  Must be called after all workers have stopped."""
    mutex_destroy(ptr[Mutex](ptr[void](ptr(sched.park_mutex))))
    condvar_destroy(ptr[CondVar](ptr[void](ptr(sched.park_cond))))
    effect.mem.free(ptr[void](sched.workers))


# ============================================================
# Task submission
# ============================================================

@compile
def sched_spawn(
    sched: ptr[Scheduler],
    entry: func[ptr[void], ptr[void]],
    arg: ptr[void],
    stack_size: u64
) -> ptr[Task]:
    """Create a new task and submit it to the scheduler.

    The task is pushed to the global queue.  A parked worker is
    signaled to pick it up.
    """
    sz: u64 = stack_size
    if sz == u64(0):
        sz = DEFAULT_STACK_SIZE

    task: ptr[Task] = task_create(
        ptr[TaskIdGen](ptr[void](ptr(sched.id_gen))),
        entry, arg, sz
    )

    # Count active tasks
    atomic_fetch_add_i64(
        ptr[i64](ptr[void](ptr(sched.active_tasks))),
        i64(1)
    )

    # Push to global queue
    taskq_push(
        ptr[TaskQueue](ptr[void](ptr(sched.global_queue))),
        task
    )

    # Wake one parked worker
    sched_notify_one(sched)

    return task


@compile
def sched_notify_one(sched: ptr[Scheduler]) -> void:
    """Wake one parked worker."""
    mutex_lock(ptr[Mutex](ptr[void](ptr(sched.park_mutex))))
    condvar_signal(ptr[CondVar](ptr[void](ptr(sched.park_cond))))
    mutex_unlock(ptr[Mutex](ptr[void](ptr(sched.park_mutex))))


@compile
def sched_notify_all(sched: ptr[Scheduler]) -> void:
    """Wake all parked workers (for shutdown)."""
    mutex_lock(ptr[Mutex](ptr[void](ptr(sched.park_mutex))))
    condvar_broadcast(ptr[CondVar](ptr[void](ptr(sched.park_cond))))
    mutex_unlock(ptr[Mutex](ptr[void](ptr(sched.park_mutex))))


@compile
def sched_current_worker(sched: ptr[Scheduler]) -> ptr[Worker]:
    current: ThreadHandle = thread_current()
    i: i32 = 0
    while i < sched.num_workers:
        w: ptr[Worker] = ptr[Worker](
            ptr[void](
                ptr[u8](ptr[void](sched.workers)) + i64(i) * i64(sizeof(Worker))
            )
        )
        if thread_equal(current, w.thread) != 0:
            return w
        i = i + 1
    return nullptr


# ============================================================
# Worker loop: main function running on each OS thread
# ============================================================

@compile
def worker_find_task(w: ptr[Worker]) -> ptr[Task]:
    """Try to find a task to run.  Search order:
    1. Local deque (pop bottom - LIFO for cache locality)
    2. Global queue (FIFO for fairness)
    3. Steal from random other worker (load balance)
    Returns nullptr if nothing found.
    """
    sched: ptr[Scheduler] = ptr[Scheduler](w.scheduler)

    # 1. Pop from local deque
    task: ptr[Task] = wsdeque_pop(ptr[WSDeque](ptr[void](ptr(w.local_deque))))
    if task != nullptr:
        atomic_store_i32(ptr[i32](ptr[void](ptr(task.state))), TASK_RUNNING)
        return task

    # 2. Check global queue
    task = taskq_pop(ptr[TaskQueue](ptr[void](ptr(sched.global_queue))))
    if task != nullptr:
        atomic_store_i32(ptr[i32](ptr[void](ptr(task.state))), TASK_RUNNING)
        return task

    # 3. Steal from other workers (round-robin starting from next)
    num: i32 = sched.num_workers
    start: i32 = (w.id + i32(1)) % num
    i: i32 = 0
    while i < num - i32(1):
        victim_idx: i32 = (start + i) % num
        victim: ptr[Worker] = ptr[Worker](
            ptr[void](
                ptr[u8](ptr[void](sched.workers)) + i64(victim_idx) * i64(sizeof(Worker))
            )
        )
        task = wsdeque_steal(ptr[WSDeque](ptr[void](ptr(victim.local_deque))))
        if task != nullptr:
            atomic_store_i32(ptr[i32](ptr[void](ptr(task.state))), TASK_RUNNING)
            return task
        i = i + 1

    return nullptr


@compile
def sched_has_active_tasks(sched: ptr[Scheduler]) -> i32:
    if atomic_load_i64(ptr[i64](ptr[void](ptr(sched.active_tasks)))) != i64(0):
        return i32(1)
    return i32(0)


@compile
def sched_should_worker_exit(sched: ptr[Scheduler]) -> i32:
    if (
        atomic_load_i64(ptr[i64](ptr[void](ptr(sched.shutdown)))) != i64(0)
        and sched_has_active_tasks(sched) == 0
        and taskq_is_empty(ptr[TaskQueue](ptr[void](ptr(sched.global_queue)))) != 0
    ):
        return i32(1)
    return i32(0)


@compile
def worker_park(w: ptr[Worker]) -> void:
    """Park this worker: sleep until notified or shutdown."""
    sched: ptr[Scheduler] = ptr[Scheduler](w.scheduler)

    mutex_lock(ptr[Mutex](ptr[void](ptr(sched.park_mutex))))
    w.is_parked = i32(1)

    while (
        taskq_is_empty(ptr[TaskQueue](ptr[void](ptr(sched.global_queue)))) != 0
        and sched_should_worker_exit(sched) == 0
    ):
        condvar_wait(
            ptr[CondVar](ptr[void](ptr(sched.park_cond))),
            ptr[Mutex](ptr[void](ptr(sched.park_mutex)))
        )

    w.is_parked = i32(0)
    mutex_unlock(ptr[Mutex](ptr[void](ptr(sched.park_mutex))))


@compile
def worker_run_task(w: ptr[Worker], task: ptr[Task]) -> void:
    """Execute one task: context switch to it, return when it yields/finishes."""
    atomic_store_i32(ptr[i32](ptr[void](ptr(task.state))), TASK_RUNNING)
    w.current_task = task
    task.scheduler_coro = ptr[Coroutine](ptr[void](ptr(w.scheduler_coro)))

    # Switch from worker's scheduler context to the task's coroutine
    coro_switch(
        ptr[Coroutine](ptr[void](ptr(w.scheduler_coro))),
        task.coro
    )

    # Back here: task either yielded or finished
    w.current_task = nullptr


@compile
def worker_loop(arg: ptr[void]) -> ptr[void]:
    """Main loop for a worker thread.

    Repeatedly: find task -> run it -> handle completion/yield.
    Parks when no work available.
    Exits when shutdown flag is set and no tasks remain.
    """
    w: ptr[Worker] = ptr[Worker](arg)
    sched: ptr[Scheduler] = ptr[Scheduler](w.scheduler)

    while True:
        task: ptr[Task] = worker_find_task(w)
        if task != nullptr:
            worker_run_task(w, task)
            _handle_task_after_run(w, task, sched)
        else:
            if sched_should_worker_exit(sched) != 0:
                return nullptr
            worker_park(w)

    return nullptr


@compile
def _handle_task_after_run(w: ptr[Worker], task: ptr[Task], sched: ptr[Scheduler]) -> void:
    """Handle task state after worker_run_task returns."""
    state: i32 = atomic_load_i32(ptr[i32](ptr[void](ptr(task.state))))
    if state == TASK_PENDING:
        # Task yielded — re-queue for later execution
        # Push to local deque (cache locality: likely to run on same core)
        if wsdeque_push(ptr[WSDeque](ptr[void](ptr(w.local_deque))), task) == 0:
            # Local deque full → overflow to global queue
            taskq_push(ptr[TaskQueue](ptr[void](ptr(sched.global_queue))), task)
    elif state == TASK_WOKEN:
        taskq_push(ptr[TaskQueue](ptr[void](ptr(sched.global_queue))), task)
        sched_notify_one(sched)
    elif state == TASK_BLOCKING:
        spinlock_lock(ptr[SpinLock](ptr[void](ptr(task.lock))))
        latest: i32 = atomic_load_i32(ptr[i32](ptr[void](ptr(task.state))))
        if latest == TASK_BLOCKING:
            atomic_store_i32(ptr[i32](ptr[void](ptr(task.state))), TASK_BLOCKED)
        elif latest == TASK_WOKEN:
            spinlock_unlock(ptr[SpinLock](ptr[void](ptr(task.lock))))
            taskq_push(ptr[TaskQueue](ptr[void](ptr(sched.global_queue))), task)
            sched_notify_one(sched)
            return
        spinlock_unlock(ptr[SpinLock](ptr[void](ptr(task.lock))))
    elif state == TASK_FINISHED:
        # Task finished: decrement active count
        old_active: i64 = atomic_fetch_add_i64(
            ptr[i64](ptr[void](ptr(sched.active_tasks))),
            i64(-1)
        )
        if (
            old_active == i64(1)
            and atomic_load_i64(ptr[i64](ptr[void](ptr(sched.shutdown)))) != i64(0)
        ):
            sched_notify_all(sched)
        if task.detached != 0:
            task_destroy(task)
            return
        # Wake joiner if someone is waiting on this task
        if task.joiner != nullptr:
            sched_requeue_task(sched, task.joiner)


@compile
def sched_requeue_task(sched: ptr[Scheduler], task: ptr[Task]) -> void:
    spinlock_lock(ptr[SpinLock](ptr[void](ptr(task.lock))))
    if atomic_load_i32(ptr[i32](ptr[void](ptr(task.state)))) == TASK_BLOCKING:
        atomic_store_i32(ptr[i32](ptr[void](ptr(task.state))), TASK_WOKEN)
        spinlock_unlock(ptr[SpinLock](ptr[void](ptr(task.lock))))
        return
    atomic_store_i32(ptr[i32](ptr[void](ptr(task.state))), TASK_PENDING)
    spinlock_unlock(ptr[SpinLock](ptr[void](ptr(task.lock))))
    taskq_push(ptr[TaskQueue](ptr[void](ptr(sched.global_queue))), task)
    sched_notify_one(sched)


# ============================================================
# Yield: cooperative suspension from within a task
# ============================================================

@compile
def sched_yield(w: ptr[Worker]) -> void:
    """Yield the current task back to the scheduler.

    Called from within a running task.  Switches back to the
    worker's scheduler context.  The task will be re-queued.
    """
    task: ptr[Task] = w.current_task
    if task == nullptr:
        return

    atomic_store_i32(ptr[i32](ptr[void](ptr(task.state))), TASK_PENDING)

    # Switch back to worker's scheduler context
    coro_switch(
        task.coro,
        ptr[Coroutine](ptr[void](ptr(w.scheduler_coro)))
    )
    # Execution resumes here when the task is scheduled again


@compile
def sched_block_current(w: ptr[Worker]) -> void:
    """Block current task without automatically re-queueing it."""
    task: ptr[Task] = w.current_task
    if task == nullptr:
        return

    atomic_store_i32(ptr[i32](ptr[void](ptr(task.state))), TASK_BLOCKED)
    sched_suspend_current(w)


@compile
def sched_suspend_current(w: ptr[Worker]) -> void:
    """Switch current task back to the scheduler with its existing state."""
    task: ptr[Task] = w.current_task
    if task == nullptr:
        return

    coro_switch(
        task.coro,
        ptr[Coroutine](ptr[void](ptr(w.scheduler_coro)))
    )


# ============================================================
# Join: block current task until target finishes
# ============================================================

@compile
def sched_join(w: ptr[Worker], target: ptr[Task]) -> ptr[void]:
    """Block current task until target task finishes.

    If target is already finished, returns immediately.
    Otherwise, registers current task as joiner and yields.
    """
    # Fast path: already done
    if atomic_load_i32(ptr[i32](ptr[void](ptr(target.state)))) == TASK_FINISHED:
        return target.result

    # Slow path: register as joiner and suspend
    current: ptr[Task] = w.current_task
    if current == nullptr:
        # Not in a task context (main thread?) → spin-wait
        while atomic_load_i32(ptr[i32](ptr[void](ptr(target.state)))) != TASK_FINISHED:
            pass
        return target.result

    # Set joiner atomically
    spinlock_lock(ptr[SpinLock](ptr[void](ptr(target.lock))))
    if atomic_load_i32(ptr[i32](ptr[void](ptr(target.state)))) == TASK_FINISHED:
        spinlock_unlock(ptr[SpinLock](ptr[void](ptr(target.lock))))
        return target.result
    target.joiner = current
    spinlock_lock(ptr[SpinLock](ptr[void](ptr(current.lock))))
    atomic_store_i32(ptr[i32](ptr[void](ptr(current.state))), TASK_BLOCKING)
    spinlock_unlock(ptr[SpinLock](ptr[void](ptr(current.lock))))
    spinlock_unlock(ptr[SpinLock](ptr[void](ptr(target.lock))))

    # Yield — woken when target finishes
    coro_switch(
        current.coro,
        ptr[Coroutine](ptr[void](ptr(w.scheduler_coro)))
    )

    # Resumed: target is now finished
    return target.result
