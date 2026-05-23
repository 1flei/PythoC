#!/usr/bin/env python3
"""
Test runtime scheduler: task queue, work-stealing deque, spinlock, atomics.

These tests exercise the core scheduling data structures in single-threaded
mode (no OS threads), verifying correctness of push/pop/steal operations,
FIFO/LIFO ordering, and edge cases.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import unittest
from pythoc.decorators.compile import compile
from pythoc.builtin_entities import void, i32, i64, u64, ptr, nullptr, sizeof
from pythoc.libc.stdlib import malloc, free
from pythoc.libc.string import memset
from pythoc.build.output_manager import flush_all_pending_outputs

from test.utils.test_utils import DeferredTestCase, expect_error

from pythoc.std.runtime.platform import (
    SpinLock, spinlock_init, spinlock_lock, spinlock_unlock,
    atomic_load_i64, atomic_store_i64, atomic_fetch_add_i64, atomic_cas_i64,
    MiniCtx, MINI_CTX_SIZE, IS_X86_64, IS_WINDOWS, IS_AARCH64,
)
from pythoc.std.runtime.task import (
    Task, TaskHandle, TaskQueue, TaskIdGen,
    task_create, task_destroy,
    taskq_init, taskq_push, taskq_pop, taskq_is_empty,
    TASK_PENDING, TASK_FINISHED,
)
from pythoc.std.runtime.deque import (
    WSDeque, wsdeque_init, wsdeque_push, wsdeque_pop, wsdeque_steal, wsdeque_size,
)
from pythoc.std.runtime.api import (
    Runtime, runtime_new, runtime_start, runtime_spawn, runtime_join,
    runtime_shutdown, runtime_free, runtime_detach, runtime_yield_now,
)
from pythoc.std.runtime.executor_effect import (
    executor_set_runtime, _exec_spawn, _exec_join,
)


# ============================================================
# Atomic operation tests
# ============================================================

@compile(suffix="rt_atomic_store_load")
def test_fn_atomic_store_load() -> i64:
    """Store then load atomically."""
    val: i64 = 0
    atomic_store_i64(ptr[i64](ptr[void](ptr(val))), i64(42))
    return atomic_load_i64(ptr[i64](ptr[void](ptr(val))))


@compile(suffix="rt_atomic_fetch_add")
def test_fn_atomic_fetch_add() -> i64:
    """Fetch-add returns old value, increments storage."""
    val: i64 = 10
    old: i64 = atomic_fetch_add_i64(ptr[i64](ptr[void](ptr(val))), i64(5))
    # old should be 10, val should now be 15
    return old * i64(100) + val  # 1015


@compile(suffix="rt_atomic_cas_success")
def test_fn_atomic_cas_success() -> i64:
    """CAS succeeds when expected matches current value."""
    val: i64 = 7
    expected: i64 = 7
    result: i32 = atomic_cas_i64(
        ptr[i64](ptr[void](ptr(val))),
        ptr[i64](ptr[void](ptr(expected))),
        i64(99)
    )
    # result=1 (success), val=99, expected unchanged
    return i64(result) * i64(1000) + val  # 1099


@compile(suffix="rt_atomic_cas_failure")
def test_fn_atomic_cas_failure() -> i64:
    """CAS fails when expected doesn't match; expected is updated."""
    val: i64 = 7
    expected: i64 = 999  # wrong expectation
    result: i32 = atomic_cas_i64(
        ptr[i64](ptr[void](ptr(val))),
        ptr[i64](ptr[void](ptr(expected))),
        i64(99)
    )
    # result=0 (failure), val unchanged=7, expected updated to 7
    return i64(result) * i64(10000) + expected * i64(100) + val  # 0*10000 + 700 + 7 = 707


# ============================================================
# Spinlock tests
# ============================================================

@compile(suffix="rt_spinlock_basic")
def test_fn_spinlock_basic() -> i32:
    """Lock and unlock spinlock, verify state transitions."""
    lock: SpinLock
    spinlock_init(ptr[SpinLock](ptr[void](ptr(lock))))

    # Initially unlocked
    if lock.state != i64(0):
        return i32(-1)

    spinlock_lock(ptr[SpinLock](ptr[void](ptr(lock))))
    # Now locked
    if lock.state != i64(1):
        return i32(-2)

    spinlock_unlock(ptr[SpinLock](ptr[void](ptr(lock))))
    # Unlocked again
    if lock.state != i64(0):
        return i32(-3)

    return i32(1)


@compile(suffix="rt_spinlock_reentrant_test")
def test_fn_spinlock_double_lock_unlock() -> i32:
    """Lock, unlock, lock again — verify re-acquisition works."""
    lock: SpinLock
    spinlock_init(ptr[SpinLock](ptr[void](ptr(lock))))

    spinlock_lock(ptr[SpinLock](ptr[void](ptr(lock))))
    spinlock_unlock(ptr[SpinLock](ptr[void](ptr(lock))))
    spinlock_lock(ptr[SpinLock](ptr[void](ptr(lock))))
    spinlock_unlock(ptr[SpinLock](ptr[void](ptr(lock))))
    return i32(1)


# ============================================================
# TaskQueue tests
# ============================================================

@compile(suffix="rt_taskq_init")
def test_fn_taskq_init() -> i32:
    """Freshly initialized queue is empty."""
    q: TaskQueue
    taskq_init(ptr[TaskQueue](ptr[void](ptr(q))))
    return taskq_is_empty(ptr[TaskQueue](ptr[void](ptr(q))))


@compile(suffix="rt_taskq_push_pop")
def test_fn_taskq_push_pop() -> i32:
    """Push then pop — returns same task."""
    q: TaskQueue
    taskq_init(ptr[TaskQueue](ptr[void](ptr(q))))

    # Create a dummy task (just need a valid ptr)
    id_gen: TaskIdGen
    id_gen.counter = i64(1)

    # Use a minimal "fake task" — just need unique pointer identity
    task: ptr[Task] = ptr[Task](malloc(i64(sizeof(Task))))
    memset(ptr[void](task), 0, i64(sizeof(Task)))
    task.id = u64(42)

    taskq_push(ptr[TaskQueue](ptr[void](ptr(q))), task)
    popped: ptr[Task] = taskq_pop(ptr[TaskQueue](ptr[void](ptr(q))))

    result: i32 = 0
    if popped == task:
        result = 1

    free(ptr[void](task))
    return result


@compile(suffix="rt_taskq_empty_pop")
def test_fn_taskq_empty_pop() -> i32:
    """Pop from empty queue returns nullptr."""
    q: TaskQueue
    taskq_init(ptr[TaskQueue](ptr[void](ptr(q))))
    popped: ptr[Task] = taskq_pop(ptr[TaskQueue](ptr[void](ptr(q))))
    if popped == nullptr:
        return i32(1)
    return i32(0)


@compile(suffix="rt_taskq_fifo_order")
def test_fn_taskq_fifo_order() -> i32:
    """Push A, B, C — pop returns A, B, C (FIFO)."""
    q: TaskQueue
    taskq_init(ptr[TaskQueue](ptr[void](ptr(q))))

    a: ptr[Task] = ptr[Task](malloc(i64(sizeof(Task))))
    b: ptr[Task] = ptr[Task](malloc(i64(sizeof(Task))))
    c: ptr[Task] = ptr[Task](malloc(i64(sizeof(Task))))
    memset(ptr[void](a), 0, i64(sizeof(Task)))
    memset(ptr[void](b), 0, i64(sizeof(Task)))
    memset(ptr[void](c), 0, i64(sizeof(Task)))
    a.id = u64(1)
    b.id = u64(2)
    c.id = u64(3)

    taskq_push(ptr[TaskQueue](ptr[void](ptr(q))), a)
    taskq_push(ptr[TaskQueue](ptr[void](ptr(q))), b)
    taskq_push(ptr[TaskQueue](ptr[void](ptr(q))), c)

    p1: ptr[Task] = taskq_pop(ptr[TaskQueue](ptr[void](ptr(q))))
    p2: ptr[Task] = taskq_pop(ptr[TaskQueue](ptr[void](ptr(q))))
    p3: ptr[Task] = taskq_pop(ptr[TaskQueue](ptr[void](ptr(q))))

    result: i32 = 0
    if p1.id == u64(1) and p2.id == u64(2) and p3.id == u64(3):
        result = 1

    free(ptr[void](a))
    free(ptr[void](b))
    free(ptr[void](c))
    return result


# ============================================================
# Work-stealing deque tests
# ============================================================

@compile(suffix="rt_wsdeque_init_empty")
def test_fn_wsdeque_init_empty() -> i64:
    """Fresh deque has size 0."""
    dq: WSDeque
    wsdeque_init(ptr[WSDeque](ptr[void](ptr(dq))))
    return wsdeque_size(ptr[WSDeque](ptr[void](ptr(dq))))


@compile(suffix="rt_wsdeque_push_pop_lifo")
def test_fn_wsdeque_push_pop_lifo() -> i32:
    """Owner push/pop is LIFO: push A,B,C → pop C,B,A."""
    dq: WSDeque
    wsdeque_init(ptr[WSDeque](ptr[void](ptr(dq))))

    a: ptr[Task] = ptr[Task](malloc(i64(sizeof(Task))))
    b: ptr[Task] = ptr[Task](malloc(i64(sizeof(Task))))
    c: ptr[Task] = ptr[Task](malloc(i64(sizeof(Task))))
    memset(ptr[void](a), 0, i64(sizeof(Task)))
    memset(ptr[void](b), 0, i64(sizeof(Task)))
    memset(ptr[void](c), 0, i64(sizeof(Task)))
    a.id = u64(1)
    b.id = u64(2)
    c.id = u64(3)

    wsdeque_push(ptr[WSDeque](ptr[void](ptr(dq))), a)
    wsdeque_push(ptr[WSDeque](ptr[void](ptr(dq))), b)
    wsdeque_push(ptr[WSDeque](ptr[void](ptr(dq))), c)

    p1: ptr[Task] = wsdeque_pop(ptr[WSDeque](ptr[void](ptr(dq))))
    p2: ptr[Task] = wsdeque_pop(ptr[WSDeque](ptr[void](ptr(dq))))
    p3: ptr[Task] = wsdeque_pop(ptr[WSDeque](ptr[void](ptr(dq))))

    result: i32 = 0
    if p1.id == u64(3) and p2.id == u64(2) and p3.id == u64(1):
        result = 1

    free(ptr[void](a))
    free(ptr[void](b))
    free(ptr[void](c))
    return result


@compile(suffix="rt_wsdeque_steal_fifo")
def test_fn_wsdeque_steal_fifo() -> i32:
    """Steal takes from top (FIFO): push A,B,C → steal A,B,C."""
    dq: WSDeque
    wsdeque_init(ptr[WSDeque](ptr[void](ptr(dq))))

    a: ptr[Task] = ptr[Task](malloc(i64(sizeof(Task))))
    b: ptr[Task] = ptr[Task](malloc(i64(sizeof(Task))))
    c: ptr[Task] = ptr[Task](malloc(i64(sizeof(Task))))
    memset(ptr[void](a), 0, i64(sizeof(Task)))
    memset(ptr[void](b), 0, i64(sizeof(Task)))
    memset(ptr[void](c), 0, i64(sizeof(Task)))
    a.id = u64(1)
    b.id = u64(2)
    c.id = u64(3)

    wsdeque_push(ptr[WSDeque](ptr[void](ptr(dq))), a)
    wsdeque_push(ptr[WSDeque](ptr[void](ptr(dq))), b)
    wsdeque_push(ptr[WSDeque](ptr[void](ptr(dq))), c)

    s1: ptr[Task] = wsdeque_steal(ptr[WSDeque](ptr[void](ptr(dq))))
    s2: ptr[Task] = wsdeque_steal(ptr[WSDeque](ptr[void](ptr(dq))))
    s3: ptr[Task] = wsdeque_steal(ptr[WSDeque](ptr[void](ptr(dq))))

    result: i32 = 0
    if s1.id == u64(1) and s2.id == u64(2) and s3.id == u64(3):
        result = 1

    free(ptr[void](a))
    free(ptr[void](b))
    free(ptr[void](c))
    return result


@compile(suffix="rt_wsdeque_pop_empty")
def test_fn_wsdeque_pop_empty() -> i32:
    """Pop from empty deque returns nullptr."""
    dq: WSDeque
    wsdeque_init(ptr[WSDeque](ptr[void](ptr(dq))))
    result: ptr[Task] = wsdeque_pop(ptr[WSDeque](ptr[void](ptr(dq))))
    if result == nullptr:
        return i32(1)
    return i32(0)


@compile(suffix="rt_wsdeque_steal_empty")
def test_fn_wsdeque_steal_empty() -> i32:
    """Steal from empty deque returns nullptr."""
    dq: WSDeque
    wsdeque_init(ptr[WSDeque](ptr[void](ptr(dq))))
    result: ptr[Task] = wsdeque_steal(ptr[WSDeque](ptr[void](ptr(dq))))
    if result == nullptr:
        return i32(1)
    return i32(0)


@compile(suffix="rt_wsdeque_push_full")
def test_fn_wsdeque_push_full() -> i32:
    """Push to a full deque returns 0."""
    dq: WSDeque
    wsdeque_init(ptr[WSDeque](ptr[void](ptr(dq))))

    # Fill to capacity (4096)
    task: ptr[Task] = ptr[Task](malloc(i64(sizeof(Task))))
    memset(ptr[void](task), 0, i64(sizeof(Task)))

    i: i32 = 0
    while i < 4096:
        wsdeque_push(ptr[WSDeque](ptr[void](ptr(dq))), task)
        i = i + 1

    # Next push should fail
    result: i32 = wsdeque_push(ptr[WSDeque](ptr[void](ptr(dq))), task)

    free(ptr[void](task))
    # result should be 0 (full)
    if result == 0:
        return i32(1)
    return i32(0)


@compile(suffix="rt_wsdeque_size_tracking")
def test_fn_wsdeque_size_tracking() -> i64:
    """Size tracks pushes and pops correctly."""
    dq: WSDeque
    wsdeque_init(ptr[WSDeque](ptr[void](ptr(dq))))

    task: ptr[Task] = ptr[Task](malloc(i64(sizeof(Task))))
    memset(ptr[void](task), 0, i64(sizeof(Task)))

    wsdeque_push(ptr[WSDeque](ptr[void](ptr(dq))), task)
    wsdeque_push(ptr[WSDeque](ptr[void](ptr(dq))), task)
    wsdeque_push(ptr[WSDeque](ptr[void](ptr(dq))), task)
    # size = 3

    wsdeque_pop(ptr[WSDeque](ptr[void](ptr(dq))))
    # size = 2

    size: i64 = wsdeque_size(ptr[WSDeque](ptr[void](ptr(dq))))
    free(ptr[void](task))
    return size  # 2


# ============================================================
# Runtime lifecycle tests
# ============================================================

@compile(suffix="rt_spawn_join_entry")
def test_fn_runtime_entry(arg: ptr[void]) -> ptr[void]:
    out: ptr[i64] = ptr[i64](malloc(i64(sizeof(i64))))
    out[0] = i64(1234)
    return ptr[void](out)


@expect_error(["not consumed"], suffix="bad_task_handle_not_consumed")
def run_error_task_handle_not_consumed():
    @compile(suffix="bad_task_handle_not_consumed")
    def bad_task_handle_not_consumed() -> void:
        rt: ptr[Runtime] = nullptr
        task: TaskHandle = runtime_spawn(
            rt, test_fn_runtime_entry, nullptr, u64(0)
        )


@expect_error(["consumed"], suffix="bad_task_handle_double_join")
def run_error_task_handle_double_join():
    @compile(suffix="bad_task_handle_double_join")
    def bad_task_handle_double_join() -> void:
        rt: ptr[Runtime] = nullptr
        task: TaskHandle = runtime_spawn(
            rt, test_fn_runtime_entry, nullptr, u64(0)
        )
        runtime_join(rt, task)
        runtime_join(rt, task)


@expect_error(["cannot assign linear token", "use move"], suffix="bad_task_handle_copy")
def run_error_task_handle_copy():
    @compile(suffix="bad_task_handle_copy")
    def bad_task_handle_copy() -> void:
        rt: ptr[Runtime] = nullptr
        task: TaskHandle = runtime_spawn(
            rt, test_fn_runtime_entry, nullptr, u64(0)
        )
        copied = task
        runtime_detach(rt, copied)


@expect_error(
    ["not consumed", "inconsistent"],
    suffix="bad_task_handle_branch",
)
def run_error_task_handle_branch():
    @compile(suffix="bad_task_handle_branch")
    def bad_task_handle_branch(flag: i32) -> void:
        rt: ptr[Runtime] = nullptr
        task: TaskHandle = runtime_spawn(
            rt, test_fn_runtime_entry, nullptr, u64(0)
        )
        if flag != 0:
            runtime_detach(rt, task)


@compile(suffix="rt_spawn_join_basic")
def test_fn_runtime_spawn_join() -> i64:
    rt = runtime_new(i32(1))
    runtime_start(rt)

    task: TaskHandle = runtime_spawn(
        rt, test_fn_runtime_entry, nullptr, u64(0)
    )
    result: ptr[void] = runtime_join(rt, task)
    out: ptr[i64] = ptr[i64](result)
    value: i64 = out[0]
    free(result)

    runtime_shutdown(rt)
    runtime_free(rt)
    return value


@compile(suffix="rt_join_many")
def test_fn_runtime_join_many() -> i64:
    rt = runtime_new(i32(1))
    runtime_start(rt)

    total: i64 = 0
    i: i32 = 0
    while i < 128:
        task: TaskHandle = runtime_spawn(
            rt, test_fn_runtime_entry, nullptr, u64(0)
        )
        result: ptr[void] = runtime_join(rt, task)
        out: ptr[i64] = ptr[i64](result)
        total = total + out[0]
        free(result)
        i = i + 1

    runtime_shutdown(rt)
    runtime_free(rt)
    return total


@compile(suffix="rt_yield_entry")
def test_fn_runtime_yield_entry(arg: ptr[void]) -> ptr[void]:
    runtime_yield_now(ptr[Runtime](arg))
    out: ptr[i64] = ptr[i64](malloc(i64(sizeof(i64))))
    out[0] = i64(5678)
    return ptr[void](out)


@compile(suffix="rt_yield_resume")
def test_fn_runtime_yield_resume() -> i64:
    rt = runtime_new(i32(1))
    runtime_start(rt)

    task: TaskHandle = runtime_spawn(
        rt, test_fn_runtime_yield_entry, ptr[void](rt), u64(0)
    )
    result: ptr[void] = runtime_join(rt, task)
    out: ptr[i64] = ptr[i64](result)
    value: i64 = out[0]
    free(result)

    runtime_shutdown(rt)
    runtime_free(rt)
    return value


@compile(suffix="rt_detach_entry")
def test_fn_runtime_detach_entry(arg: ptr[void]) -> ptr[void]:
    out: ptr[i64] = ptr[i64](arg)
    out[0] = i64(2468)
    return nullptr


@compile(suffix="rt_detach_shutdown")
def test_fn_runtime_detach_shutdown() -> i64:
    rt = runtime_new(i32(1))
    runtime_start(rt)

    value: i64 = 0
    task: TaskHandle = runtime_spawn(
        rt, test_fn_runtime_detach_entry, ptr[void](ptr(value)), u64(0)
    )
    runtime_detach(rt, task)

    runtime_shutdown(rt)
    runtime_free(rt)
    return value


@compile(suffix="rt_executor_entry")
def test_fn_runtime_executor_entry(arg: ptr[void]) -> ptr[void]:
    out: ptr[i64] = ptr[i64](malloc(i64(sizeof(i64))))
    out[0] = i64(1357)
    return ptr[void](out)


@compile(suffix="rt_executor_global")
def test_fn_runtime_executor_global() -> i64:
    rt = runtime_new(i32(1))
    executor_set_runtime(rt)
    runtime_start(rt)

    task: TaskHandle = _exec_spawn(test_fn_runtime_executor_entry, nullptr)
    result: ptr[void] = _exec_join(task)
    out: ptr[i64] = ptr[i64](result)
    value: i64 = out[0]
    free(result)

    runtime_shutdown(rt)
    runtime_free(rt)
    return value


@compile
class RuntimeStressArgs:
    rt: ptr[Runtime]
    counter: ptr[i64]


@compile(suffix="rt_stress_yield_entry")
def test_fn_runtime_stress_yield_entry(arg: ptr[void]) -> ptr[void]:
    args: ptr[RuntimeStressArgs] = ptr[RuntimeStressArgs](arg)
    atomic_fetch_add_i64(args.counter, i64(1))
    runtime_yield_now(args.rt)
    atomic_fetch_add_i64(args.counter, i64(1))
    return nullptr


@compile(suffix="rt_four_worker_shutdown_drain")
def test_fn_runtime_four_worker_shutdown_drain() -> i64:
    rt = runtime_new(i32(4))
    counter: i64 = 0

    args: RuntimeStressArgs
    args.rt = rt
    args.counter = ptr[i64](ptr[void](ptr(counter)))

    runtime_start(rt)

    i: i32 = 0
    while i < 64:
        task: TaskHandle = runtime_spawn(
            rt, test_fn_runtime_stress_yield_entry, ptr[void](ptr(args)), u64(0)
        )
        runtime_detach(rt, task)
        i = i + 1

    runtime_shutdown(rt)
    runtime_free(rt)
    return counter


@compile
class RuntimeJoinArgs:
    rt: ptr[Runtime]


@compile(suffix="rt_join_child_entry")
def test_fn_runtime_join_child_entry(arg: ptr[void]) -> ptr[void]:
    out: ptr[i64] = ptr[i64](malloc(i64(sizeof(i64))))
    out[0] = i64(4321)
    return ptr[void](out)


@compile(suffix="rt_join_parent_entry")
def test_fn_runtime_join_parent_entry(arg: ptr[void]) -> ptr[void]:
    args: ptr[RuntimeJoinArgs] = ptr[RuntimeJoinArgs](arg)
    child: TaskHandle = runtime_spawn(
        args.rt, test_fn_runtime_join_child_entry, nullptr, u64(0)
    )
    return runtime_join(args.rt, child)


@compile(suffix="rt_join_inside_task_multi_worker")
def test_fn_runtime_join_inside_task_multi_worker() -> i64:
    rt = runtime_new(i32(4))
    args: RuntimeJoinArgs
    args.rt = rt

    runtime_start(rt)
    parent: TaskHandle = runtime_spawn(
        rt, test_fn_runtime_join_parent_entry, ptr[void](ptr(args)), u64(0)
    )
    result: ptr[void] = runtime_join(rt, parent)
    out: ptr[i64] = ptr[i64](result)
    value: i64 = out[0]
    free(result)

    runtime_shutdown(rt)
    runtime_free(rt)
    return value


# ============================================================
# Test class
# ============================================================

class TestRuntimeScheduler(DeferredTestCase):
    """Tests for runtime scheduling data structures."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        flush_all_pending_outputs()

    # --- Platform / context sizing ---

    def test_mini_ctx_size_matches_asm_layout(self):
        """MiniCtx must be large enough for the asm context-switch layout.

        Regression: Win64 ABI saves rdi/rsi (and xmm6-xmm15) in addition to
        the System V callee-saved set. AArch64 also needs d8-d15. If MiniCtx
        is undersized, ctx_swap writes past the struct, corrupting adjacent
        Coroutine fields (heap corruption / access violation at runtime).
        """
        if IS_X86_64 and IS_WINDOWS:
            # GP: rsp, rbp, rbx, r12-r15, rdi, rsi, rip = 80 bytes
            # XMM: xmm6..xmm15 = 10 * 16 = 160 bytes (offset 80..240)
            self.assertGreaterEqual(MINI_CTX_SIZE, 240)
        elif IS_X86_64:
            # rsp, rbp, rbx, r12-r15, rip = 8 u64
            self.assertGreaterEqual(MINI_CTX_SIZE, 64)
        elif IS_AARCH64:
            # GP: sp, x19-x28, x29, x30 = 13 u64 (104 bytes)
            # SIMD: d8..d15 = 8 u64 (64 bytes), total 168 bytes
            self.assertGreaterEqual(MINI_CTX_SIZE, 168)

    # --- Atomics ---

    def test_atomic_store_load(self):
        self.assertEqual(test_fn_atomic_store_load(), 42)

    def test_atomic_fetch_add(self):
        self.assertEqual(test_fn_atomic_fetch_add(), 1015)

    def test_atomic_cas_success(self):
        self.assertEqual(test_fn_atomic_cas_success(), 1099)

    def test_atomic_cas_failure(self):
        self.assertEqual(test_fn_atomic_cas_failure(), 707)

    # --- Spinlock ---

    def test_spinlock_basic(self):
        self.assertEqual(test_fn_spinlock_basic(), 1)

    def test_spinlock_double_lock_unlock(self):
        self.assertEqual(test_fn_spinlock_double_lock_unlock(), 1)

    # --- TaskQueue ---

    def test_taskq_init(self):
        self.assertEqual(test_fn_taskq_init(), 1)

    def test_taskq_push_pop(self):
        self.assertEqual(test_fn_taskq_push_pop(), 1)

    def test_taskq_empty_pop(self):
        self.assertEqual(test_fn_taskq_empty_pop(), 1)

    def test_taskq_fifo_order(self):
        self.assertEqual(test_fn_taskq_fifo_order(), 1)

    # --- Work-stealing deque ---

    def test_wsdeque_init_empty(self):
        self.assertEqual(test_fn_wsdeque_init_empty(), 0)

    def test_wsdeque_push_pop_lifo(self):
        self.assertEqual(test_fn_wsdeque_push_pop_lifo(), 1)

    def test_wsdeque_steal_fifo(self):
        self.assertEqual(test_fn_wsdeque_steal_fifo(), 1)

    def test_wsdeque_pop_empty(self):
        self.assertEqual(test_fn_wsdeque_pop_empty(), 1)

    def test_wsdeque_steal_empty(self):
        self.assertEqual(test_fn_wsdeque_steal_empty(), 1)

    def test_wsdeque_push_full(self):
        self.assertEqual(test_fn_wsdeque_push_full(), 1)

    def test_wsdeque_size_tracking(self):
        self.assertEqual(test_fn_wsdeque_size_tracking(), 2)

    # --- Runtime lifecycle ---

    def test_runtime_spawn_join(self):
        self.assertEqual(test_fn_runtime_spawn_join(), 1234)

    def test_runtime_join_many(self):
        self.assertEqual(test_fn_runtime_join_many(), 157952)

    def test_runtime_yield_resume(self):
        self.assertEqual(test_fn_runtime_yield_resume(), 5678)

    def test_runtime_detach_shutdown(self):
        self.assertEqual(test_fn_runtime_detach_shutdown(), 2468)

    def test_runtime_executor_global(self):
        self.assertEqual(test_fn_runtime_executor_global(), 1357)

    def test_runtime_four_worker_shutdown_drain(self):
        self.assertEqual(test_fn_runtime_four_worker_shutdown_drain(), 128)

    def test_runtime_join_inside_task_multi_worker(self):
        self.assertEqual(test_fn_runtime_join_inside_task_multi_worker(), 4321)

    def test_task_handle_not_consumed_error(self):
        passed, msg = run_error_task_handle_not_consumed()
        self.assertTrue(passed, msg)

    def test_task_handle_double_join_error(self):
        passed, msg = run_error_task_handle_double_join()
        self.assertTrue(passed, msg)

    def test_task_handle_copy_error(self):
        passed, msg = run_error_task_handle_copy()
        self.assertTrue(passed, msg)

    def test_task_handle_branch_error(self):
        passed, msg = run_error_task_handle_branch()
        self.assertTrue(passed, msg)


if __name__ == '__main__':
    unittest.main()
