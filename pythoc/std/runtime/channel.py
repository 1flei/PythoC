"""
Channel: Bounded MPMC channel for inter-task communication.

A channel is a fixed-capacity ring buffer that allows tasks
to send and receive messages.  When the buffer is full, senders
block (yield to scheduler).  When empty, receivers block.

Design (PythoC-ic):
- Plain struct: ring buffer + head/tail + sender/receiver wait queues
- No hidden allocations beyond initial capacity
- Blocking = yield to scheduler (not OS block)
- Generic via Python factory function (like Vector)

Architecture:
    +--[ ring buffer: array[T, capacity] ]--+
    |                                        |
    head --->  [ slot | slot | ... | slot ] <--- tail
    |                                        |
    +--- send_waiters (blocked senders) -----+
    +--- recv_waiters (blocked receivers) ---+
"""
from __future__ import annotations

from .policy import bind_mem
bind_mem()

from pythoc import compile, effect, i32, i64, u64, ptr, void, struct, nullptr, sizeof, array
from pythoc.libc.string import memset, memcpy

from .platform import (
    SpinLock, spinlock_init, spinlock_lock, spinlock_unlock, atomic_store_i32,
)
from .task import (
    Task, TaskQueue, taskq_init, taskq_push, taskq_pop,
    TASK_BLOCKING, TASK_PENDING,
)
from .scheduler import (
    Worker, Scheduler, sched_current_worker, sched_suspend_current,
    sched_requeue_task,
)


# ============================================================
# Channel factory: generates type-specialized channels
#
# Usage:
#   Ch_i32 = Channel(i32, capacity=64)
#   ch = Ch_i32.create()
#   Ch_i32.send(ch, value)
#   value = Ch_i32.recv(ch)
#   Ch_i32.destroy(ch)
# ============================================================

def Channel(element_type, capacity=64):
    """Generate a typed bounded channel.

    This is a Python-time factory (metaprogramming) that produces
    PythoC @compile functions specialized for the element type.

    Args:
        element_type: PythoC type for channel elements (e.g., i32, ptr[void])
        capacity: maximum number of buffered messages

    Returns:
        SimpleNamespace with:
            .type       - the Channel struct type
            .create()   - allocate and init a channel
            .destroy()  - free a channel
            .send()     - send a value (blocks if full)
            .recv()     - receive a value (blocks if empty)
            .try_send() - non-blocking send (returns 0/1)
            .try_recv() - non-blocking receive (returns 0/1)
    """
    from types import SimpleNamespace

    type_suffix = (element_type, capacity)

    # ---- Channel struct ----
    @compile(suffix=type_suffix)
    class _Channel:
        buffer: array[element_type, capacity]   # ring buffer
        head: u64                               # read position
        tail: u64                               # write position
        count: u64                              # current number of items
        cap: u64                                # capacity (= capacity)
        lock: SpinLock                          # protects all fields
        send_waiters: TaskQueue                 # tasks blocked on send
        recv_waiters: TaskQueue                 # tasks blocked on recv
        scheduler: ptr[void]                    # scheduler for blocked waiters
        closed: i32                             # 1 if channel is closed

    # ---- create ----
    @compile(suffix=type_suffix)
    def channel_create() -> ptr[_Channel]:
        """Allocate and initialize a channel."""
        ch: ptr[_Channel] = ptr[_Channel](effect.mem.malloc(u64(sizeof(_Channel))))
        memset(ptr[void](ch), 0, i64(sizeof(_Channel)))
        ch.head = u64(0)
        ch.tail = u64(0)
        ch.count = u64(0)
        ch.cap = u64(capacity)
        ch.closed = i32(0)
        ch.scheduler = nullptr
        spinlock_init(ptr[SpinLock](ptr[void](ptr(ch.lock))))
        taskq_init(ptr[TaskQueue](ptr[void](ptr(ch.send_waiters))))
        taskq_init(ptr[TaskQueue](ptr[void](ptr(ch.recv_waiters))))
        return ch

    # ---- destroy ----
    @compile(suffix=type_suffix)
    def channel_destroy(ch: ptr[_Channel]) -> void:
        """Destroy channel and free memory."""
        effect.mem.free(ptr[void](ch))

    # ---- try_send (non-blocking) ----
    @compile(suffix=type_suffix)
    def channel_try_send(ch: ptr[_Channel], value: element_type) -> i32:
        """Try to send without blocking.  Returns 1 on success, 0 if full."""
        spinlock_lock(ptr[SpinLock](ptr[void](ptr(ch.lock))))

        if ch.count >= ch.cap:
            spinlock_unlock(ptr[SpinLock](ptr[void](ptr(ch.lock))))
            return i32(0)

        # Write to buffer at tail position
        idx: u64 = ch.tail % ch.cap
        ch.buffer[idx] = value
        ch.tail = ch.tail + u64(1)
        ch.count = ch.count + u64(1)

        # Wake one receiver if any are waiting
        waiter: ptr[Task] = taskq_pop(ptr[TaskQueue](ptr[void](ptr(ch.recv_waiters))))
        sched: ptr[Scheduler] = ptr[Scheduler](ch.scheduler)

        spinlock_unlock(ptr[SpinLock](ptr[void](ptr(ch.lock))))

        # Re-queue woken receiver (outside lock)
        if waiter != nullptr:
            if sched != nullptr:
                sched_requeue_task(sched, waiter)
            else:
                atomic_store_i32(
                    ptr[i32](ptr[void](ptr(waiter.state))), TASK_PENDING
                )

        return i32(1)

    # ---- try_recv (non-blocking) ----
    @compile(suffix=type_suffix)
    def channel_try_recv(ch: ptr[_Channel], out: ptr[element_type]) -> i32:
        """Try to receive without blocking.  Returns 1 on success, 0 if empty."""
        spinlock_lock(ptr[SpinLock](ptr[void](ptr(ch.lock))))

        if ch.count == u64(0):
            spinlock_unlock(ptr[SpinLock](ptr[void](ptr(ch.lock))))
            return i32(0)

        # Read from buffer at head position
        idx: u64 = ch.head % ch.cap
        out[0] = ch.buffer[idx]
        ch.head = ch.head + u64(1)
        ch.count = ch.count - u64(1)

        # Wake one sender if any are waiting
        waiter: ptr[Task] = taskq_pop(ptr[TaskQueue](ptr[void](ptr(ch.send_waiters))))
        sched: ptr[Scheduler] = ptr[Scheduler](ch.scheduler)

        spinlock_unlock(ptr[SpinLock](ptr[void](ptr(ch.lock))))

        if waiter != nullptr:
            if sched != nullptr:
                sched_requeue_task(sched, waiter)
            else:
                atomic_store_i32(
                    ptr[i32](ptr[void](ptr(waiter.state))), TASK_PENDING
                )

        return i32(1)

    # ---- close ----
    @compile(suffix=type_suffix)
    def channel_close(ch: ptr[_Channel]) -> void:
        """Close the channel.  Wakes all blocked senders/receivers."""
        spinlock_lock(ptr[SpinLock](ptr[void](ptr(ch.lock))))
        ch.closed = i32(1)
        sched: ptr[Scheduler] = ptr[Scheduler](ch.scheduler)

        # Wake all waiters
        waiter: ptr[Task] = taskq_pop(ptr[TaskQueue](ptr[void](ptr(ch.send_waiters))))
        while waiter != nullptr:
            if sched != nullptr:
                sched_requeue_task(sched, waiter)
            else:
                atomic_store_i32(
                    ptr[i32](ptr[void](ptr(waiter.state))), TASK_PENDING
                )
            waiter = taskq_pop(ptr[TaskQueue](ptr[void](ptr(ch.send_waiters))))

        waiter = taskq_pop(ptr[TaskQueue](ptr[void](ptr(ch.recv_waiters))))
        while waiter != nullptr:
            if sched != nullptr:
                sched_requeue_task(sched, waiter)
            else:
                atomic_store_i32(
                    ptr[i32](ptr[void](ptr(waiter.state))), TASK_PENDING
                )
            waiter = taskq_pop(ptr[TaskQueue](ptr[void](ptr(ch.recv_waiters))))

        spinlock_unlock(ptr[SpinLock](ptr[void](ptr(ch.lock))))

    # ---- send (blocking from worker context) ----
    @compile(suffix=type_suffix)
    def channel_send(w: ptr[Worker], ch: ptr[_Channel], value: element_type) -> i32:
        """Send, blocking the current task while the channel is full."""
        sched: ptr[Scheduler] = ptr[Scheduler](w.scheduler)

        while True:
            w = sched_current_worker(sched)
            if w == nullptr:
                return i32(0)

            spinlock_lock(ptr[SpinLock](ptr[void](ptr(ch.lock))))
            ch.scheduler = ptr[void](sched)

            if ch.closed != i32(0):
                spinlock_unlock(ptr[SpinLock](ptr[void](ptr(ch.lock))))
                return i32(0)

            if ch.count < ch.cap:
                idx: u64 = ch.tail % ch.cap
                ch.buffer[idx] = value
                ch.tail = ch.tail + u64(1)
                ch.count = ch.count + u64(1)

                waiter: ptr[Task] = taskq_pop(
                    ptr[TaskQueue](ptr[void](ptr(ch.recv_waiters)))
                )
                spinlock_unlock(ptr[SpinLock](ptr[void](ptr(ch.lock))))

                if waiter != nullptr:
                    sched_requeue_task(sched, waiter)
                return i32(1)

            current: ptr[Task] = w.current_task
            if current == nullptr:
                spinlock_unlock(ptr[SpinLock](ptr[void](ptr(ch.lock))))
                return i32(0)

            spinlock_lock(ptr[SpinLock](ptr[void](ptr(current.lock))))
            atomic_store_i32(
                ptr[i32](ptr[void](ptr(current.state))), TASK_BLOCKING
            )
            spinlock_unlock(ptr[SpinLock](ptr[void](ptr(current.lock))))
            taskq_push(ptr[TaskQueue](ptr[void](ptr(ch.send_waiters))), current)
            spinlock_unlock(ptr[SpinLock](ptr[void](ptr(ch.lock))))
            sched_suspend_current(w)

    # ---- recv (blocking from worker context) ----
    @compile(suffix=type_suffix)
    def channel_recv(w: ptr[Worker], ch: ptr[_Channel], out: ptr[element_type]) -> i32:
        """Receive, blocking the current task while the channel is empty."""
        sched: ptr[Scheduler] = ptr[Scheduler](w.scheduler)

        while True:
            w = sched_current_worker(sched)
            if w == nullptr:
                return i32(0)

            spinlock_lock(ptr[SpinLock](ptr[void](ptr(ch.lock))))
            ch.scheduler = ptr[void](sched)

            if ch.count != u64(0):
                idx: u64 = ch.head % ch.cap
                out[0] = ch.buffer[idx]
                ch.head = ch.head + u64(1)
                ch.count = ch.count - u64(1)

                waiter: ptr[Task] = taskq_pop(
                    ptr[TaskQueue](ptr[void](ptr(ch.send_waiters)))
                )
                spinlock_unlock(ptr[SpinLock](ptr[void](ptr(ch.lock))))

                if waiter != nullptr:
                    sched_requeue_task(sched, waiter)
                return i32(1)

            if ch.closed != i32(0):
                spinlock_unlock(ptr[SpinLock](ptr[void](ptr(ch.lock))))
                return i32(0)

            current: ptr[Task] = w.current_task
            if current == nullptr:
                spinlock_unlock(ptr[SpinLock](ptr[void](ptr(ch.lock))))
                return i32(0)

            spinlock_lock(ptr[SpinLock](ptr[void](ptr(current.lock))))
            atomic_store_i32(
                ptr[i32](ptr[void](ptr(current.state))), TASK_BLOCKING
            )
            spinlock_unlock(ptr[SpinLock](ptr[void](ptr(current.lock))))
            taskq_push(ptr[TaskQueue](ptr[void](ptr(ch.recv_waiters))), current)
            spinlock_unlock(ptr[SpinLock](ptr[void](ptr(ch.lock))))
            sched_suspend_current(w)

    # ---- Bundle API ----
    return SimpleNamespace(
        type=_Channel,
        create=channel_create,
        destroy=channel_destroy,
        send=channel_send,
        recv=channel_recv,
        try_send=channel_try_send,
        try_recv=channel_try_recv,
        close=channel_close,
    )
