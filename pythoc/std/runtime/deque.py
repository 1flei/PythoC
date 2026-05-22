"""
Work-Stealing Deque: Chase-Lev lock-free deque (cross-platform).

Each worker owns a local deque.  The owner pushes/pops from the bottom
(LIFO for cache locality).  Other workers steal from the top (FIFO for
load balancing).

This is the standard Chase-Lev algorithm:
- push_bottom / pop_bottom: owner-only, lock-free
- steal: thieves, lock-free with CAS

Design:
    Fixed-capacity circular buffer.  No dynamic resizing (PythoC: explicit).
    If the deque is full, overflow goes to the global queue.

    Uses LLVM-native atomic operations (portable across all targets).

Reference:
    Chase, Lev. "Dynamic Circular Work-Stealing Deque" (2005)
"""
from __future__ import annotations

from .policy import bind_mem
bind_mem()

from pythoc import compile, i32, i64, u64, ptr, void, struct, nullptr, sizeof, array
from pythoc.libc.string import memset

from .platform import atomic_load_i64, atomic_store_i64, atomic_cas_i64
from .task import Task


# ============================================================
# Steal results
# ============================================================

STEAL_SUCCESS = i32(0)
STEAL_EMPTY   = i32(1)
STEAL_ABORT   = i32(2)


# ============================================================
# Work-stealing deque (fixed capacity)
# ============================================================

DEQUE_CAPACITY = 4096  # max tasks per worker deque

@compile
class WSDeque:
    top: i64                            # steal index (atomic)
    bottom: i64                         # owner index (atomic)
    buffer: array[ptr[Task], 4096]      # circular buffer


@compile
def wsdeque_init(dq: ptr[WSDeque]) -> void:
    """Initialize a work-stealing deque."""
    dq.top = i64(0)
    dq.bottom = i64(0)
    memset(ptr[void](ptr(dq.buffer)), 0, i64(sizeof(array[ptr[Task], 4096])))


@compile
def wsdeque_push(dq: ptr[WSDeque], task: ptr[Task]) -> i32:
    """Push task to bottom (owner only).
    Returns 1 on success, 0 if deque is full.
    """
    b: i64 = atomic_load_i64(ptr[i64](ptr[void](ptr(dq.bottom))))
    t: i64 = atomic_load_i64(ptr[i64](ptr[void](ptr(dq.top))))

    # Check capacity
    size: i64 = b - t
    if size >= i64(DEQUE_CAPACITY):
        return i32(0)  # full

    # Write task to buffer[b % capacity]
    idx: i64 = b % i64(DEQUE_CAPACITY)
    dq.buffer[idx] = task

    # Publish: increment bottom (acts as release fence)
    atomic_store_i64(ptr[i64](ptr[void](ptr(dq.bottom))), b + i64(1))
    return i32(1)


@compile
def wsdeque_pop(dq: ptr[WSDeque]) -> ptr[Task]:
    """Pop task from bottom (owner only).  Returns nullptr if empty."""
    b: i64 = atomic_load_i64(ptr[i64](ptr[void](ptr(dq.bottom))))
    b = b - i64(1)
    atomic_store_i64(ptr[i64](ptr[void](ptr(dq.bottom))), b)

    t: i64 = atomic_load_i64(ptr[i64](ptr[void](ptr(dq.top))))

    if t <= b:
        # Non-empty: read task
        idx: i64 = b % i64(DEQUE_CAPACITY)
        task: ptr[Task] = dq.buffer[idx]

        if t == b:
            # Last element — race with steal
            expected: i64 = t
            if atomic_cas_i64(
                ptr[i64](ptr[void](ptr(dq.top))),
                ptr[i64](ptr[void](ptr(expected))),
                t + i64(1)
            ) == 0:
                # Lost race to a stealer
                atomic_store_i64(ptr[i64](ptr[void](ptr(dq.bottom))), t + i64(1))
                return nullptr

            atomic_store_i64(ptr[i64](ptr[void](ptr(dq.bottom))), t + i64(1))
        return task
    else:
        # Empty: restore bottom
        atomic_store_i64(ptr[i64](ptr[void](ptr(dq.bottom))), t)
        return nullptr


@compile
def wsdeque_steal(dq: ptr[WSDeque]) -> ptr[Task]:
    """Steal task from top (any thread).  Returns nullptr if empty/aborted."""
    t: i64 = atomic_load_i64(ptr[i64](ptr[void](ptr(dq.top))))
    b: i64 = atomic_load_i64(ptr[i64](ptr[void](ptr(dq.bottom))))

    if t >= b:
        return nullptr  # empty

    # Read task at top
    idx: i64 = t % i64(DEQUE_CAPACITY)
    task: ptr[Task] = dq.buffer[idx]

    # Try to advance top (claim this slot)
    expected: i64 = t
    if atomic_cas_i64(
        ptr[i64](ptr[void](ptr(dq.top))),
        ptr[i64](ptr[void](ptr(expected))),
        t + i64(1)
    ) == 0:
        return nullptr  # another stealer won

    return task


@compile
def wsdeque_size(dq: ptr[WSDeque]) -> i64:
    """Approximate size (snapshot, may be stale)."""
    b: i64 = atomic_load_i64(ptr[i64](ptr[void](ptr(dq.bottom))))
    t: i64 = atomic_load_i64(ptr[i64](ptr[void](ptr(dq.top))))
    size: i64 = b - t
    if size < i64(0):
        return i64(0)
    return size
