"""Skynet benchmark for the internal typed task adapter.

This is not a user-facing runtime example.  Public PythoC task code should use
Future.  This file exercises the lower adapter used by Future while keeping
recursive raw task control for the benchmark shape.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'PythoC'))

from pythoc import compile, i32, i64, u64, ptr, void, nullptr, sizeof
from pythoc.libc.stdlib import malloc, free
from pythoc.std.runtime.raw import (
    Runtime,
    Task,
    runtime_new_raw as runtime_new,
    runtime_start_raw as runtime_start,
    runtime_shutdown_raw as runtime_shutdown,
    runtime_free_raw as runtime_free,
)
from pythoc.std.runtime.spawn_typed import _TypedTask


# The actual work function used by the adapter benchmark.
@compile
def skynet(rt: ptr[Runtime], num: i64, size: i64, div: i64) -> i64:
    """Skynet node: if leaf, return num. Otherwise spawn children and sum."""
    if size == i64(1):
        return num

    sub_size: i64 = size / div
    total: i64 = 0

    # Allocate task pointer array
    tasks: ptr[ptr[Task]] = ptr[ptr[Task]](malloc(div * i64(sizeof(ptr[Task]))))

    # Spawn all children
    i: i64 = 0
    while i < div:
        tasks[i] = Skynet.spawn_raw(
            rt,
            rt, num + i * sub_size, sub_size, div
        )
        i = i + 1

    # Join all children and sum results
    i = i64(0)
    while i < div:
        total = total + Skynet.join_raw(rt, tasks[i])
        i = i + 1

    free(ptr[void](tasks))
    return total


# Generates Args struct + trampoline + typed spawn/join for this benchmark.
Skynet = _TypedTask(skynet, stack_size=u64(8192))


@compile(suffix="skynet_typed_main")
def skynet_main(num_workers: i32, total_nodes: i64, fan_out: i64) -> i64:
    """Top-level: create runtime, spawn root skynet, join, return sum."""
    rt: ptr[Runtime] = runtime_new(num_workers)
    runtime_start(rt)

    root: ptr[Task] = Skynet.spawn_raw(rt, rt, i64(0), total_nodes, fan_out)
    result: i64 = Skynet.join_raw(rt, root)

    runtime_shutdown(rt)
    runtime_free(rt)
    return result


def run(total: int = 1_000_000, fan_out: int = 10, num_workers: int = 8) -> int:
    """Run skynet benchmark via the internal adapter."""
    return int(skynet_main(i32(num_workers), i64(total), i64(fan_out)))


if __name__ == "__main__":
    import time

    # Correctness check
    r = run(10000, 10, 4)
    expected = sum(range(10000))
    print(f"skynet(10K): {r} (expected {expected}) {'OK' if r == expected else 'FAIL'}")

    # Performance
    t0 = time.perf_counter()
    r = run(100_000, 10, 8)
    t1 = time.perf_counter()
    print(f"skynet(100K): {r} in {t1-t0:.3f}s")
