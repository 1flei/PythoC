# -*- coding: utf-8 -*-
"""Static counters for testing - separate module to avoid effect propagation"""

from pythoc import compile, u64, void, i64, static


@compile
def malloc_counter(op: i64) -> u64:
    """Static malloc counter"""
    count: static[u64] = u64(0)
    if op == i64(1):
        count = count + u64(1)
    elif op == i64(-1):
        count = u64(0)
    return count


@compile
def free_counter(op: i64) -> u64:
    """Static free counter"""
    count: static[u64] = u64(0)
    if op == i64(1):
        count = count + u64(1)
    elif op == i64(-1):
        count = u64(0)
    return count


@compile
def init_counters() -> void:
    """Reset counters"""
    malloc_counter(i64(-1))
    free_counter(i64(-1))


@compile
def get_malloc_count() -> u64:
    return malloc_counter(i64(0))


@compile
def get_free_count() -> u64:
    return free_counter(i64(0))
