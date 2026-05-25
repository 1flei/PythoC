#!/usr/bin/env python3
"""
Test runtime channel: bounded MPMC channel operations.

Tests cover:
- Basic send/recv
- FIFO ordering
- Full buffer (try_send returns 0)
- Empty buffer (try_recv returns 0)
- Ring buffer wrap-around
- Close semantics
- Different element types (i32, ptr[void])
- Edge case: capacity=1
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import unittest
from pythoc.decorators.compile import compile
from pythoc.builtin_entities import (
    void, i32, i64, u64, ptr, nullptr, sizeof,
)
from pythoc.libc.stdlib import malloc, free
from pythoc.libc.string import memset
from pythoc.build.output_manager import flush_all_pending_outputs

from test.utils.test_utils import DeferredTestCase

from pythoc.std.runtime.channel import Channel
from pythoc.std.runtime.api import (
    Runtime, runtime_spawn, runtime_join, runtime_current_worker,
)
from pythoc.std.runtime.raw import (
    runtime_new_raw as runtime_new,
    runtime_start_raw as runtime_start,
    runtime_shutdown_raw as runtime_shutdown,
    runtime_free_raw as runtime_free,
)
from pythoc.std.runtime.scheduler import Worker
from pythoc.std.runtime.task import TaskHandle


# ============================================================
# Create typed channels for testing
# ============================================================

Ch_i32 = Channel(i32, capacity=4)
Ch_i32_cap1 = Channel(i32, capacity=1)


@compile
class BlockingRecvArgs:
    rt: ptr[Runtime]
    ch: ptr[Ch_i32_cap1.type]
    out: ptr[i32]


@compile
class BlockingSendArgs:
    rt: ptr[Runtime]
    ch: ptr[Ch_i32_cap1.type]
    value: i32


@compile
class BlockingPipeArgs:
    rt: ptr[Runtime]
    ch: ptr[Ch_i32_cap1.type]
    sum_out: ptr[i32]


# ============================================================
# Test functions: basic send/recv
# ============================================================

@compile(suffix="ch_send_recv_basic")
def test_fn_ch_send_recv() -> i32:
    """Send one value, receive it back."""
    ch: ptr[Ch_i32.type] = Ch_i32.create()
    Ch_i32.try_send(ch, i32(42))

    out: i32 = 0
    Ch_i32.try_recv(ch, ptr[i32](ptr[void](ptr(out))))

    Ch_i32.destroy(ch)
    return out


@compile(suffix="ch_fifo_order")
def test_fn_ch_fifo_order() -> i32:
    """Send 1,2,3 — receive 1,2,3 (FIFO)."""
    ch: ptr[Ch_i32.type] = Ch_i32.create()

    Ch_i32.try_send(ch, i32(1))
    Ch_i32.try_send(ch, i32(2))
    Ch_i32.try_send(ch, i32(3))

    a: i32 = 0
    b: i32 = 0
    c: i32 = 0
    Ch_i32.try_recv(ch, ptr[i32](ptr[void](ptr(a))))
    Ch_i32.try_recv(ch, ptr[i32](ptr[void](ptr(b))))
    Ch_i32.try_recv(ch, ptr[i32](ptr[void](ptr(c))))

    Ch_i32.destroy(ch)
    # Encode order: a*100 + b*10 + c = 123
    return a * 100 + b * 10 + c


@compile(suffix="ch_full_returns_zero")
def test_fn_ch_full() -> i32:
    """try_send on full channel returns 0."""
    ch: ptr[Ch_i32.type] = Ch_i32.create()  # capacity=4

    # Fill to capacity
    Ch_i32.try_send(ch, i32(1))
    Ch_i32.try_send(ch, i32(2))
    Ch_i32.try_send(ch, i32(3))
    Ch_i32.try_send(ch, i32(4))

    # 5th send should fail
    result: i32 = Ch_i32.try_send(ch, i32(5))

    Ch_i32.destroy(ch)
    return result  # 0 = failure


@compile(suffix="ch_empty_returns_zero")
def test_fn_ch_empty() -> i32:
    """try_recv on empty channel returns 0."""
    ch: ptr[Ch_i32.type] = Ch_i32.create()

    out: i32 = 99
    result: i32 = Ch_i32.try_recv(ch, ptr[i32](ptr[void](ptr(out))))

    Ch_i32.destroy(ch)
    # result=0 (empty), out unchanged
    return result * 100 + out  # 0*100 + 99 = 99


@compile(suffix="ch_wraparound")
def test_fn_ch_wraparound() -> i32:
    """Ring buffer wrap-around: fill, drain, fill again."""
    ch: ptr[Ch_i32.type] = Ch_i32.create()  # capacity=4

    # Fill
    Ch_i32.try_send(ch, i32(10))
    Ch_i32.try_send(ch, i32(20))
    Ch_i32.try_send(ch, i32(30))
    Ch_i32.try_send(ch, i32(40))

    # Drain 2
    out: i32 = 0
    Ch_i32.try_recv(ch, ptr[i32](ptr[void](ptr(out))))  # 10
    Ch_i32.try_recv(ch, ptr[i32](ptr[void](ptr(out))))  # 20

    # Fill 2 more (these wrap around in the ring buffer)
    Ch_i32.try_send(ch, i32(50))
    Ch_i32.try_send(ch, i32(60))

    # Drain all 4
    a: i32 = 0
    b: i32 = 0
    c: i32 = 0
    d: i32 = 0
    Ch_i32.try_recv(ch, ptr[i32](ptr[void](ptr(a))))  # 30
    Ch_i32.try_recv(ch, ptr[i32](ptr[void](ptr(b))))  # 40
    Ch_i32.try_recv(ch, ptr[i32](ptr[void](ptr(c))))  # 50
    Ch_i32.try_recv(ch, ptr[i32](ptr[void](ptr(d))))  # 60

    Ch_i32.destroy(ch)
    # Verify order
    if a == 30 and b == 40 and c == 50 and d == 60:
        return i32(1)
    return i32(0)


@compile(suffix="ch_capacity_one")
def test_fn_ch_capacity_one() -> i32:
    """Edge case: channel with capacity=1."""
    ch: ptr[Ch_i32_cap1.type] = Ch_i32_cap1.create()

    # Send 1 item
    r1: i32 = Ch_i32_cap1.try_send(ch, i32(77))
    # Should succeed
    if r1 != 1:
        Ch_i32_cap1.destroy(ch)
        return i32(-1)

    # Second send should fail (full)
    r2: i32 = Ch_i32_cap1.try_send(ch, i32(88))
    if r2 != 0:
        Ch_i32_cap1.destroy(ch)
        return i32(-2)

    # Receive the item
    out: i32 = 0
    r3: i32 = Ch_i32_cap1.try_recv(ch, ptr[i32](ptr[void](ptr(out))))
    if r3 != 1 or out != 77:
        Ch_i32_cap1.destroy(ch)
        return i32(-3)

    # Now empty — recv should fail
    r4: i32 = Ch_i32_cap1.try_recv(ch, ptr[i32](ptr[void](ptr(out))))
    if r4 != 0:
        Ch_i32_cap1.destroy(ch)
        return i32(-4)

    Ch_i32_cap1.destroy(ch)
    return i32(1)


@compile(suffix="ch_alternating_send_recv")
def test_fn_ch_alternating() -> i32:
    """Alternating send/recv pattern (common producer-consumer)."""
    ch: ptr[Ch_i32.type] = Ch_i32.create()
    total: i32 = 0
    out: i32 = 0

    i: i32 = 0
    while i < 10:
        Ch_i32.try_send(ch, i)
        Ch_i32.try_recv(ch, ptr[i32](ptr[void](ptr(out))))
        total = total + out
        i = i + 1

    Ch_i32.destroy(ch)
    return total  # 0+1+2+...+9 = 45


@compile(suffix="ch_close_basic")
def test_fn_ch_close() -> i32:
    """Close marks channel closed."""
    ch: ptr[Ch_i32.type] = Ch_i32.create()
    Ch_i32.try_send(ch, i32(1))
    Ch_i32.close(ch)

    # Can still recv buffered items after close
    out: i32 = 0
    result: i32 = Ch_i32.try_recv(ch, ptr[i32](ptr[void](ptr(out))))

    Ch_i32.destroy(ch)
    if result == 1 and out == 1:
        return i32(1)
    return i32(0)


@compile(suffix="ch_blocking_recv_entry")
def test_fn_ch_blocking_recv_entry(arg: ptr[void]) -> ptr[void]:
    args: ptr[BlockingRecvArgs] = ptr[BlockingRecvArgs](arg)
    worker: ptr[Worker] = runtime_current_worker(args.rt)
    Ch_i32_cap1.recv(worker, args.ch, args.out)
    return nullptr


@compile(suffix="ch_blocking_send_entry")
def test_fn_ch_blocking_send_entry(arg: ptr[void]) -> ptr[void]:
    args: ptr[BlockingSendArgs] = ptr[BlockingSendArgs](arg)
    worker: ptr[Worker] = runtime_current_worker(args.rt)
    Ch_i32_cap1.send(worker, args.ch, args.value)
    return nullptr


@compile(suffix="ch_blocking_recv_then_send")
def test_fn_ch_blocking_recv_then_send() -> i32:
    rt: ptr[Runtime] = runtime_new(i32(1))
    ch: ptr[Ch_i32_cap1.type] = Ch_i32_cap1.create()
    out: i32 = 0

    recv_args: BlockingRecvArgs
    recv_args.rt = rt
    recv_args.ch = ch
    recv_args.out = ptr[i32](ptr[void](ptr(out)))

    send_args: BlockingSendArgs
    send_args.rt = rt
    send_args.ch = ch
    send_args.value = i32(314)

    runtime_start(rt)
    recv_task: TaskHandle = runtime_spawn(
        rt, test_fn_ch_blocking_recv_entry, ptr[void](ptr(recv_args)), u64(0)
    )
    send_task: TaskHandle = runtime_spawn(
        rt, test_fn_ch_blocking_send_entry, ptr[void](ptr(send_args)), u64(0)
    )

    runtime_join(rt, recv_task)
    runtime_join(rt, send_task)
    runtime_shutdown(rt)
    runtime_free(rt)
    Ch_i32_cap1.destroy(ch)
    return out


@compile(suffix="ch_blocking_send_then_recv")
def test_fn_ch_blocking_send_then_recv() -> i32:
    rt: ptr[Runtime] = runtime_new(i32(1))
    ch: ptr[Ch_i32_cap1.type] = Ch_i32_cap1.create()
    Ch_i32_cap1.try_send(ch, i32(11))

    recv_out: i32 = 0
    recv_args: BlockingRecvArgs
    recv_args.rt = rt
    recv_args.ch = ch
    recv_args.out = ptr[i32](ptr[void](ptr(recv_out)))

    send_args: BlockingSendArgs
    send_args.rt = rt
    send_args.ch = ch
    send_args.value = i32(22)

    runtime_start(rt)
    send_task: TaskHandle = runtime_spawn(
        rt, test_fn_ch_blocking_send_entry, ptr[void](ptr(send_args)), u64(0)
    )
    recv_task: TaskHandle = runtime_spawn(
        rt, test_fn_ch_blocking_recv_entry, ptr[void](ptr(recv_args)), u64(0)
    )

    runtime_join(rt, send_task)
    runtime_join(rt, recv_task)

    final_out: i32 = 0
    Ch_i32_cap1.try_recv(ch, ptr[i32](ptr[void](ptr(final_out))))

    runtime_shutdown(rt)
    runtime_free(rt)
    Ch_i32_cap1.destroy(ch)
    return recv_out * i32(100) + final_out


@compile(suffix="ch_blocking_pipe_producer")
def test_fn_ch_blocking_pipe_producer(arg: ptr[void]) -> ptr[void]:
    args: ptr[BlockingPipeArgs] = ptr[BlockingPipeArgs](arg)
    i: i32 = 0
    while i < 10:
        worker: ptr[Worker] = runtime_current_worker(args.rt)
        Ch_i32_cap1.send(worker, args.ch, i)
        i = i + 1
    return nullptr


@compile(suffix="ch_blocking_pipe_consumer")
def test_fn_ch_blocking_pipe_consumer(arg: ptr[void]) -> ptr[void]:
    args: ptr[BlockingPipeArgs] = ptr[BlockingPipeArgs](arg)
    total: i32 = 0
    value: i32 = 0
    i: i32 = 0
    while i < 10:
        worker: ptr[Worker] = runtime_current_worker(args.rt)
        Ch_i32_cap1.recv(worker, args.ch, ptr[i32](ptr[void](ptr(value))))
        total = total + value
        i = i + 1
    args.sum_out[0] = total
    return nullptr


@compile(suffix="ch_blocking_pipe_two_workers")
def test_fn_ch_blocking_pipe_two_workers() -> i32:
    rt: ptr[Runtime] = runtime_new(i32(2))
    ch: ptr[Ch_i32_cap1.type] = Ch_i32_cap1.create()
    total: i32 = 0

    args: BlockingPipeArgs
    args.rt = rt
    args.ch = ch
    args.sum_out = ptr[i32](ptr[void](ptr(total)))

    runtime_start(rt)
    consumer: TaskHandle = runtime_spawn(
        rt, test_fn_ch_blocking_pipe_consumer, ptr[void](ptr(args)), u64(0)
    )
    producer: TaskHandle = runtime_spawn(
        rt, test_fn_ch_blocking_pipe_producer, ptr[void](ptr(args)), u64(0)
    )

    runtime_join(rt, producer)
    runtime_join(rt, consumer)
    runtime_shutdown(rt)
    runtime_free(rt)
    Ch_i32_cap1.destroy(ch)
    return total


# ============================================================
# Test class
# ============================================================

class TestRuntimeChannel(DeferredTestCase):
    """Tests for bounded MPMC channels."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        flush_all_pending_outputs()

    def test_send_recv_basic(self):
        self.assertEqual(test_fn_ch_send_recv(), 42)

    def test_fifo_order(self):
        self.assertEqual(test_fn_ch_fifo_order(), 123)

    def test_full_returns_zero(self):
        self.assertEqual(test_fn_ch_full(), 0)

    def test_empty_returns_zero(self):
        self.assertEqual(test_fn_ch_empty(), 99)

    def test_wraparound(self):
        self.assertEqual(test_fn_ch_wraparound(), 1)

    def test_capacity_one(self):
        self.assertEqual(test_fn_ch_capacity_one(), 1)

    def test_alternating_send_recv(self):
        self.assertEqual(test_fn_ch_alternating(), 45)

    def test_close_basic(self):
        self.assertEqual(test_fn_ch_close(), 1)

    def test_blocking_recv_then_send(self):
        self.assertEqual(test_fn_ch_blocking_recv_then_send(), 314)

    def test_blocking_send_then_recv(self):
        self.assertEqual(test_fn_ch_blocking_send_then_recv(), 1122)

    def test_blocking_pipe_two_workers(self):
        self.assertEqual(test_fn_ch_blocking_pipe_two_workers(), 45)

    def test_blocking_pipe_two_workers_repeated(self):
        for _ in range(16):
            self.assertEqual(test_fn_ch_blocking_pipe_two_workers(), 45)


if __name__ == '__main__':
    unittest.main()
