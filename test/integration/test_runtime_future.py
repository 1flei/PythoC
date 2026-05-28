#!/usr/bin/env python3
"""Integration tests for typed linear Future runtime harness."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import unittest
from pythoc import compile, effect, i32, i64, ptr, void, u8, nullptr
from pythoc.build.output_manager import flush_all_pending_outputs
from test.utils.test_utils import DeferredTestCase, expect_error

import pythoc.std.runtime as runtime_facade
from pythoc.std.runtime import (
    runtime_start, runtime_shutdown, Future, ThreadPoolExecutor,
)
from pythoc.std.runtime.thread_pool_executor import _ThreadJob, thread_pool_shutdown


@compile
def add_pair(x: i64, y: i64) -> i64:
    return x + y


@compile(suffix="future_join_basic")
def test_fn_future_join_basic() -> i64:
    rt = runtime_start(i32(2))

    f = Future.spawn(add_pair, i64(1), i64(2))
    result: i64 = Future.join(f)

    runtime_shutdown(rt)
    return result


@compile(suffix="future_view_basic")
def test_fn_future_view_basic() -> i64:
    rt = runtime_start(i32(2))

    f = Future.spawn(add_pair, i64(10), i64(20))
    result: i64 = 0
    for value in Future.view(f):
        result = value

    runtime_shutdown(rt)
    return result


@compile(suffix="future_do_composition_future_v4")
def test_fn_future_do_composition() -> i64:
    rt = runtime_start(i32(4))

    fx = Future.spawn(add_pair, i64(1), i64(2))
    fy = Future.spawn(add_pair, i64(10), i64(20))
    fz = Future.do(
        x + y
        for x in Future.view(fx)
        for y in Future.view(fy)
    )
    result: i64 = Future.join(fz)

    runtime_shutdown(rt)
    return result


@compile
class Counter:
    value: i64


@compile
def bump(counter: ptr[Counter]) -> void:
    counter.value = counter.value + i64(1)


@compile(suffix="future_void_join")
def test_fn_future_void_join() -> i64:
    rt = runtime_start(i32(2))

    counter: Counter
    counter.value = i64(0)

    f1 = Future.spawn(bump, ptr[Counter](ptr[void](ptr(counter))))
    Future.join(f1)
    f2 = Future.spawn(bump, ptr[Counter](ptr[void](ptr(counter))))
    Future.join(f2)

    runtime_shutdown(rt)
    return counter.value


@compile(suffix="future_detach")
def test_fn_future_detach() -> i64:
    rt = runtime_start(i32(1))

    counter: Counter
    counter.value = i64(0)

    f = Future.spawn(bump, ptr[Counter](ptr[void](ptr(counter))))
    Future.detach(f)

    runtime_shutdown(rt)
    return counter.value


with effect(executor=ThreadPoolExecutor, suffix="thread_pool"):
    @compile(suffix="future_thread_pool_join")
    def test_fn_future_thread_pool_join() -> i64:
        f1 = Future.spawn(add_pair, i64(3), i64(4))
        f2 = Future.spawn(add_pair, i64(30), i64(40))
        r1: i64 = Future.join(f1)
        r2: i64 = Future.join(f2)
        return r1 + r2


with effect(executor=ThreadPoolExecutor, suffix="thread_pool_do"):
    @compile(suffix="future_thread_pool_do_future_v4")
    def test_fn_future_thread_pool_do() -> i64:
        fx = Future.spawn(add_pair, i64(5), i64(6))
        fy = Future.spawn(add_pair, i64(50), i64(60))
        fz = Future.do(
            x + y
            for x in Future.view(fx)
            for y in Future.view(fy)
        )
        return Future.join(fz)


@compile(suffix="thread_pool_job_alignment")
def test_fn_thread_pool_job_alignment() -> i64:
    job: _ThreadJob
    base: ptr[u8] = ptr[u8](ptr[void](ptr(job)))
    lock_offset: i64 = ptr[u8](ptr[void](ptr(job.lock))) - base
    cv_offset: i64 = ptr[u8](ptr[void](ptr(job.done_cv))) - base
    if lock_offset % i64(8) != i64(0):
        return i64(0)
    if cv_offset % i64(8) != i64(0):
        return i64(0)
    return i64(1)


@expect_error(["not consumed"], suffix="bad_future_not_consumed")
def run_error_future_not_consumed():
    @compile(suffix="bad_future_not_consumed")
    def bad_future_not_consumed() -> void:
        rt = runtime_start(i32(1))
        f = Future.spawn(add_pair, i64(1), i64(2))
        runtime_shutdown(rt)


@expect_error(["consumed"], suffix="bad_future_double_join")
def run_error_future_double_join():
    @compile(suffix="bad_future_double_join")
    def bad_future_double_join() -> void:
        rt = runtime_start(i32(1))
        f = Future.spawn(add_pair, i64(1), i64(2))
        Future.join(f)
        Future.join(f)
        runtime_shutdown(rt)


@expect_error(["inconsistent", "not consumed"], suffix="bad_future_branch")
def run_error_future_branch():
    @compile(suffix="bad_future_branch")
    def bad_future_branch(flag: i32) -> void:
        rt = runtime_start(i32(1))
        f = Future.spawn(add_pair, i64(1), i64(2))
        if flag != 0:
            Future.detach(f)
        runtime_shutdown(rt)


@expect_error(["not consumed"], suffix="bad_future_do_not_consumed")
def run_error_future_do_not_consumed():
    @compile(suffix="bad_future_do_not_consumed")
    def bad_future_do_not_consumed() -> void:
        rt = runtime_start(i32(2))
        fx = Future.spawn(add_pair, i64(1), i64(2))
        fy = Future.spawn(add_pair, i64(10), i64(20))
        fz = Future.do(
            x + y
            for x in Future.view(fx)
            for y in Future.view(fy)
        )
        runtime_shutdown(rt)


@expect_error(["not consumed"], suffix="bad_runtime_not_shutdown")
def run_error_runtime_not_shutdown():
    @compile(suffix="bad_runtime_not_shutdown")
    def bad_runtime_not_shutdown() -> void:
        rt = runtime_start(i32(1))


@expect_error(["consumed"], suffix="bad_runtime_double_shutdown")
def run_error_runtime_double_shutdown():
    @compile(suffix="bad_runtime_double_shutdown")
    def bad_runtime_double_shutdown() -> void:
        rt = runtime_start(i32(1))
        runtime_shutdown(rt)
        runtime_shutdown(rt)


@expect_error(["inconsistent", "not consumed"], suffix="bad_runtime_branch")
def run_error_runtime_branch():
    @compile(suffix="bad_runtime_branch")
    def bad_runtime_branch(flag: i32) -> void:
        rt = runtime_start(i32(1))
        if flag != 0:
            runtime_shutdown(rt)


class TestRuntimeFuture(DeferredTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        flush_all_pending_outputs()

    def test_runtime_facade_hides_raw_task_api(self):
        self.assertFalse(hasattr(runtime_facade, "runtime_spawn_raw"))
        self.assertFalse(hasattr(runtime_facade, "runtime_spawn"))
        self.assertFalse(hasattr(runtime_facade, "TaskHandle"))
        self.assertFalse(hasattr(runtime_facade, "TypedTask"))
        self.assertFalse(hasattr(runtime_facade, "runtime_new"))
        self.assertFalse(hasattr(runtime_facade, "runtime_free"))
        self.assertFalse(hasattr(runtime_facade, "FutureTask"))

    def test_00_future_not_consumed_error(self):
        passed, msg = run_error_future_not_consumed()
        self.assertTrue(passed, msg)

    def test_01_future_double_join_error(self):
        passed, msg = run_error_future_double_join()
        self.assertTrue(passed, msg)

    def test_02_future_branch_error(self):
        passed, msg = run_error_future_branch()
        self.assertTrue(passed, msg)

    def test_03_future_do_not_consumed_error(self):
        passed, msg = run_error_future_do_not_consumed()
        self.assertTrue(passed, msg)

    def test_04_runtime_not_shutdown_error(self):
        passed, msg = run_error_runtime_not_shutdown()
        self.assertTrue(passed, msg)

    def test_05_runtime_double_shutdown_error(self):
        passed, msg = run_error_runtime_double_shutdown()
        self.assertTrue(passed, msg)

    def test_06_runtime_branch_error(self):
        passed, msg = run_error_runtime_branch()
        self.assertTrue(passed, msg)

    def test_10_future_do_composition(self):
        self.assertEqual(test_fn_future_do_composition(), 33)

    def test_11_future_thread_pool_do(self):
        try:
            self.assertEqual(test_fn_future_thread_pool_do(), 121)
        finally:
            thread_pool_shutdown()

    def test_12_future_join_basic(self):
        self.assertEqual(test_fn_future_join_basic(), 3)

    def test_13_future_view_basic(self):
        self.assertEqual(test_fn_future_view_basic(), 30)

    def test_14_future_void_join(self):
        self.assertEqual(test_fn_future_void_join(), 2)

    def test_15_future_detach(self):
        self.assertEqual(test_fn_future_detach(), 1)

    def test_16_future_thread_pool_join(self):
        try:
            self.assertEqual(test_fn_future_thread_pool_join(), 77)
        finally:
            thread_pool_shutdown()

    def test_thread_pool_job_alignment(self):
        self.assertEqual(test_fn_thread_pool_job_alignment(), 1)


if __name__ == '__main__':
    unittest.main()
