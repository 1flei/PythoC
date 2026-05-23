#!/usr/bin/env python3
"""Integration tests for native thread_local storage."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc import compile, i32, ptr, void, nullptr, array, thread_local
from pythoc.build.output_manager import flush_all_pending_outputs
from pythoc.std.runtime.platform import ThreadHandle, thread_create, thread_join
from test.utils.test_utils import DeferredTestCase


@compile
def tls_next() -> i32:
    counter: thread_local[i32] = 0
    counter = counter + i32(1)
    return counter


@compile
def tls_worker(arg: ptr[void]) -> ptr[void]:
    out: ptr[i32] = ptr[i32](arg)
    out[0] = tls_next()
    out[1] = tls_next()
    return nullptr


@compile
def test_fn_thread_local_isolation() -> i32:
    values: array[i32, 4] = [0, 0, 0, 0]

    first: ThreadHandle = thread_create(
        ptr[void](tls_worker),
        ptr[void](ptr(values[0])),
    )
    thread_join(first)

    second: ThreadHandle = thread_create(
        ptr[void](tls_worker),
        ptr[void](ptr(values[2])),
    )
    thread_join(second)

    if values[0] != i32(1):
        return i32(1)
    if values[1] != i32(2):
        return i32(2)
    if values[2] != i32(1):
        return i32(3)
    if values[3] != i32(2):
        return i32(4)
    return i32(0)


flush_all_pending_outputs()


class TestThreadLocal(DeferredTestCase):
    def test_thread_local_isolation(self):
        self.assertEqual(test_fn_thread_local_isolation(), 0)


if __name__ == '__main__':
    unittest.main()
