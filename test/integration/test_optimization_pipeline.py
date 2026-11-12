"""Optimized native loops preserve pointer aliases and observable effects."""

import unittest
from pythoc import compile, array, i32, i64, u64, ptr, static


@compile
def dispatch(first: ptr[u64], second: ptr[u64], count: i64) -> u64:
    current: ptr[u64] = first
    other: ptr[u64] = second
    i: i64 = 0
    while i < count:
        mode: i64 = i % i64(3)
        if mode == i64(0):
            current = first
            other = second
        elif mode == i64(1):
            current = second
            other = first
        else:
            saved: ptr[u64] = current
            current = other
            other = saved
        index: i64 = i % i64(8)
        source: i64 = (i + i64(3)) % i64(8)
        current[index] = (other[source] + u64(i) * u64(17)) & u64(65535)
        other[index] = (current[index] + other[index]) & u64(65535)
        i = i + i64(1)
    result: u64 = 0
    i = i64(0)
    while i < i64(8):
        result = result + first[i] * u64(i + i64(1))
        result = result + second[i] * u64(i + i64(9))
        i = i + i64(1)
    return result


@compile
def run_dispatch(count: i64, alias: i32) -> u64:
    first: array[u64, 8]
    second: array[u64, 8]
    i: i64 = 0
    while i < i64(8):
        first[i] = u64(i) * u64(7) + u64(1)
        second[i] = u64(i) * u64(13) + u64(3)
        i = i + i64(1)
    p: ptr[u64] = ptr(first[0])
    q: ptr[u64] = ptr(second[0])
    if alias != i32(0):
        q = p
    return dispatch(p, q, count)


@compile
class Effects:
    calls: static[i64] = 0


@compile(attrs={'noinline'})
def record_effect() -> i64:
    Effects.calls = Effects.calls + i64(1)
    return i64(123)


@compile
def run_effects(count: i64) -> i64:
    Effects.calls = i64(0)
    i: i64 = 0
    while i < count:
        unused: i64 = record_effect() * i64(0)
        i = i + i64(1)
    return Effects.calls


def reference(count, alias):
    first = [i * 7 + 1 for i in range(8)]
    second = first if alias else [i * 13 + 3 for i in range(8)]
    current, other = first, second
    for i in range(count):
        if i % 3 == 0:
            current, other = first, second
        elif i % 3 == 1:
            current, other = second, first
        else:
            current, other = other, current
        index, source = i % 8, (i + 3) % 8
        current[index] = (other[source] + i * 17) & 65535
        other[index] = (current[index] + other[index]) & 65535
    return sum(first[i] * (i + 1) + second[i] * (i + 9) for i in range(8))


class OptimizationPipelineTest(unittest.TestCase):
    def test_loop_dispatch_and_aliasing(self):
        for count in [0, 1, 2, 3, 15, 127, 1007]:
            for alias in [0, 1]:
                with self.subTest(count=count, alias=alias):
                    self.assertEqual(run_dispatch(count, alias), reference(count, alias))

    def test_unused_results_do_not_remove_effects(self):
        for count in [0, 1, 17, 1007]:
            self.assertEqual(run_effects(count), count)


if __name__ == '__main__':
    unittest.main()
