#!/usr/bin/env python3
"""Integration tests for generator expression replay-by-expansion support."""

import unittest

from pythoc import compile, i32, i64


@compile
def seq(begin: i32, end: i32) -> i32:
    i: i32 = begin
    while i < end:
        yield i
        i = i + 1


@compile
def test_lazy_chain() -> i32:
    input = [1]
    y = (x + 1 for x in input)
    z = (2 * x for x in y)
    zz = (x * x for x in z)
    for x in zz:
        return x
    return 0


@compile
def test_replay_two_for_loops() -> i32:
    xs = seq(3, 5)
    ys = (x + 1 for x in xs)

    y: i32 = 0
    z: i32 = 0

    for x in ys:
        y = x * 2
        break

    for x in ys:
        z = x * 3
        break

    return y * 10 + z

@compile
def test_multiple_layers() -> i32:
    x0 = seq(1, 10)
    x1 = (x + 1 for x in x0)
    x2 = (x + xx + 1 for x in x0 for xx in x1)
    x3 = (x + xx + xxx + 1 for x in x0 for xx in x1 for xxx in x2)
    x4 = (x + xx + xxx + xxxx + 1 for x in x0 for xx in x1 for xxx in x2 for xxxx in x3)
    sum: i64 = 0
    for x in x4:
        sum += x
    return sum


class TestGeneratorExpression(unittest.TestCase):
    def test_lazy_chain(self):
        self.assertEqual(test_lazy_chain(), 16)

    def test_replay_two_for_loops(self):
        self.assertEqual(test_replay_two_for_loops(), 92)

    def test_multiple_layers(self):
        self.assertEqual(test_multiple_layers(), 2066242608)


if __name__ == "__main__":
    unittest.main()
