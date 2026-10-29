"""Yield expansion consumes evaluated arguments once, in source order."""

import unittest

from pythoc import compile, consume, i32, linear, ptr
from pythoc.std.result import result_wrap


Result, result = result_wrap(i32, i32, name='YieldArgumentSequence')


@compile
def step(count: ptr[i32]) -> Result:
    count[0] += 1
    return result.ok(count[0])


@compile
def compose() -> i32:
    count: i32 = 0
    outcome = result.do(a * 100 + b * 10 + c
                        for a in result.bind(step(ptr(count)))
                        for b in result.bind(step(ptr(count)))
                        for c in result.bind(step(ptr(count))))
    match outcome:
        case (Result.Ok, value):
            return value * 10 + count
        case Result.Err:
            return -1


@compile
def choose(count: ptr[i32], fail: i32) -> Result:
    count[0] += 1
    if fail != 0:
        return result.err(7)
    return result.ok(count[0])


@compile
def short_circuit() -> i32:
    count: i32 = 0
    outcome = result.do(a + b + c
                        for a in result.bind(choose(ptr(count), 0))
                        for b in result.bind(choose(ptr(count), 1))
                        for c in result.bind(choose(ptr(count), 0)))
    match outcome:
        case Result.Ok:
            return -1
        case (Result.Err, error):
            return count * 10 + error


@compile
class Owner:
    value: i32
    token: linear


Owned, owned = result_wrap(Owner, i32, name='YieldArgumentOwner')


@compile
def acquire(count: ptr[i32]) -> Owned:
    count[0] += 1
    value: Owner = (count[0], linear())
    return owned.ok(value)


@compile
def release(value: Owner, count: ptr[i32]) -> i32:
    consume(value.token)
    count[0] += 1
    return value.value


@compile
def owned_sequence() -> i32:
    made: i32 = 0
    released: i32 = 0
    outcome = result.do(release(a, ptr(released))
                        for a in owned.bind(acquire(ptr(made))))
    match outcome:
        case (Result.Ok, value):
            return value * 100 + made * 10 + released
        case Result.Err:
            return -1


@compile
def bump(count: ptr[i32]) -> i32:
    count[0] += 1
    return count[0]


@compile
def closure_sequence() -> i32:
    count: i32 = 0

    def values(value: i32) -> i32:
        yield value

    total: i32 = 0
    for value in values(bump(ptr(count))):
        total += value
    return total * 10 + count


class YieldArgumentOnceTest(unittest.TestCase):
    def test_argument_side_effects_run_once_in_order(self):
        self.assertEqual(compose(), 1233)

    def test_failure_skips_later_argument_evaluation(self):
        self.assertEqual(short_circuit(), 27)

    def test_yield_closure_arguments_run_once(self):
        self.assertEqual(closure_sequence(), 11)

    def test_owned_constructors_and_destructors_balance(self):
        self.assertEqual(owned_sequence(), 111)


if __name__ == '__main__':
    unittest.main()
