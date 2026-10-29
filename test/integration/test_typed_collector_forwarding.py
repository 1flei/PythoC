"""Deferred raw arguments share the normal typed collector ABI."""

import unittest

from pythoc import compile, consume, defer, enum, i32, linear, static, struct


@enum(i32)
class Input:
    Value: i32


@compile
class State:
    total: static[i32] = 0
    released: static[i32] = 0


@compile
class Owner:
    value: i32
    token: linear


@compile
def release_collector(*args: struct[Owner]) -> None:
    consume(args[0].token)
    State.total += args[0].value
    State.released += 1


@compile
def schedule_owned() -> None:
    resource: Owner = (7, linear())
    defer(release_collector, resource)
    State.total += 100


@compile
def observe_owned() -> i32:
    State.total = 0
    State.released = 0
    schedule_owned()
    return State.total * 10 + State.released


@compile
def capture(*args: struct[Input]) -> None:
    match args[0]:
        case (Input.Value, value):
            State.total = State.total + value


@compile
def schedule(value: Input) -> None:
    defer(capture, value)
    State.total = State.total + 100


@compile
def observe() -> i32:
    State.total = 0
    schedule(Input(Input.Value, 7))
    return State.total


@compile
def capture_pair(*items: struct[struct[i32, i32]]) -> None:
    State.total = items[0][0] * 10 + items[0][1]


@compile
def observe_pair(deferred: i32) -> i32:
    State.total = 0
    if deferred != 0:
        defer(capture_pair, (3, 7))
    else:
        capture_pair((3, 7))
    return State.total


class TypedCollectorForwardingTest(unittest.TestCase):
    def test_defer_packs_a_raw_singleton_enum(self):
        self.assertEqual(observe(), 107)

    def test_defer_transfers_a_linear_collector_and_releases_it_once(self):
        self.assertEqual(observe_owned(), 1071)

    def test_raw_tuple_payload_has_the_same_direct_and_deferred_binding(self):
        self.assertEqual(observe_pair(0), 37)
        self.assertEqual(observe_pair(1), 37)

    def test_z_deferred_resource_cannot_be_consumed_twice(self):
        from pythoc.logger import set_raise_on_error
        set_raise_on_error(True)

        @compile(suffix='invalid_deferred_collector_double_consume')
        def invalid() -> None:
            resource: Owner = (7, linear())
            defer(release_collector, resource)
            consume(resource.token)

        with self.assertRaisesRegex(RuntimeError, 'consumed'):
            invalid()


if __name__ == '__main__':
    unittest.main()
