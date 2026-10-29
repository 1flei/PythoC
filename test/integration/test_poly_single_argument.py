"""Single-argument Poly calls infer dispatch without explicit preparation."""

import unittest

from pythoc import compile, consume, enum, i32, linear, struct
from pythoc.std.poly import Poly
from pythoc.std.result import result_wrap


Array = struct['bound': i32]
Function = struct['arity': i32]


@enum(i32)
class Operation:
    Array: Array
    Function: Function


@compile
def array_cost(value: Array) -> i32:
    return value.bound


@compile
def function_cost(value: Function) -> i32:
    return value.arity + 10


cost = Poly(array_cost, function_cost)


@compile
def array_other(value: Array) -> i32:
    return value.bound + 100


@compile
def function_other(value: Function) -> i32:
    return value.arity + 200


other_cost = Poly(array_other, function_other)


@compile
def independent_dispatchers() -> i32:
    payload: Array
    payload.bound = 9
    operation = Operation(Operation.Array, payload)
    return cost(operation) * 1000 + other_cost(operation)


@compile
class Owner:
    value: i32
    token: linear


Outcome, outcome = result_wrap(Owner, i32, name='PolyOwnedOutcome')


@compile
def owned_value(resource: Owner) -> i32:
    value: i32 = resource.value
    consume(resource.token)
    return value


@compile
def error_value(error: i32) -> i32:
    return -error


consume_outcome = Poly(owned_value, error_value)


@compile
def dispatch_owned(success: i32) -> i32:
    if success != 0:
        resource: Owner
        resource.value = 7
        resource.token = linear()
        return consume_outcome(outcome.ok(resource))
    return consume_outcome(outcome.err(41))


@compile
def dispatch(function: i32) -> i32:
    if function != 0:
        payload: Function
        payload.arity = 2
        return cost(Operation(Operation.Function, payload))
    payload: Array
    payload.bound = 9
    return cost(Operation(Operation.Array, payload))


class SingleArgumentPolyTest(unittest.TestCase):
    def test_each_struct_payload(self):
        self.assertEqual(dispatch(0), 9)
        self.assertEqual(dispatch(1), 12)

    def test_linear_result_transfers_each_payload_once(self):
        self.assertEqual(dispatch_owned(1), 7)
        self.assertEqual(dispatch_owned(0), -41)

    def test_same_input_types_keep_distinct_overloads(self):
        self.assertEqual(independent_dispatchers(), 9109)


if __name__ == '__main__':
    unittest.main()
