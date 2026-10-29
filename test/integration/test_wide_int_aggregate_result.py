"""Wide integer arguments compose with indirect aggregate and void returns."""

import unittest

from pythoc import compile, i64, i128, ptr


@compile
class Record:
    low: i64
    high: i64
    bias: i64


@compile
def make_record(value: i128, bias: i64) -> Record:
    result: Record
    result.low = i64(value)
    result.high = i64(value >> 64)
    result.bias = bias
    return result


@compile
def consume_wide(value: i128, output: ptr[i64]) -> None:
    output[0] = i64(value >> 64)


@compile
def combine(value: i128, item: Record) -> Record:
    return make_record(value + i128(item.low), item.bias)


@compile
def native_caller() -> i64:
    result = make_record((i128(3) << 80) + 7, 9)
    observed: i64 = 0
    consume_wide(i128(5) << 64, ptr(observed))
    return result.low + result.high + result.bias + observed


class WideAggregateResultTest(unittest.TestCase):
    def test_python_to_native_aggregate_result(self):
        result = make_record((3 << 80) + 7, 9)
        self.assertEqual(result.low, 7)
        self.assertEqual(result.high, 3 << 16)
        self.assertEqual(result.bias, 9)

    def test_negative_wide_value(self):
        result = make_record(-((2 << 64) + 7), -3)
        self.assertEqual(result.low, -7)
        self.assertEqual(result.high, -3)
        self.assertEqual(result.bias, -3)

    def test_native_calls_preserve_the_original_abi(self):
        self.assertEqual(native_caller(), (3 << 16) + 21)

    def test_wide_argument_and_byval_record(self):
        item = make_record(7, 11)
        result = combine((5 << 64) + 9, item)
        self.assertEqual(result.low, 16)
        self.assertEqual(result.high, 5)
        self.assertEqual(result.bias, 11)


if __name__ == '__main__':
    unittest.main()
