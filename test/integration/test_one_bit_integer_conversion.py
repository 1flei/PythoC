"""One-bit integers truncate; bool uses truth conversion."""

import unittest
from pythoc import compile, i1, i32, u1, bool as pc_bool, struct, f64, ptr


@compile
def runtime_casts(value: i32) -> i32:
    return (i32(u1(value)) + 10 * i32(i1(value))
            + 100 * i32(pc_bool(value)))


@compile
def constant_casts() -> i32:
    even: i32 = 2
    odd: i32 = 3
    return (i32(u1(even)) + 10 * i32(i1(even))
            + 100 * i32(u1(odd)) + 1000 * i32(i1(odd))
            + 10000 * i32(pc_bool(even)))


@struct(storage_unit=8)
class Flags:
    ready: u1
    other: u1


@compile
def field_store(value: i32) -> i32:
    flags: Flags = (0, 1)
    flags.ready = value
    return i32(flags.ready) + 10 * i32(flags.other)


@compile
def scalar_truth(value: f64) -> i32:
    local: i32 = 1
    address: ptr[i32] = ptr(local)
    return i32(pc_bool(value)) + 10 * i32(pc_bool(address))


@compile
def literal_truth() -> i32:
    return i32(pc_bool(42)) + 10 * i32(pc_bool(0.25))


class OneBitIntegerConversionTest(unittest.TestCase):
    def test_boolean_conversion_uses_truth_for_float_pointer_and_literal_values(self):
        self.assertEqual(literal_truth(), 11)
        for value in (0.0, -0.0, 0.25, -0.25, float('nan')):
            self.assertEqual(scalar_truth(value), 10 + int(value != 0))

    def test_constants_use_the_same_integer_conversion_as_runtime_values(self):
        self.assertEqual(constant_casts(), 9100)

    def test_signed_and_unsigned_one_bit_values_truncate_before_extension(self):
        for value in (-32, -3, -2, -1, 0, 1, 2, 3, 32):
            with self.subTest(value=value):
                expected = (value & 1) - 10 * (value & 1) + 100 * int(value != 0)
                self.assertEqual(runtime_casts(value), expected)

    def test_one_bit_field_assignment_preserves_neighbors_and_low_bit(self):
        for value in (-32, -3, -2, -1, 0, 1, 2, 3, 32):
            with self.subTest(value=value):
                self.assertEqual(field_store(value), 10 + (value & 1))


if __name__ == '__main__':
    unittest.main()
