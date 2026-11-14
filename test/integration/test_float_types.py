import math
import unittest

from pythoc import array, bool as pc_bool, i64, static
from pythoc import f16, bf16, f32, f64, f128, i32, compile
from pythoc.libc.stdio import printf

def gen_fadd(T):
    @compile(suffix=T)
    def add(a: T, b: T) -> T:
        return a + b
    return add

def gen_test_fx(T):
    fadd = gen_fadd(T)
    @compile(suffix=T)
    def test_fx() -> i32:
        x: T = T(1.0)
        y: T = T(2.0)
        z: T = fadd(x, y)
        printf("%f + %f = %f\n", x, y, z)
        return 0
    return test_fx

# f16/bf16/f128 requires higher version of gcc/clang
# test_f16 = gen_test_fx(f16)
# test_bf16 = gen_test_fx(bf16)
test_f32 = gen_test_fx(f32)
test_f64 = gen_test_fx(f64)
# test_f128 = gen_test_fx(f128)

@compile
def main() -> i32:
    printf("=== Float Types Test ===\n\n")
    
    # printf("--- f16 test ---\n")
    # test_f16()
    # printf("\n")
    
    # printf("--- bf16 test ---\n")
    # test_bf16()
    # printf("\n")
    
    printf("--- f32 test ---\n")
    test_f32()
    printf("\n")
    
    printf("--- f64 test ---\n")
    test_f64()
    printf("\n")
    
    # printf("--- f128 test ---\n")
    # test_f128()
    # printf("\n")
    
    printf("=== All Tests Complete ===\n")
    return 0


def float_operations(T):
    @compile(suffix=T)
    def negate(value: f64) -> f64:
        converted: T = T(value)
        return f64(-converted)

    @compile(suffix=T)
    def truth(value: f64) -> i32:
        converted: T = T(value)
        return i32(pc_bool(converted))

    @compile(suffix=T)
    def logical_not(value: f64) -> i32:
        converted: T = T(value)
        return i32(not converted)

    return negate, truth, logical_not


def float_constants(T):
    @compile(suffix=T)
    def read(index: i32) -> f64:
        values: static[array[T, 4]] = (
            -T(2.5), -T(0.0), -(-T(0.0)), -T(-2.5),
        )
        return f64(values[index])

    return read


class FloatingUnaryTruthinessTest(unittest.TestCase):
    def test_negated_constants_initialize_static_storage(self):
        for T in (f16, f32, f64):
            with self.subTest(type=T):
                read = float_constants(T)
                self.assertEqual(read(0), -2.5)
                self.assertEqual(math.copysign(1.0, read(1)), -1.0)
                self.assertEqual(math.copysign(1.0, read(2)), 1.0)
                self.assertEqual(read(3), 2.5)

    def test_f64_negation_preserves_zero_sign(self):
        negate, _, _ = float_operations(f64)
        self.assertEqual(math.copysign(1.0, negate(0.0)), -1.0)
        self.assertEqual(math.copysign(1.0, negate(-0.0)), 1.0)

    def test_half_negation_and_truthiness(self):
        negate, truth, logical_not = float_operations(f16)
        self.assertEqual(negate(2.5), -2.5)
        self.assertEqual(negate(-2.5), 2.5)
        self.assertEqual(math.copysign(1.0, negate(0.0)), -1.0)
        for value in (0.0, -0.0, 2.5, -2.5, math.inf, -math.inf, math.nan):
            with self.subTest(value=value):
                self.assertEqual(truth(value), int(bool(value)))
                self.assertEqual(logical_not(value), int(not value))

    def test_f32_negation_preserves_zero_sign(self):
        negate, _, _ = float_operations(f32)
        self.assertEqual(math.copysign(1.0, negate(0.0)), -1.0)
        self.assertEqual(math.copysign(1.0, negate(-0.0)), 1.0)


def constant_conversions(T):
    captured = T(1.9999)

    @compile(suffix=T)
    def literal_to_int() -> i32:
        return i32(T(1.9999))

    @compile(suffix=T)
    def captured_to_int() -> i32:
        return i32(captured)

    @compile(suffix=T)
    def tiny_to_bool() -> i32:
        return i32(pc_bool(T(1.0e-100)))

    @compile(suffix=T)
    def static_arithmetic() -> i32:
        value: static[T] = (T(1.5) + T(2.5)) * T(2.0) / T(4.0)
        return i32(value)

    return literal_to_int, captured_to_int, tiny_to_bool, static_arithmetic


large_quad = f128(9007199254740993)


@compile
def wide_integer_literal() -> i64:
    return i64(f128(9007199254740993))


@compile
def captured_wide_integer() -> i64:
    return i64(large_quad)


class FloatingConstantConversionTest(unittest.TestCase):
    def test_literals_and_captures_follow_target_rounding(self):
        for T, integer, truth in (
                (f16, 2, 0), (bf16, 2, 0), (f32, 1, 0),
                (f64, 1, 1), (f128, 1, 1)):
            with self.subTest(type=T):
                literal, captured, tiny, static_value = constant_conversions(T)
                self.assertEqual(literal(), integer)
                self.assertEqual(captured(), integer)
                self.assertEqual(tiny(), truth)
                self.assertEqual(static_value(), 2)

    def test_quad_integer_literals_and_captures_keep_all_bits(self):
        self.assertEqual(wide_integer_literal(), 9007199254740993)
        self.assertEqual(captured_wide_integer(), 9007199254740993)


if __name__ == "__main__":
    main()
    print("Float types test passed!")
    unittest.main()
