#!/usr/bin/env python3
"""
Comprehensive tests for functions including recursion, complex signatures,
function pointers, and various calling patterns.
"""

import unittest
from pythoc import (
    i8, i16, i32, i64,
    u8, u16, u32, u64,
    f32, f64, bool, ptr, array, struct, func, compile
)


# =============================================================================
# Basic Function Calls
# =============================================================================

@compile
def simple_add(a: i32, b: i32) -> i32:
    """Simple two-argument function"""
    return a + b


@compile
def test_simple_call() -> i32:
    """Test simple function call"""
    return simple_add(10, 20)  # 30


@compile
def no_args() -> i32:
    """Function with no arguments"""
    return 42


@compile
def test_no_args_call() -> i32:
    """Test calling function with no arguments"""
    return no_args()


@compile
def many_args(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32, g: i32, h: i32) -> i32:
    """Function with many arguments"""
    return a + b + c + d + e + f + g + h


@compile
def test_many_args_call() -> i32:
    """Test calling function with many arguments"""
    return many_args(1, 2, 3, 4, 5, 6, 7, 8)  # 36


@compile
def void_return(x: ptr[i32]) -> i32:
    """Function that modifies through pointer (simulating void return)"""
    x[0] = 42
    return 0


@compile
def test_void_return() -> i32:
    """Test function with void-like return"""
    x: i32 = 0
    void_return(ptr(x))
    return x  # 42


# =============================================================================
# Recursion
# =============================================================================

@compile
def factorial(n: i32) -> i32:
    """Recursive factorial"""
    if n <= 1:
        return 1
    return n * factorial(n - 1)


@compile
def test_factorial() -> i32:
    """Test recursive factorial"""
    return factorial(5)  # 120


@compile
def fibonacci_rec(n: i32) -> i32:
    """Recursive Fibonacci"""
    if n <= 1:
        return n
    return fibonacci_rec(n - 1) + fibonacci_rec(n - 2)


@compile
def test_fibonacci_rec() -> i32:
    """Test recursive Fibonacci"""
    return fibonacci_rec(10)  # 55


@compile
def sum_to_n(n: i32) -> i32:
    """Recursive sum from 1 to n"""
    if n <= 0:
        return 0
    return n + sum_to_n(n - 1)


@compile
def test_sum_to_n() -> i32:
    """Test recursive sum"""
    return sum_to_n(10)  # 55


@compile
def gcd(a: i32, b: i32) -> i32:
    """Recursive GCD using Euclidean algorithm"""
    if b == 0:
        return a
    return gcd(b, a % b)


@compile
def test_gcd() -> i32:
    """Test recursive GCD"""
    return gcd(48, 18)  # 6


@compile
def power(base: i32, exp: i32) -> i32:
    """Recursive power function"""
    if exp == 0:
        return 1
    if exp == 1:
        return base
    return base * power(base, exp - 1)


@compile
def test_power() -> i32:
    """Test recursive power"""
    return power(2, 10)  # 1024


@compile
def count_digits(n: i32) -> i32:
    """Count digits in number recursively"""
    if n < 10:
        return 1
    return 1 + count_digits(n / 10)


@compile
def test_count_digits() -> i32:
    """Test digit counting"""
    return count_digits(12345)  # 5


# =============================================================================
# Tail Recursion Patterns
# =============================================================================

@compile
def factorial_tail_helper(n: i32, acc: i32) -> i32:
    """Tail-recursive factorial helper"""
    if n <= 1:
        return acc
    return factorial_tail_helper(n - 1, n * acc)


@compile
def factorial_tail(n: i32) -> i32:
    """Tail-recursive factorial"""
    return factorial_tail_helper(n, 1)


@compile
def test_factorial_tail() -> i32:
    """Test tail-recursive factorial"""
    return factorial_tail(5)  # 120


@compile
def sum_tail_helper(n: i32, acc: i32) -> i32:
    """Tail-recursive sum helper"""
    if n <= 0:
        return acc
    return sum_tail_helper(n - 1, acc + n)


@compile
def sum_tail(n: i32) -> i32:
    """Tail-recursive sum"""
    return sum_tail_helper(n, 0)


@compile
def test_sum_tail() -> i32:
    """Test tail-recursive sum"""
    return sum_tail(100)  # 5050


# =============================================================================
# Mutual Recursion
# =============================================================================

@compile
def is_even(n: i32) -> i32:
    """Check if even using mutual recursion"""
    if n == 0:
        return 1
    return is_odd(n - 1)


@compile
def is_odd(n: i32) -> i32:
    """Check if odd using mutual recursion"""
    if n == 0:
        return 0
    return is_even(n - 1)


@compile
def test_mutual_recursion() -> i32:
    """Test mutual recursion"""
    e10: i32 = is_even(10)  # 1
    o10: i32 = is_odd(10)   # 0
    e7: i32 = is_even(7)    # 0
    o7: i32 = is_odd(7)     # 1
    return e10 * 1000 + o10 * 100 + e7 * 10 + o7  # 1001


# =============================================================================
# Function with Different Return Types
# =============================================================================

@compile
def return_i8() -> i8:
    """Return i8"""
    return 127


@compile
def return_i16() -> i16:
    """Return i16"""
    return 32767


@compile
def return_i64() -> i64:
    """Return i64"""
    return 9223372036854775807


@compile
def return_f32() -> f32:
    """Return f32"""
    return f32(3.14)


@compile
def return_f64() -> f64:
    """Return f64"""
    return 3.14159265358979


@compile
def return_bool_true() -> bool:
    """Return bool true"""
    return True


@compile
def return_bool_false() -> bool:
    """Return bool false"""
    return False


@compile
def test_return_types() -> i32:
    """Test various return types"""
    a: i8 = return_i8()
    b: i16 = return_i16()
    c: i64 = return_i64()
    d: f32 = return_f32()
    e: f64 = return_f64()
    t: bool = return_bool_true()
    f: bool = return_bool_false()
    
    result: i32 = 0
    if a == 127:
        result = result + 1
    if b == 32767:
        result = result + 1
    if c > 0:
        result = result + 1
    if d > f32(3.0):
        result = result + 1
    if e > 3.0:
        result = result + 1
    if t:
        result = result + 1
    if not f:
        result = result + 1
    return result  # 7


# =============================================================================
# Function with Different Parameter Types
# =============================================================================

@compile
def mixed_params(a: i8, b: i16, c: i32, d: i64, e: f32, f: f64) -> i32:
    """Function with mixed parameter types"""
    return i32(a) + i32(b) + c + i32(d) + i32(e) + i32(f)


@compile
def test_mixed_params() -> i32:
    """Test mixed parameter types"""
    return mixed_params(1, 2, 3, 4, f32(5.5), 6.5)  # 1+2+3+4+5+6 = 21


@compile
def ptr_param(p: ptr[i32]) -> i32:
    """Function with pointer parameter"""
    return p[0]


@compile
def test_ptr_param() -> i32:
    """Test pointer parameter"""
    x: i32 = 42
    return ptr_param(ptr(x))


@compile
def array_param(arr: ptr[i32], len: i32) -> i32:
    """Function with array parameter (as pointer)"""
    sum: i32 = 0
    i: i32 = 0
    while i < len:
        sum = sum + arr[i]
        i = i + 1
    return sum


@compile
def test_array_param() -> i32:
    """Test array parameter"""
    arr: array[i32, 5] = [1, 2, 3, 4, 5]
    return array_param(arr, 5)  # 15


@compile
def struct_param(s: struct[i32, i32]) -> i32:
    """Function with struct parameter"""
    return s[0] + s[1]


@compile
def test_struct_param() -> i32:
    """Test struct parameter"""
    s: struct[i32, i32] = (10, 20)
    return struct_param(s)  # 30


# =============================================================================
# Function Pointers
# =============================================================================

@compile
def add_op(a: i32, b: i32) -> i32:
    """Addition operation"""
    return a + b


@compile
def sub_op(a: i32, b: i32) -> i32:
    """Subtraction operation"""
    return a - b


@compile
def mul_op(a: i32, b: i32) -> i32:
    """Multiplication operation"""
    return a * b


@compile
def div_op(a: i32, b: i32) -> i32:
    """Division operation"""
    if b == 0:
        return 0
    return a / b


@compile
def apply_op(op: func[[i32, i32], i32], a: i32, b: i32) -> i32:
    """Apply operation via function pointer"""
    return op(a, b)


@compile
def test_func_ptr_apply() -> i32:
    """Test function pointer application"""
    r1: i32 = apply_op(add_op, 10, 5)  # 15
    r2: i32 = apply_op(sub_op, 10, 5)  # 5
    r3: i32 = apply_op(mul_op, 10, 5)  # 50
    r4: i32 = apply_op(div_op, 10, 5)  # 2
    return r1 + r2 + r3 + r4  # 72


@compile
def select_op(code: i32) -> func[[i32, i32], i32]:
    """Select operation by code"""
    if code == 0:
        return add_op
    elif code == 1:
        return sub_op
    elif code == 2:
        return mul_op
    else:
        return div_op


@compile
def test_func_ptr_select() -> i32:
    """Test function pointer selection"""
    op: func[[i32, i32], i32] = select_op(2)
    return op(6, 7)  # 42


@compile
def unary_square(x: i32) -> i32:
    """Square function"""
    return x * x


@compile
def unary_double(x: i32) -> i32:
    """Double function"""
    return x * 2


@compile
def apply_unary(f: func[[i32], i32], x: i32) -> i32:
    """Apply unary function"""
    return f(x)


@compile
def test_unary_func_ptr() -> i32:
    """Test unary function pointer"""
    r1: i32 = apply_unary(unary_square, 5)  # 25
    r2: i32 = apply_unary(unary_double, 5)  # 10
    return r1 + r2  # 35


# =============================================================================
# Nested Function Calls
# =============================================================================

@compile
def inner_func(x: i32) -> i32:
    """Inner function"""
    return x * 2


@compile
def middle_func(x: i32) -> i32:
    """Middle function"""
    return inner_func(x) + 10


@compile
def outer_func(x: i32) -> i32:
    """Outer function"""
    return middle_func(x) + 100


@compile
def test_nested_calls() -> i32:
    """Test nested function calls"""
    return outer_func(5)  # (5*2)+10+100 = 120


@compile
def test_chained_calls() -> i32:
    """Test chained function calls"""
    return simple_add(simple_add(1, 2), simple_add(3, 4))  # (1+2)+(3+4) = 10


@compile
def test_call_in_expression() -> i32:
    """Test function call in expression"""
    x: i32 = 10
    return x + simple_add(5, 3) * 2  # 10 + 8*2 = 26


# =============================================================================
# Multiple Return Points
# =============================================================================

@compile
def multiple_returns(x: i32) -> i32:
    """Function with multiple return points"""
    if x < 0:
        return -1
    if x == 0:
        return 0
    if x < 10:
        return 1
    if x < 100:
        return 2
    return 3


@compile
def test_multiple_returns() -> i32:
    """Test multiple return points"""
    r1: i32 = multiple_returns(-5)   # -1
    r2: i32 = multiple_returns(0)    # 0
    r3: i32 = multiple_returns(5)    # 1
    r4: i32 = multiple_returns(50)   # 2
    r5: i32 = multiple_returns(500)  # 3
    return r1 + r2 + r3 + r4 + r5  # -1+0+1+2+3 = 5


@compile
def early_return_in_loop(n: i32) -> i32:
    """Early return from within loop"""
    i: i32 = 0
    while i < n:
        if i == 7:
            return i * 10
        i = i + 1
    return -1


@compile
def test_early_return() -> i32:
    """Test early return from loop"""
    r1: i32 = early_return_in_loop(10)  # 70
    r2: i32 = early_return_in_loop(5)   # -1
    return r1 + r2  # 69


# =============================================================================
# Local Variables and Scope
# =============================================================================

@compile
def local_vars() -> i32:
    """Function with many local variables"""
    a: i32 = 1
    b: i32 = 2
    c: i32 = 3
    d: i32 = 4
    e: i32 = 5
    f: i32 = a + b
    g: i32 = c + d
    h: i32 = e + f
    i: i32 = g + h
    return i  # (3+4) + (5 + (1+2)) = 7 + 8 = 15


@compile
def test_local_vars() -> i32:
    """Test local variables"""
    return local_vars()


@compile
def shadow_outer(x: i32) -> i32:
    """Function demonstrating variable shadowing in different scopes"""
    result: i32 = x
    if x > 0:
        y: i32 = x * 2
        result = result + y
    # y is not accessible here
    return result


@compile
def test_shadow() -> i32:
    """Test variable shadowing"""
    return shadow_outer(5)  # 5 + 10 = 15


# =============================================================================
# Complex Call Patterns
# =============================================================================

@compile
def compute_chain(x: i32) -> i32:
    """Chain of computations"""
    step1: i32 = x + 10
    step2: i32 = step1 * 2
    step3: i32 = step2 - 5
    step4: i32 = step3 / 3
    return step4


@compile
def test_compute_chain() -> i32:
    """Test computation chain"""
    return compute_chain(5)  # ((5+10)*2-5)/3 = 25/3 = 8


@compile
def conditional_call(flag: i32, x: i32) -> i32:
    """Conditional function call"""
    if flag == 1:
        return unary_square(x)
    else:
        return unary_double(x)


@compile
def test_conditional_call() -> i32:
    """Test conditional function call"""
    r1: i32 = conditional_call(1, 5)  # 25
    r2: i32 = conditional_call(0, 5)  # 10
    return r1 + r2  # 35


@compile
def recursive_with_helper(n: i32) -> i32:
    """Recursive function that calls helper"""
    if n <= 0:
        return 0
    return simple_add(n, recursive_with_helper(n - 1))


@compile
def test_recursive_with_helper() -> i32:
    """Test recursive with helper"""
    return recursive_with_helper(5)  # 15


# =============================================================================
# Edge Cases
# =============================================================================

@compile
def identity(x: i32) -> i32:
    """Identity function"""
    return x


@compile
def test_identity() -> i32:
    """Test identity function"""
    return identity(42)


@compile
def constant_return() -> i32:
    """Function returning constant"""
    return 42


@compile
def test_constant_return() -> i32:
    """Test constant return"""
    return constant_return()


@compile
def deep_recursion_sum(n: i32) -> i32:
    """Deep recursion test (be careful with stack)"""
    if n <= 0:
        return 0
    return n + deep_recursion_sum(n - 1)


@compile
def test_deep_recursion() -> i32:
    """Test deeper recursion"""
    return deep_recursion_sum(50)  # 1275


# =============================================================================
# Test Runner
# =============================================================================

class TestBasicCalls(unittest.TestCase):
    def test_simple_call(self):
        self.assertEqual(test_simple_call(), 30)
    
    def test_no_args(self):
        self.assertEqual(test_no_args_call(), 42)
    
    def test_many_args(self):
        self.assertEqual(test_many_args_call(), 36)
    
    def test_void_return(self):
        self.assertEqual(test_void_return(), 42)


class TestRecursion(unittest.TestCase):
    def test_factorial(self):
        self.assertEqual(test_factorial(), 120)
    
    def test_fibonacci(self):
        self.assertEqual(test_fibonacci_rec(), 55)
    
    def test_sum_to_n(self):
        self.assertEqual(test_sum_to_n(), 55)
    
    def test_gcd(self):
        self.assertEqual(test_gcd(), 6)
    
    def test_power(self):
        self.assertEqual(test_power(), 1024)
    
    def test_count_digits(self):
        self.assertEqual(test_count_digits(), 5)


class TestTailRecursion(unittest.TestCase):
    def test_factorial_tail(self):
        self.assertEqual(test_factorial_tail(), 120)
    
    def test_sum_tail(self):
        self.assertEqual(test_sum_tail(), 5050)


class TestMutualRecursion(unittest.TestCase):
    def test_mutual(self):
        self.assertEqual(test_mutual_recursion(), 1001)


class TestReturnTypes(unittest.TestCase):
    def test_various_types(self):
        self.assertEqual(test_return_types(), 7)


class TestParameterTypes(unittest.TestCase):
    def test_mixed_params(self):
        self.assertEqual(test_mixed_params(), 21)
    
    def test_ptr_param(self):
        self.assertEqual(test_ptr_param(), 42)
    
    def test_array_param(self):
        self.assertEqual(test_array_param(), 15)
    
    def test_struct_param(self):
        self.assertEqual(test_struct_param(), 30)


class TestFunctionPointers(unittest.TestCase):
    def test_apply(self):
        self.assertEqual(test_func_ptr_apply(), 72)
    
    def test_select(self):
        self.assertEqual(test_func_ptr_select(), 42)
    
    def test_unary(self):
        self.assertEqual(test_unary_func_ptr(), 35)


class TestNestedCalls(unittest.TestCase):
    def test_nested(self):
        self.assertEqual(test_nested_calls(), 120)
    
    def test_chained(self):
        self.assertEqual(test_chained_calls(), 10)
    
    def test_in_expression(self):
        self.assertEqual(test_call_in_expression(), 26)


class TestMultipleReturns(unittest.TestCase):
    def test_multiple(self):
        self.assertEqual(test_multiple_returns(), 5)
    
    def test_early(self):
        self.assertEqual(test_early_return(), 69)


class TestLocalVars(unittest.TestCase):
    def test_local(self):
        self.assertEqual(test_local_vars(), 15)
    
    def test_shadow(self):
        self.assertEqual(test_shadow(), 15)


class TestComplexPatterns(unittest.TestCase):
    def test_chain(self):
        self.assertEqual(test_compute_chain(), 8)
    
    def test_conditional(self):
        self.assertEqual(test_conditional_call(), 35)
    
    def test_recursive_helper(self):
        self.assertEqual(test_recursive_with_helper(), 15)


class TestEdgeCases(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(test_identity(), 42)
    
    def test_constant(self):
        self.assertEqual(test_constant_return(), 42)
    
    def test_deep_recursion(self):
        self.assertEqual(test_deep_recursion(), 1275)


if __name__ == '__main__':
    unittest.main()
