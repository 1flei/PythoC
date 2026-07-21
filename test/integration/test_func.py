#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function Pointer Demo

Demonstrates various uses of function pointers:
1. Passing functions as arguments
2. Function pointer arrays (function tables)
3. Callbacks
"""

from pythoc import compile
from pythoc.builtin_entities import i32, i64, f64, func, void, ptr, u64, static
from pythoc.libc.stdlib import malloc
from pythoc.libc.stdio import printf


@compile
def add(x: i32, y: i32) -> i32:
    return x + y


@compile
def subtract(x: i32, y: i32) -> i32:
    return x - y


@compile
def multiply(x: i32, y: i32) -> i32:
    return x * y


@compile
def divide(x: i32, y: i32) -> i32:
    if y == 0:
        return 0
    return x / y


@compile
def apply_binary_op(op: func[i32, i32, i32], x: i32, y: i32) -> i32:
    """Apply a binary operation to two numbers"""
    return op(x, y)


@compile
def square(x: i32) -> i32:
    return x * x


@compile
def double(x: i32) -> i32:
    return x + x


@compile
def apply_unary_op(op: func[i32, i32], x: i32) -> i32:
    """Apply a unary operation to a number"""
    return op(x)


@compile
def select_operation(op_code: i32) -> func[i32, i32, i32]:
    """Select an operation based on code (function table)"""
    if op_code == 0:
        return add
    elif op_code == 1:
        return subtract
    elif op_code == 2:
        return multiply
    else:
        return divide


@compile
def calculator(op_code: i32, x: i32, y: i32) -> i32:
    """Simple calculator using function pointers"""
    operation: func[i32, i32, i32] = select_operation(op_code)
    return operation(x, y)


@compile
def main() -> i32:
    """Test various function pointer operations"""
    # Test 1: Direct function pointer call
    result1: i32 = apply_binary_op(add, 10, 5)  # 15

    # Test 2: Different operations
    result2: i32 = apply_binary_op(multiply, 3, 4)  # 12

    # Test 3: Unary operations
    result3: i32 = apply_unary_op(square, 4)  # 16
    result4: i32 = apply_unary_op(double, 5)  # 10

    # Test 4: Calculator with function table
    calc1: i32 = calculator(0, 20, 10)  # add: 30
    calc2: i32 = calculator(1, 30, 10)  # subtract: 20
    calc3: i32 = calculator(2, 5, 3)    # multiply: 15
    calc4: i32 = calculator(3, 40, 4)   # divide: 10

    # Sum all results
    total: i32 = result1 + result2 + result3 + result4
    total = total + calc1 + calc2 + calc3 + calc4

    return total  # 15 + 12 + 16 + 10 + 30 + 20 + 15 + 10 = 128


@compile
def static_func_ptr_call() -> i32:
    """Initialize a static function pointer and call through it."""
    s: static[func[i32, i32, i32]] = add
    return s(2, 3)


@compile
def static_func_ptr_return() -> ptr[func[i32, i32, i32]]:
    """Return the address of a static function pointer."""
    s: static[func[i32, i32, i32]] = add
    return ptr(s)

@compile
def foo(size: u64) -> ptr[void]:
    return ptr[void](0)


@compile
def get_f() -> ptr[func[u64, ptr[void]]]:
    s: static[func[u64, ptr[void]]] = foo
    return ptr(s)

@compile
def bar(ptr_: ptr[void], size: u64) -> ptr[void]:
    return ptr[void](0)


@compile
def set_f(f: func[ptr[void], u64, ptr[void]]) -> void:
    g: func[ptr[void], u64, ptr[void]] = (f if f else bar)


@compile
def ternary_func_ptr(op: i32, x: i32, y: i32) -> i32:
    f: func[i32, i32, i32] = (add if op == 0 else subtract)
    return f(x, y)


@compile
def ternary_int_literals(flag: i32) -> i32:
    return (1 if flag else 2)


@compile
def ternary_runtime_mixed(flag: i32, x: i32) -> i32:
    return (1 if flag else x)


@compile
def ternary_pc_widening(flag: i32, a: i32, b: i64) -> i64:
    return (a if flag else b)


@compile
def ternary_pc_int_float(flag: i32, a: i32, b: f64) -> f64:
    return (a if flag else b)


@compile
def ternary_pyconst_float_pc_int(flag: i32, x: i32) -> i32:
    return (1.5 if flag else x)


@compile
def ternary_pyconst_mixed_literals(flag: i32) -> f64:
    return (1 if flag else 2.0)


@compile
def ternary_compile_time_true() -> i32:
    return (1 if True else 2)


@compile
def ternary_compile_time_false() -> i32:
    return (1 if False else 2)


@compile
def _bump(p: ptr[i32]) -> void:
    p[0] = p[0] + 1


@compile
def ternary_void_else_branch(flag: i32) -> i32:
    """C assert-macro shape: `cond ? (void)0 : side_effect()`.

    One branch is void, so the whole ternary is void and the branches are
    evaluated for side effects only.
    """
    cell: i32 = 0
    (void(0) if flag else _bump(ptr(cell)))
    return cell


@compile
def ternary_void_then_branch(flag: i32) -> i32:
    """Same void ternary with the void branch on the then side."""
    cell: i32 = 0
    (_bump(ptr(cell)) if flag else void(0))
    return cell


@compile
def void_discard_cast(x: i32) -> i32:
    """(void)expr discard cast: operand is evaluated, result discarded."""
    void(x + 1)
    return x


import unittest


class TestFunc(unittest.TestCase):
    """Test function pointer operations"""

    def test_main(self):
        """Test all function pointer operations"""
        # Expected: 15 + 12 + 16 + 10 + 30 + 20 + 15 + 10 = 128
        result = main()
        self.assertEqual(result, 128)

    def test_static_func_ptr(self):
        """Static function pointer initialized from a @compile def."""
        self.assertEqual(static_func_ptr_call(), 5)
        # The returned pointer should be a non-null function pointer.
        p = static_func_ptr_return()
        self.assertNotEqual(int(p), 0)

    def test_static_func_ptr_void_ptr(self):
        """Static function pointer with ptr[void]/u64 signature."""
        p = get_f()
        self.assertNotEqual(int(p), 0)

    def test_set_f_null_and_valid(self):
        """Ternary function pointer assignment with null/valid input."""
        set_f(0)
        set_f(bar)

    def test_ternary_func_ptr(self):
        """Ternary expression selecting a function pointer."""
        self.assertEqual(ternary_func_ptr(0, 2, 3), 5)
        self.assertEqual(ternary_func_ptr(1, 2, 3), -1)

    def test_ternary_int_literals(self):
        """Ternary expression with two integer literals."""
        self.assertEqual(ternary_int_literals(1), 1)
        self.assertEqual(ternary_int_literals(0), 2)

    def test_ternary_runtime_mixed(self):
        """Ternary expression with a literal branch and a runtime branch."""
        self.assertEqual(ternary_runtime_mixed(1, 42), 1)
        self.assertEqual(ternary_runtime_mixed(0, 42), 42)

    def test_ternary_pc_widening(self):
        """PC + PC integer widening follows C conditional-expression rules."""
        self.assertEqual(ternary_pc_widening(0, 10, 20), 20)
        self.assertEqual(ternary_pc_widening(1, 10, 20), 10)

    def test_ternary_pc_int_float(self):
        """PC + PC int/float unifies to f64."""
        self.assertEqual(ternary_pc_int_float(0, 10, 2.5), 2.5)
        self.assertEqual(ternary_pc_int_float(1, 10, 2.5), 10.0)

    def test_ternary_pyconst_float_pc_int(self):
        """pyconst float + PC int: PC type wins, float is truncated to int."""
        self.assertEqual(ternary_pyconst_float_pc_int(1, 42), 1)
        self.assertEqual(ternary_pyconst_float_pc_int(0, 42), 42)

    def test_ternary_pyconst_mixed_literals(self):
        """pyconst int + pyconst float unify to f64."""
        self.assertEqual(ternary_pyconst_mixed_literals(1), 1.0)
        self.assertEqual(ternary_pyconst_mixed_literals(0), 2.0)

    def test_ternary_compile_time_flag(self):
        """Compile-time Python flag keeps Python semantics."""
        self.assertEqual(ternary_compile_time_true(), 1)
        self.assertEqual(ternary_compile_time_false(), 2)

    def test_ternary_void_branch(self):
        """Ternary with a void branch is void; side effects still run."""
        self.assertEqual(ternary_void_else_branch(1), 0)
        self.assertEqual(ternary_void_else_branch(0), 1)
        self.assertEqual(ternary_void_then_branch(1), 1)
        self.assertEqual(ternary_void_then_branch(0), 0)

    def test_void_discard_cast(self):
        """(void)expr evaluates the operand and discards the result."""
        self.assertEqual(void_discard_cast(41), 41)


if __name__ == '__main__':
    unittest.main()
