from __future__ import annotations

from pythoc import *
from pythoc.std.poly import Poly
from pythoc.libc.stdio import printf

@compile
def add_i32(a: i32, b: i32) -> i32:
    printf("Adding i32: %d + %d\n", a, b)
    return a + b

@compile
def add_f64(a: f64, b: f64) -> i32:
    printf("Adding f64: %f + %f\n", a, b)
    return i32(a + b)

@compile
def add_if(a: i32, b: f64) -> i32:
    printf("Adding i32 + f64: %d + %f\n", a, b)
    return i32(f64(a) + b)

@compile
def add_fi(a: f64, b: i32) -> i32:
    printf("Adding f64 + i32: %f + %d\n", a, b)
    return i32(a + f64(b))

add = Poly(add_i32, add_f64, add_if, add_fi)

@compile
def add_str(a: ptr[i8], b: ptr[i8]) -> i32:
    printf("Adding str: %s + %s\n", a, b)
    return 42

@compile
def test_static_poly():
    add(i32(1), i32(2))
    add(f64(1.0), f64(2.0))

add.append(add_str)

@compile
def test_static_poly2():
    add("Hello", "World")

@enum(i32)
class num:
    I32: i32 = 0
    F64: f64

@compile
def mannual_dispatch(n1: num, n2: num) -> i32:
    ret: i32 = 0
    match (n1, n2):
        case ((num.I32, x), (num.I32, y)):
            ret = add_i32(x, y)
        case ((num.F64, x), (num.F64, y)):
            ret = add_f64(x, y)
        case ((num.I32, x), (num.F64, y)):
            ret = add_if(x, y)
        case ((num.F64, x), (num.I32, y)):
            ret = add_fi(x, y)
        case _:
            pass
    return ret

@compile
def test_runtime_poly():
    # Test with same enum type - all combinations covered
    a: num = num(num.I32, 1)
    b: num = num(num.I32, 2)
    c: num = num(num.F64, 1.0)
    d: num = num(num.F64, 2.0)

    # Mannual dispatch to illutrate the dispatch logic
    # mannual_dispatch(a, b)
    # mannual_dispatch(c, d)
    # mannual_dispatch(a, c)
    # mannual_dispatch(c, b)

    # Single dispatch
    add(a, i32(1))  # add_i32(i32, i32)
    
    add(a, b)  # Will dispatch to add_i32(i32, i32)
    add(c, d)  # Will dispatch to add_f64(f64, f64)
    add(a, c)  # Will dispatch to add_if(i32, f64)
    add(c, b)  # Will dispatch to add_fi(f64, i32)
    pass

if __name__ == "__main__":
    test_static_poly()
    test_static_poly2()
    test_runtime_poly()
    print("All poly tests passed!")
