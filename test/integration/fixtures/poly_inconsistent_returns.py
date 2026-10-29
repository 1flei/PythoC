"""All overloads of a dynamic Poly call need a common return type."""

from pythoc import compile, enum, f64, i32
from pythoc.std.poly import Poly


@enum(i32)
class Input:
    Integer: i32
    Float: f64


@compile
def integer(value: i32) -> i32:
    return value


@compile
def floating(value: f64) -> f64:
    return value


operation = Poly(integer, floating)


@compile
def main() -> i32:
    return operation(Input(Input.Integer, 7))


if __name__ == '__main__':
    main()
