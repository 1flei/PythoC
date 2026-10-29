"""Every dynamic payload type needs a matching Poly overload."""

from pythoc import compile, enum, f64, i32
from pythoc.std.poly import Poly


@enum(i32)
class Input:
    Integer: i32
    Float: f64


@compile
def integer(value: i32) -> i32:
    return value


operation = Poly(integer)


@compile
def main() -> i32:
    return operation(Input(Input.Integer, 7))


if __name__ == '__main__':
    main()
