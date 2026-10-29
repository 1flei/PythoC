"""A payloadless variant cannot silently fall through dynamic dispatch."""

from pythoc import compile, enum, i32
from pythoc.std.poly import Poly


@enum(i32)
class Input:
    Integer: i32
    Absent: None


@compile
def integer(value: i32) -> i32:
    return value


operation = Poly(integer)


@compile
def main() -> i32:
    return operation(Input(Input.Integer, 7))


if __name__ == '__main__':
    main()
