"""Dynamic dispatch transfers ownership of a linear sum exactly once."""

from pythoc import compile, consume, enum, i32, linear
from pythoc.std.poly import Poly


@compile
class Owner:
    value: i32
    token: linear


@enum(i32)
class Input:
    Owned: Owner
    Copy: i32


@compile
def owned(item: Owner) -> i32:
    consume(item.token)
    return item.value


@compile
def copy(item: i32) -> i32:
    return item


operation = Poly(owned, copy)


@compile
def main() -> i32:
    owner: Owner = (7, linear())
    value = Input(Input.Owned, owner)
    a = operation(value)
    return a + operation(value)


if __name__ == '__main__':
    main()
