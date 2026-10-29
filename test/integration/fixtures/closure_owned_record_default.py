"""Typed aggregates retain the ownership restriction on default values."""

from pythoc import compile, consume, i32, linear


@compile
class Owner:
    value: i32
    token: linear


@compile
def main() -> i32:
    owner: Owner = (7, linear())

    def take(value: Owner = owner) -> i32:
        consume(value.token)
        return value.value

    return take() + take()


if __name__ == '__main__':
    main()
