"""Literal aggregates cannot hide linear values in closure defaults."""

from pythoc import compile, consume, i32, linear, struct


@compile
def main() -> i32:
    token = linear()

    def take(value: struct[linear] = (token,)) -> i32:
        consume(value[0])
        return 1

    return take() + take()


if __name__ == '__main__':
    main()
