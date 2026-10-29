"""Reusable closures cannot store linear default values."""

from pythoc import compile, consume, i32, linear


@compile
def main() -> i32:
    token = linear()

    def take(value: linear = token) -> i32:
        consume(value)
        return 1

    return take() + take()


if __name__ == '__main__':
    main()
