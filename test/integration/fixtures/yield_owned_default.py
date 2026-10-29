"""Yield closures cannot repeat consumption of a linear default."""

from pythoc import compile, consume, i32, linear


@compile
def main() -> i32:
    token = linear()

    def take(value: linear = token) -> i32:
        consume(value)
        yield 1

    total: i32 = 0
    for value in take():
        total += value
    for value in take():
        total += value
    return total


if __name__ == '__main__':
    main()
