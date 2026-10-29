"""Yield calls reject keyword splats at the shared call boundary."""

from pythoc import compile, i32


@compile
def digits(a: i32, b: i32 = 8) -> i32:
    yield a * 10 + b


@compile
def main() -> i32:
    total: i32 = 0
    for value in digits(**{'a': i32(1), 'b': i32(2)}):
        total += value
    return total


if __name__ == '__main__':
    main()
