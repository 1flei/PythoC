"""Positional-only yield parameters reject keyword binding."""

from pythoc import compile, i32


@compile
def values(a: i32, /) -> i32:
    yield a


@compile
def main() -> i32:
    total: i32 = 0
    for value in values(a=1):
        total += value
    return total


if __name__ == '__main__':
    main()
