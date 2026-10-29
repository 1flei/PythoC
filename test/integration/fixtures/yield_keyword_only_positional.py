"""Keyword-only yield parameters reject positional arguments."""

from pythoc import compile, i32


@compile
def values(a: i32, *, b: i32 = 8) -> i32:
    yield a * 10 + b


@compile
def main() -> i32:
    total: i32 = 0
    for value in values(1, 2):
        total += value
    return total


if __name__ == '__main__':
    main()
