"""Required keyword-only yield parameters must be supplied."""

from pythoc import compile, i32


@compile
def values(a: i32, *, b: i32) -> i32:
    yield a + b


@compile
def main() -> i32:
    total: i32 = 0
    for value in values(1):
        total += value
    return total


if __name__ == '__main__':
    main()
