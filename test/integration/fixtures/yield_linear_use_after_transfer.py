"""A token passed to a yield collector cannot be consumed again."""

from pythoc import compile, consume, i32, linear, struct


@compile
def values(*tokens: struct[linear]) -> i32:
    consume(tokens[0])
    yield 7


@compile
def main() -> i32:
    token = linear()
    total: i32 = 0
    for value in values(token):
        total += value
    consume(token)
    return total


if __name__ == '__main__':
    main()
