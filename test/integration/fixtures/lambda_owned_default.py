"""Lambda defaults obey the same ownership boundary as named closures."""

from pythoc import compile, consume, i32, linear


@compile
def main() -> i32:
    token = linear()
    take = lambda value=token: consume(value)
    take()
    take()
    return 2


if __name__ == '__main__':
    main()
