"""Same-named handlers keep their identity across dispatch and cache reuse."""

from pythoc import compile, i32
from pythoc.std.poly import Poly

import first
import second
from types_mod import Input


one = Poly(first.integer, first.floating)
two = Poly(second.integer, second.floating)


@compile
def main() -> i32:
    a = one(Input(Input.Integer, 9))
    b = two(Input(Input.Integer, 9))
    c = one(Input(Input.Float, 2.0))
    d = two(Input(Input.Float, 2.0))
    return i32(a != 9 or b != 109 or c != 12 or d != 112)


if __name__ == '__main__':
    assert main() == 0
