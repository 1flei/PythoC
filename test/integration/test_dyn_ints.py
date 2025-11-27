from pythoc import compile, i32
from pythoc.libc import printf
from pythoc import i3, u5, i1, u1

@compile
def test_dyn_ints() -> i32:
    a: i3 = i3(5)      # 5 mod 8 -> 5
    b: u5 = u5(31)     # 31 fits
    c: i1 = i1(1)      # bool-ish
    d: u1 = u1(0)
    printf("a=%d b=%d c=%d d=%d\n", a, b, c, d)
    return 0


@compile
def main() -> i32:
    test_dyn_ints()
    return 0

main()