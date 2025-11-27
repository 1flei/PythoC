from pythoc import *
from pythoc.libc.stdio import printf
from pythoc.libc.stdlib import malloc

@compile
def test_runtime_type():
    MyType = i32
    x: MyType = 1
    for T in [i32, f64]:
        pt = ptr[T]
        p: pt = malloc(sizeof(T))
        p[0] = T(x)

        printf("x: %d, px: %d\n", x, p[0])

test_runtime_type()