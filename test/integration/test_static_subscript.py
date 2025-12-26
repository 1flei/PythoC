from pythoc import *
from pythoc.libc.stdio import printf

type_to_double_type = {}
for t, dt in [(i32, i64), (i16, i32), (f32, f64)]:
    type_to_double_type[t] = dt

def make_st(t):
    @struct(suffix=t)
    class St:
        a: t
        b: type_to_double_type[t]
    return St

type_to_st = {}
for t in [i16, i32, f32]:
    type_to_st[t] = make_st(t)


def test_static_subscript():
    type_to_stfunc = {}
    for T in [i16, i32, f32]:
        @compile(suffix=T)
        def test_st() -> type_to_st[T]:
            st: type_to_st[T] = type_to_st[T]()
            st.a = T(1)
            st.b = type_to_double_type[T](2)
            printf("St: a=%d, b=%d\n", i32(st.a), i32(st.b))
            return st
        type_to_stfunc[T] = test_st
    return type_to_stfunc

if __name__ == "__main__":
    type_to_stfunc = test_static_subscript()
    for T in [i16, i32, f32]:
        type_to_stfunc[T]()
    print("test_static_subscript passed!")