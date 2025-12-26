from pythoc import *

@enum
class Num:
    I32: i32
    F64: f64

MyPair = struct[Num, Num]

@compile
def sum_pair(*args: MyPair) -> i32:
    match args[0], args[1]:
        case (Num.I32, x), (Num.I32, y):
            return x + y
        case (Num.F64, x), (Num.F64, y):
            return i32(x + y)
        case (Num.I32, x), (Num.F64, y):
            return i32(f64(x) + y)
        case (Num.F64, x), (Num.I32, y):
            return i32(x + f64(y))

@compile
def test_pair() -> i32:
    n0 = Num(Num.I32, 1)
    n1 = Num(Num.F64, 2.0)

    n00 = sum_pair(n0, n0)  # 1 + 1 = 2
    n11 = sum_pair(n1, n1)  # 2.0 + 2.0 = 4
    n01 = sum_pair(n0, n1)  # 1 + 2.0 = 3
    n10 = sum_pair(n1, n0)  # 2.0 + 1 = 3
    return n00 + n11 + n01 + n10  # 2 + 4 + 3 + 3 = 12

if __name__ == "__main__":
    result = test_pair()
    print(f"test_pair() = {result}")
    assert result == 12, f"Expected 12, got {result}"
    print("test_varargs_struct_enum passed!")