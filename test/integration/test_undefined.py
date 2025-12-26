from __future__ import annotations
from pythoc import i32, compile
from pythoc.libc.stdio import printf

@compile
def test_empty_ann_assign() -> i32:
    a: i32
    printf("undefined %d\n", a)
    return a

if __name__ == "__main__":
    # This test just verifies that undefined variables compile correctly
    # The value is undefined, so we just check it runs without error
    test_empty_ann_assign()
    print("test_undefined passed!")
