from __future__ import annotations
from pythoc import i32, compile
from pythoc.libc.stdio import printf

@compile
def test_empty_ann_assign() -> i32:
    a: i32
    printf("undefined %d\n", a)
    return a

test_empty_ann_assign()
