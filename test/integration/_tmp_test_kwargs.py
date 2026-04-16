"""Minimal test for kwargs -> struct."""
from pythoc import compile, i32, struct

Rect = struct["width": i32, "height": i32]

@compile
def make_rect(**kwargs: Rect) -> i32:
    return kwargs.width * kwargs.height

@compile
def test_rect_kwargs() -> i32:
    return make_rect(width=i32(6), height=i32(7))

print("result:", test_rect_kwargs())
