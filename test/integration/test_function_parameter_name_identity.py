"""Named function signatures remain distinct through record canonicalization."""

import unittest

from pythoc import compile, func, i32, ptr
from pythoc.std.poly import Poly


def callback_holder(signature):
    @compile
    class Callbacks:
        callback: signature

    return Callbacks


NamedX = callback_holder(func['x':i32, i32])
NamedY = callback_holder(func['y':i32, i32])
UnnamedFirst = callback_holder(func[i32, i32])
UnnamedSecond = callback_holder(func[i32, i32])


@compile
def first(value: ptr[NamedX]) -> i32:
    return 11


@compile
def second(value: ptr[NamedY]) -> i32:
    return 22


choose = Poly(first, second)


@compile
def dispatch_named_callbacks() -> i32:
    a: NamedX
    b: NamedY
    return choose(ptr(a)) * 100 + choose(ptr(b))


@compile
def callback_x(x: i32) -> i32:
    return x + 1


@compile
def callback_y(y: i32) -> i32:
    return y + 10


@compile
def call_named_callbacks() -> i32:
    a: NamedX
    b: NamedY
    a.callback = callback_x
    b.callback = callback_y
    return a.callback(x=3) * 100 + b.callback(y=5)


@compile
def call_unnamed_callbacks() -> i32:
    a: UnnamedFirst
    b: UnnamedSecond
    a.callback = callback_x
    b.callback = callback_y
    return a.callback(3) * 100 + b.callback(5)


class FunctionParameterNameIdentityTest(unittest.TestCase):
    def test_poly_preserves_overloads_with_different_callback_parameter_names(self):
        self.assertEqual(dispatch_named_callbacks(), 1122)

    def test_named_callback_fields_accept_their_own_keyword(self):
        self.assertEqual(call_named_callbacks(), 415)

    def test_unnamed_callback_fields_use_positional_c_style_calls(self):
        self.assertEqual(call_unnamed_callbacks(), 415)


if __name__ == '__main__':
    unittest.main()
