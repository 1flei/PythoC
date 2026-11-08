"""A library can pin a binding during its first import."""

from pythoc import compile, effect, i32

effect.import_pin = 37


@compile
def read_pin() -> i32:
    return i32(effect.import_pin)
