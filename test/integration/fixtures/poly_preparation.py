"""Run Poly in a fresh process with AOT or the build-time queue guard."""

from pathlib import Path
import sys

import pythoc
from pythoc import compile, enum, f64, i32
from pythoc.std.poly import Poly


@enum(i32)
class Input:
    Integer: i32
    Float: f64


@compile
def integer(value: i32) -> i32:
    return value


@compile
def floating(value: f64) -> i32:
    return i32(value) * 10


dispatch = Poly(integer, floating)
if '--prepare' in sys.argv:
    dispatch.prepare_dynamic_dispatch(Input)


@compile
def main() -> i32:
    result = dispatch(Input(Input.Integer, 7))
    result += dispatch(Input(Input.Float, 2.0))
    return i32(result != 27)


if __name__ == '__main__':
    if '--aot' in sys.argv:
        pythoc.compile_to_executable(str(Path.cwd() / 'native'))
    else:
        assert main() == 0
