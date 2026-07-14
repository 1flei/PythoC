"""
Regression test for calling an imported @compile function with bare *args.

The bug: when a function declared as ``def wrapper(fmt: ptr[i8], *args) -> i32``
is defined in one module and imported into another, the cross-module call path
used to declare the callee without the LLVM ``var_arg`` flag.  The caller then
saw a non-varargs function type and reported:

    Function expects 1 arguments, got 2

This test verifies that the imported wrapper can be called with extra positional
arguments from a compiled function in a different source file.
"""

import os
import sys

# Make the helper module importable by bare name.
sys.path.insert(0, os.path.dirname(__file__))

from pythoc import compile, i32, i8, ptr
from varargs_cross_module_helper import wrapper


@compile
def caller() -> i32:
    # Calling an imported bare-*args wrapper with one fixed + one extra arg.
    wrapper(ptr[i8]("hello\n"), 42)
    return 1


def test_imported_varargs_call():
    result = caller()
    assert result == 1, f"Expected 1, got {result}"
    print("OK cross-module bare *args call works")


if __name__ == "__main__":
    test_imported_varargs_call()
