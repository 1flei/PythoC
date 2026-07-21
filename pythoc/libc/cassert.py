"""
Assertion support (<assert.h>).

The ``assert`` macro expands to a call of the platform assertion handler;
on macOS that is ``__assert_rtn``, on glibc ``__assert_fail``.  The UCRT
handler is the wide-string ``_wassert``, which is not bound yet, so this
module refuses to import on Windows.
"""

from ..decorators import extern
from ..builtin_entities import i8, i32, u32, ptr, void
from ._platform import IS_MACOS, IS_LINUX


if IS_MACOS:
    @extern(lib="c")
    def __assert_rtn(func: ptr[i8], file: ptr[i8], line: i32, expr: ptr[i8]) -> void:
        pass
elif IS_LINUX:
    @extern(lib="c")
    def __assert_fail(expr: ptr[i8], file: ptr[i8], line: u32, func: ptr[i8]) -> void:
        pass
else:
    raise NotImplementedError(
        "<assert.h> bindings are only available on macOS and Linux"
    )


__all__ = ["__assert_rtn" if IS_MACOS else "__assert_fail"]
