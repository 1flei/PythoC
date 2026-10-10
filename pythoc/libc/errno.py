"""
Error numbers API (errno.h).

The `errno` macro expands to a thread-local slot accessed through a
per-platform function: glibc/musl use `__errno_location()`, macOS uses
`__error()`, Windows uses `_errno()`. All return a pointer to the current
thread's errno value; generated code reads and writes through `...[0]`.
"""

import sys

from ..decorators import extern
from ..builtin_entities import ptr, i32


__all__ = ['__error', '__errno_location', '_errno', 'errno_slot']


@extern(lib='c')
def __error() -> ptr[i32]:
    """macOS: return a pointer to the current thread's errno value."""
    pass


@extern(lib='c')
def __errno_location() -> ptr[i32]:
    """glibc/musl: return a pointer to the current thread's errno value."""
    pass


@extern(lib='c')
def _errno() -> ptr[i32]:
    """Windows: return a pointer to the current thread's errno value."""
    pass


if sys.platform == 'darwin':
    errno_slot = __error
elif sys.platform == 'win32':
    errno_slot = _errno
else:
    errno_slot = __errno_location

