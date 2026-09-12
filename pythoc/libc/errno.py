"""
Error numbers API (errno.h).

The `errno` macro expands to a thread-local slot accessed through a
per-platform function: glibc/musl use `__errno_location()`, macOS uses
`__error()`.  Both return a pointer to the current thread's errno value;
generated code reads and writes through `...[0]`.
"""

from ..decorators import extern
from ..builtin_entities import ptr, i32


__all__ = ['__error', '__errno_location']


@extern(lib='c')
def __error() -> ptr[i32]:
    """macOS: return a pointer to the current thread's errno value."""
    pass


@extern(lib='c')
def __errno_location() -> ptr[i32]:
    """glibc/musl: return a pointer to the current thread's errno value."""
    pass

