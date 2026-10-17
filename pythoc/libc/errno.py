"""
Error numbers API (errno.h).

The `errno` macro expands to a thread-local slot accessed through a
per-platform function: glibc/musl use `__errno_location()`, macOS uses
`__error()`.  Both return a pointer to the current thread's errno value;
generated code reads and writes through `...[0]`.

Both handler names are exported on every supported platform: the
foreign-platform name is a small @compile shim forwarding to the native
one, so bindings generated on one platform also compile on the other.
"""

from ..decorators import compile, extern
from ..builtin_entities import ptr, i32
from ._platform import IS_MACOS, IS_LINUX


__all__ = ['__error', '__errno_location']


if IS_MACOS:
    @extern(lib='c')
    def __error() -> ptr[i32]:
        """macOS: return a pointer to the current thread's errno value."""
        pass

    @compile
    def __errno_location() -> ptr[i32]:
        """glibc/musl name, forwarded to __error() on macOS."""
        return __error()
elif IS_LINUX:
    @extern(lib='c')
    def __errno_location() -> ptr[i32]:
        """glibc/musl: return a pointer to the current thread's errno value."""
        pass

    @compile
    def __error() -> ptr[i32]:
        """macOS name, forwarded to __errno_location() on Linux."""
        return __errno_location()
else:
    raise NotImplementedError(
        "<errno.h> bindings are only available on macOS and Linux"
    )
