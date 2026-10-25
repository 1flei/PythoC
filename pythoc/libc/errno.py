"""Thread-local errno access through the native C runtime.

The POSIX spellings remain available through a compiled forwarding shim
on macOS and Linux. errno_slot selects the platform's native accessor.
"""

from ..decorators import compile, extern
from ..builtin_entities import ptr, i32
from ._platform import IS_MACOS, IS_LINUX, IS_WINDOWS


__all__ = ['__error', '__errno_location', '_errno', 'errno_slot']


if IS_MACOS:
    @extern(lib='c')
    def __error() -> ptr[i32]:
        """macOS: return a pointer to the current thread's errno value."""
        pass

    @compile
    def __errno_location() -> ptr[i32]:
        """glibc/musl name, forwarded to __error() on macOS."""
        return __error()

    errno_slot = __error
    _errno = errno_slot
elif IS_LINUX:
    @extern(lib='c')
    def __errno_location() -> ptr[i32]:
        """glibc/musl: return a pointer to the current thread's errno value."""
        pass

    @compile
    def __error() -> ptr[i32]:
        """macOS name, forwarded to __errno_location() on Linux."""
        return __errno_location()

    errno_slot = __errno_location
    _errno = errno_slot
elif IS_WINDOWS:
    @extern(lib='c')
    def _errno() -> ptr[i32]:
        """Windows: return a pointer to the current thread's errno value."""
        pass

    errno_slot = _errno
    __error = errno_slot
    __errno_location = errno_slot
else:
    raise NotImplementedError(
        "<errno.h> bindings are only available on macOS, Linux and Windows"
    )
