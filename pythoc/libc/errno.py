"""
Error numbers API (errno.h).

On macOS the `errno` macro expands to `(*__error())`; expose the underlying
accessor so generated code can call `__error()[0]`.
"""

from ..decorators import extern
from ..builtin_entities import ptr, i32


__all__ = ['__error']


@extern(lib='c')
def __error() -> ptr[i32]:
    """Return a pointer to the current thread's errno value."""
    pass
