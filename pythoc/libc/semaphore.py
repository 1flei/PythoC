"""
POSIX semaphore API (semaphore.h).

`sem_t` is treated as an opaque buffer; its layout is ABI-specific.
"""

from ..decorators import extern
from ..builtin_entities import array, i32, i64, u8, ptr

# Opaque POSIX semaphore object.  32 bytes is large enough for the common ABIs
# used by this toolchain; it is embedded by value in translated structs.
sem_t = array[u8, 32]

__all__ = [
    'sem_t',
    'sem_init',
    'sem_wait',
    'sem_post',
    'sem_destroy',
]


@extern(lib='c')
def sem_init(sem: ptr[sem_t], pshared: i32, value: u8) -> i32:
    """Initialize an unnamed semaphore."""
    pass


@extern(lib='c')
def sem_wait(sem: ptr[sem_t]) -> i32:
    """Lock a semaphore."""
    pass


@extern(lib='c')
def sem_post(sem: ptr[sem_t]) -> i32:
    """Unlock a semaphore."""
    pass


@extern(lib='c')
def sem_destroy(sem: ptr[sem_t]) -> i32:
    """Destroy an unnamed semaphore."""
    pass
