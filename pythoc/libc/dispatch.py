"""
Apple dispatch (GCD) semaphore API.

Only the handful of symbols used by translated C code are exposed here.
"""

from ..decorators import extern
from ..builtin_entities import i64, u64, ptr, i8

dispatch_semaphore_t = ptr[i8]
dispatch_time_t = u64

__all__ = [
    'dispatch_semaphore_t',
    'dispatch_time_t',
    'dispatch_semaphore_create',
    'dispatch_semaphore_wait',
    'dispatch_semaphore_signal',
]


@extern(lib='c')
def dispatch_semaphore_create(value: i64) -> dispatch_semaphore_t:
    """Create a new counting semaphore."""
    pass


@extern(lib='c')
def dispatch_semaphore_wait(dsema: dispatch_semaphore_t, timeout: dispatch_time_t) -> i64:
    """Wait for a signal on the semaphore."""
    pass


@extern(lib='c')
def dispatch_semaphore_signal(dsema: dispatch_semaphore_t) -> i64:
    """Signal the semaphore."""
    pass
