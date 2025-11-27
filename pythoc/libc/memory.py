"""
Memory Management Utilities
Additional memory management functions and helpers
"""

from ..decorators import extern
from ..builtin_entities import ptr, i8, i32, i64

# Additional memory utilities
def memalloc(size: i64) -> ptr[i8]:
    """Convenient wrapper for malloc"""
    from .stdlib import malloc
    return malloc(size)

def memfree(ptr: ptr[i8]) -> None:
    """Convenient wrapper for free"""
    from .stdlib import free
    return free(ptr)

def memzero(ptr: ptr[i8], size: i64) -> ptr[i8]:
    """Zero out memory block"""
    from .string import memset
    return memset(ptr, 0, size)

# Memory debugging helpers (could be extended)
@extern(lib='c')
def memcheck(ptr: ptr[i8]) -> i32:
    """Check if memory is valid (placeholder)"""
    pass