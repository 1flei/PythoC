"""
POSIX memory-mapped files (<sys/mman.h>).
"""

from ..decorators import extern
from ..builtin_entities import ptr, i32, i64, void


PROT_NONE = 0
PROT_READ = 1
PROT_WRITE = 2
PROT_EXEC = 4

MAP_SHARED = 1
MAP_PRIVATE = 2
MAP_FIXED = 16
MAP_FAILED = -1


@extern(lib="c")
def mmap(addr: ptr[void], length: i64, prot: i32, flags: i32, fd: i32, offset: i64) -> ptr[void]:
    """Map files or devices into memory."""
    pass


@extern(lib="c")
def munmap(addr: ptr[void], length: i64) -> i32:
    """Unmap a memory region."""
    pass


__all__ = [
    "mmap", "munmap",
    "PROT_NONE", "PROT_READ", "PROT_WRITE", "PROT_EXEC",
    "MAP_SHARED", "MAP_PRIVATE", "MAP_FIXED", "MAP_FAILED",
]
