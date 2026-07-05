"""
POSIX operating-system API (unistd.h).

These declarations are not part of the C standard library but are required by
many portable C sources (e.g. sysconf, getpid, read, write).
"""

from ..decorators import extern
from ..builtin_entities import ptr, i8, i32, i64, u64, void


@extern(lib='c')
def sysconf(name: i32) -> i64:
    """Get configurable system variables."""
    pass


@extern(lib='c')
def getpid() -> i32:
    """Return the process ID."""
    pass


@extern(lib='c')
def getcwd(buf: ptr[i8], size: i64) -> ptr[i8]:
    """Get current working directory."""
    pass


@extern(lib='c')
def close(fd: i32) -> i32:
    """Close a file descriptor."""
    pass


@extern(lib='c')
def read(fd: i32, buf: ptr[void], count: i64) -> i64:
    """Read from a file descriptor."""
    pass


@extern(lib='c')
def write(fd: i32, buf: ptr[void], count: i64) -> i64:
    """Write to a file descriptor."""
    pass


@extern(lib='c')
def lseek(fd: i32, offset: i64, whence: i32) -> i64:
    """Reposition read/write file offset."""
    pass


@extern(lib='c')
def access(path: ptr[i8], mode: i32) -> i32:
    """Check user's permissions for a file."""
    pass


@extern(lib='c')
def isatty(fd: i32) -> i32:
    """Test whether a file descriptor refers to a terminal."""
    pass


@extern(lib='c')
def getpagesize() -> i32:
    """Return the underlying hardware page size."""
    pass


@extern(lib='c')
def mprotect(addr: ptr[void], len: u64, prot: i32) -> i32:
    """Set protection on a region of memory."""
    pass
