"""
POSIX file control (<fcntl.h>).

PythoC already exposes ``open`` through ``unistd`` because it appears in both
headers; this module adds the file-control function itself and a few common
command constants.
"""

from ..decorators import compile, extern
from ..builtin_entities import i16, i32, ptr, void
from ..forward_ref import mark_type_defined
from ._platform import IS_MACOS, IS_WINDOWS
from .sys_types import off_t, pid_t


if not IS_WINDOWS:
    if IS_MACOS:
        @compile
        class flock:
            """struct flock from <fcntl.h> (macOS/BSD layout)."""
            l_start: off_t
            l_len: off_t
            l_pid: pid_t
            l_type: i16
            l_whence: i16
    else:
        @compile
        class flock:
            """struct flock from <fcntl.h> (glibc layout)."""
            l_type: i16
            l_whence: i16
            l_start: off_t
            l_len: off_t
            l_pid: pid_t
else:
    # Windows placeholder; MSVC has no fcntl file locking.
    @compile
    class flock:
        l_start: off_t
        l_len: off_t
        l_pid: pid_t
        l_type: i16
        l_whence: i16


mark_type_defined("flock", flock)


@extern(lib="c")
def fcntl(fd: i32, cmd: i32, *args) -> i32:
    """Manipulate a file descriptor."""
    pass


# Common command values.  These are object-like macros in the C header, so the
# preprocessor expands them before pcc sees them, but the constants are useful
# when writing PythoC code by hand.
F_DUPFD = 0
F_GETFD = 1
F_SETFD = 2
F_GETFL = 3
F_SETFL = 4
F_GETLK = 7
F_SETLK = 8
F_SETLKW = 9

FD_CLOEXEC = 1
F_RDLCK = 1
F_UNLCK = 2
F_WRLCK = 3

__all__ = [
    "flock", "fcntl",
    "F_DUPFD", "F_GETFD", "F_SETFD", "F_GETFL", "F_SETFL",
    "F_GETLK", "F_SETLK", "F_SETLKW",
    "FD_CLOEXEC", "F_RDLCK", "F_UNLCK", "F_WRLCK",
]
