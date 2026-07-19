"""
POSIX resource usage (<sys/resource.h>).

Provides ``struct rusage`` and ``getrusage`` for the performance-timer code
used by the SQLite shell.
"""

from ..decorators import compile, extern
from ..builtin_entities import i32, i64, ptr, array, void
from ..forward_ref import mark_type_defined
from ._platform import IS_WINDOWS
from .time import timeval


if not IS_WINDOWS:
    @compile
    class rusage:
        """struct rusage (macOS 64-bit / glibc common layout, 144 bytes).

        Both ABIs store every field after the two timevals as a long.
        getrusage() writes the whole struct, so the definition must cover
        the full 144 bytes even if only the leading fields are read.
        """
        ru_utime: timeval
        ru_stime: timeval
        ru_maxrss: i64
        ru_ixrss: i64
        ru_idrss: i64
        ru_isrss: i64
        ru_minflt: i64
        ru_majflt: i64
        ru_nswap: i64
        ru_inblock: i64
        ru_oublock: i64
        ru_msgsnd: i64
        ru_msgrcv: i64
        ru_nsignals: i64
        ru_nvcsw: i64
        ru_nivcsw: i64
else:
    # Windows has no getrusage; opaque placeholder of matching size.
    @compile
    class rusage:
        _opaque: array[i64, 18]


mark_type_defined("rusage", rusage)

RUSAGE_SELF = 0
RUSAGE_CHILDREN = -1


@extern(lib="c")
def getrusage(who: i32, usage: ptr[rusage]) -> i32:
    """Get resource usage."""
    pass


__all__ = [
    "rusage", "RUSAGE_SELF", "RUSAGE_CHILDREN", "getrusage",
]
