"""
Time-related functions and types (time.h / sys/time.h).

These are POSIX rather than strict ISO C, but the preprocessor leaves the
underlying typedef names intact and they are referenced by real-world sources.
"""

from ..decorators import extern
from ..builtin_entities import struct, ptr, i32, i64, void

# struct timeval is defined in <sys/time.h>.  Layout here matches macOS
# (time_t tv_sec, suseconds_t tv_usec); the size is the same on common
# 64-bit platforms even when suseconds_t is 32-bit.
timeval = struct["tv_sec": i64, "tv_usec": i32]


@extern(lib='c')
def gettimeofday(tv: ptr[timeval], tz: ptr[void]) -> i32:
    """Get current time of day."""
    pass


@extern(lib='c')
def time(t: ptr[i64]) -> i64:
    """Return the current calendar time."""
    pass
