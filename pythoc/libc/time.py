"""
Time-related functions and types (time.h / sys/time.h).

These are POSIX rather than strict ISO C, but the preprocessor leaves the
underlying typedef names intact and they are referenced by real-world sources.
Layouts are platform-specific because ``time_t``, ``suseconds_t`` and the
``struct tm`` extension fields differ across operating systems.
"""

from ..decorators import compile, extern
from ..builtin_entities import ptr, i8, i32, i64, u64, void
from ._platform import IS_MACOS, IS_LINUX, IS_WINDOWS
from .sys_types import time_t


if IS_MACOS:
    @compile
    class timeval:
        """struct timeval from <sys/time.h>.  Layout matches macOS."""
        tv_sec: time_t
        tv_usec: i32

elif IS_LINUX:
    @compile
    class timeval:
        """struct timeval from <sys/time.h>.  Layout matches glibc."""
        tv_sec: time_t
        tv_usec: i64

else:
    # Windows MSVCRT: time_t is 64-bit, long is 32-bit on x64.
    @compile
    class timeval:
        """struct timeval from <sys/time.h>.  Layout matches MSVCRT."""
        tv_sec: time_t
        tv_usec: i32


if IS_WINDOWS:
    @compile
    class tm:
        """struct tm from <time.h>.  MSVCRT layout lacks BSD extension fields."""
        tm_sec: i32
        tm_min: i32
        tm_hour: i32
        tm_mday: i32
        tm_mon: i32
        tm_year: i32
        tm_wday: i32
        tm_yday: i32
        tm_isdst: i32
else:
    @compile
    class tm:
        """struct tm from <time.h>.  BSD/glibc layout with tm_gmtoff and tm_zone."""
        tm_sec: i32
        tm_min: i32
        tm_hour: i32
        tm_mday: i32
        tm_mon: i32
        tm_year: i32
        tm_wday: i32
        tm_yday: i32
        tm_isdst: i32
        tm_gmtoff: i64
        tm_zone: ptr[i8]


@extern(lib='c')
def gettimeofday(tv: ptr[timeval], tz: ptr[void]) -> i32:
    """Get current time of day."""
    pass


@extern(lib='c')
def utimes(path: ptr[i8], times: ptr[timeval]) -> i32:
    """Set file access and modification times."""
    pass


@extern(lib='c')
def time(t: ptr[i64]) -> i64:
    """Return the current calendar time."""
    pass


@extern(lib='c')
def localtime(t: ptr[time_t]) -> ptr[tm]:
    """Convert calendar time to broken-down local time."""
    pass


@extern(lib='c')
def strftime(s: ptr[i8], maxsize: u64, format: ptr[i8], tm: ptr[tm]) -> u64:
    """Format broken-down time into a string."""
    pass
