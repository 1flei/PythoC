"""
POSIX threads (<pthread.h>).

Only the subset used by SQLite's default Unix mutex implementation is
provided.  The opaque object sizes and the initializer signature are
platform-specific:

- macOS: pthread_mutex_t 64 bytes, mutexattr 16, cond 48; a statically
  initialized mutex carries the signature 0x32AAABA7 in its first word.
- glibc (x86_64 / aarch64): pthread_mutex_t 40 bytes, mutexattr 4,
  cond 48; an all-zero mutex is a valid static initializer.

Opaque storage uses i64 arrays so the aggregates are 8-byte aligned,
matching the alignment the C library expects for these objects.
"""

from ..decorators import compile, extern
from ..builtin_entities import i8, i32, i64, ptr, array, void
from ..forward_ref import mark_type_defined
from ._platform import IS_MACOS, IS_LINUX
from .sys_types import pthread_t


if IS_MACOS:
    @compile
    class pthread_mutex_t:
        """Opaque pthread mutex (macOS, 64 bytes)."""
        _opaque: array[i64, 8]

    @compile
    class pthread_mutexattr_t:
        """Opaque pthread mutex attributes (macOS, 16 bytes)."""
        _opaque: array[i64, 2]

    PTHREAD_MUTEX_RECURSIVE = 2
    # PTHREAD_MUTEX_INITIALIZER: __sig = 0x32AAABA7 followed by zeros.
    PTHREAD_MUTEX_INITIALIZER = pthread_mutex_t(
        _opaque=[0x32AAABA7, 0, 0, 0, 0, 0, 0, 0])
elif IS_LINUX:
    @compile
    class pthread_mutex_t:
        """Opaque pthread mutex (glibc, 40 bytes)."""
        _opaque: array[i64, 5]

    @compile
    class pthread_mutexattr_t:
        """Opaque pthread mutex attributes (glibc, 4 bytes)."""
        _opaque: i32

    PTHREAD_MUTEX_RECURSIVE = 1
    # glibc PTHREAD_MUTEX_INITIALIZER is all zeros.
    PTHREAD_MUTEX_INITIALIZER = pthread_mutex_t(_opaque=[0, 0, 0, 0, 0])
else:
    # Windows placeholders (no native pthreads).
    @compile
    class pthread_mutex_t:
        _opaque: array[i64, 8]

    @compile
    class pthread_mutexattr_t:
        _opaque: array[i64, 2]

    PTHREAD_MUTEX_RECURSIVE = 2
    PTHREAD_MUTEX_INITIALIZER = pthread_mutex_t(
        _opaque=[0, 0, 0, 0, 0, 0, 0, 0])


@compile
class pthread_cond_t:
    """Opaque pthread condition variable (macOS/glibc, 48 bytes)."""
    _opaque: array[i64, 6]


for _name in ("pthread_mutex_t", "pthread_mutexattr_t", "pthread_cond_t"):
    mark_type_defined(_name, globals()[_name])


@extern(lib="c")
def pthread_self() -> pthread_t:
    pass


@extern(lib="c")
def pthread_equal(t1: pthread_t, t2: pthread_t) -> i32:
    pass


@extern(lib="c")
def pthread_mutex_init(mutex: ptr[pthread_mutex_t], attr: ptr[pthread_mutexattr_t]) -> i32:
    pass


@extern(lib="c")
def pthread_mutex_lock(mutex: ptr[pthread_mutex_t]) -> i32:
    pass


@extern(lib="c")
def pthread_mutex_trylock(mutex: ptr[pthread_mutex_t]) -> i32:
    pass


@extern(lib="c")
def pthread_mutex_unlock(mutex: ptr[pthread_mutex_t]) -> i32:
    pass


@extern(lib="c")
def pthread_mutex_destroy(mutex: ptr[pthread_mutex_t]) -> i32:
    pass


@extern(lib="c")
def pthread_mutexattr_init(attr: ptr[pthread_mutexattr_t]) -> i32:
    pass


@extern(lib="c")
def pthread_mutexattr_settype(attr: ptr[pthread_mutexattr_t], type_: i32) -> i32:
    pass


@extern(lib="c")
def pthread_mutexattr_destroy(attr: ptr[pthread_mutexattr_t]) -> i32:
    pass


__all__ = [
    "pthread_t",
    "pthread_mutex_t", "pthread_mutexattr_t", "pthread_cond_t",
    "PTHREAD_MUTEX_INITIALIZER", "PTHREAD_MUTEX_RECURSIVE",
    "pthread_self", "pthread_equal",
    "pthread_mutex_init", "pthread_mutex_lock", "pthread_mutex_trylock",
    "pthread_mutex_unlock", "pthread_mutex_destroy",
    "pthread_mutexattr_init", "pthread_mutexattr_settype",
    "pthread_mutexattr_destroy",
]
