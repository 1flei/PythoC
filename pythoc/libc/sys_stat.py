"""
POSIX file status (<sys/stat.h>).

Provides the ``stat`` / ``timespec`` aggregates and the common status
functions.  Layouts are platform- and architecture-specific.

Note: C keeps functions and struct tags in separate namespaces, so
``stat()`` and ``struct stat`` coexist there.  PythoC has a single Python
namespace; here the struct keeps the plain name ``stat`` and the function
is exported as ``stat_`` (bound to the C symbol ``stat``), following the
``raise_`` / ``sigaction_`` convention in ``signal.py``.
"""

from ..decorators import compile, extern
from ..builtin_entities import ptr, i8, i32, i64, u16, u32, u64, array, void
from ..forward_ref import mark_type_defined
from ._platform import IS_MACOS, IS_LINUX, IS_WINDOWS, IS_ARM64
from .sys_types import (
    dev_t, ino_t, mode_t, nlink_t, uid_t, gid_t,
    off_t, blkcnt_t, blksize_t, time_t,
)


if IS_MACOS:
    @compile
    class timespec:
        """struct timespec from <sys/_types/_timespec.h>."""
        tv_sec: time_t
        tv_nsec: i64

    @compile
    class stat:
        """struct stat from <sys/stat.h> (macOS 64-bit inode layout)."""
        st_dev: dev_t
        st_mode: mode_t
        st_nlink: nlink_t
        st_ino: ino_t
        st_uid: uid_t
        st_gid: gid_t
        st_rdev: dev_t
        st_atimespec: timespec
        st_mtimespec: timespec
        st_ctimespec: timespec
        st_birthtimespec: timespec
        st_size: off_t
        st_blocks: blkcnt_t
        st_blksize: blksize_t
        st_flags: u32
        st_gen: u32
        st_lspare: i32
        st_qspare: array[i64, 2]

elif IS_LINUX and not IS_ARM64:
    @compile
    class timespec:
        """struct timespec from <time.h> (glibc)."""
        tv_sec: time_t
        tv_nsec: i64

    @compile
    class stat:
        """struct stat from <sys/stat.h> (glibc x86_64 layout)."""
        st_dev: dev_t
        st_ino: ino_t
        st_nlink: nlink_t
        st_mode: mode_t
        st_uid: uid_t
        st_gid: gid_t
        __pad0: i32
        st_rdev: dev_t
        st_size: off_t
        st_blksize: blksize_t
        st_blocks: blkcnt_t
        st_atime: time_t
        st_atimensec: i64
        st_mtime: time_t
        st_mtimensec: i64
        st_ctime: time_t
        st_ctimensec: i64
        __unused: array[i64, 3]

elif IS_LINUX and IS_ARM64:
    @compile
    class timespec:
        """struct timespec from <time.h> (glibc)."""
        tv_sec: time_t
        tv_nsec: i64

    @compile
    class stat:
        """struct stat from <sys/stat.h> (glibc aarch64, asm-generic layout)."""
        st_dev: dev_t
        st_ino: ino_t
        st_mode: mode_t
        st_nlink: nlink_t
        st_uid: uid_t
        st_gid: gid_t
        st_rdev: dev_t
        __pad1: u64
        st_size: off_t
        st_blksize: blksize_t
        __pad2: i32
        st_blocks: blkcnt_t
        st_atime: time_t
        st_atimensec: i64
        st_mtime: time_t
        st_mtimensec: i64
        st_ctime: time_t
        st_ctimensec: i64
        __unused: array[i32, 2]

else:
    # Windows placeholder; MSVC provides _stat32/_stat64 instead.
    @compile
    class timespec:
        tv_sec: time_t
        tv_nsec: i64

    @compile
    class stat:
        st_dev: i32
        st_ino: u64
        st_mode: u16
        st_nlink: u16
        st_uid: u32
        st_gid: u32
        st_rdev: i32
        st_size: i64
        st_atime: time_t
        st_mtime: time_t
        st_ctime: time_t


mark_type_defined("timespec", timespec)
mark_type_defined("stat", stat)


@extern(lib="c", name="stat")
def stat_(path: ptr[i8], buf: ptr[stat]) -> i32:
    """Get file status (C symbol ``stat``)."""
    pass


@extern(lib="c")
def lstat(path: ptr[i8], buf: ptr[stat]) -> i32:
    """Get file status, do not follow symlinks."""
    pass


@extern(lib="c")
def fstat(fd: i32, buf: ptr[stat]) -> i32:
    """Get file status from an open file descriptor."""
    pass


@extern(lib="c")
def chmod(path: ptr[i8], mode: mode_t) -> i32:
    """Change file mode bits."""
    pass


@extern(lib="c")
def mkdir(path: ptr[i8], mode: mode_t) -> i32:
    """Create a directory."""
    pass


__all__ = [
    "timespec", "stat", "stat_", "lstat", "fstat", "chmod", "mkdir",
]
