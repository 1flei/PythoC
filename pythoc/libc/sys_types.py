"""
POSIX system types (<sys/types.h>).

These typedefs are introduced by system headers that the C preprocessor
expands.  The concrete widths are platform-specific; this module follows
the macOS / glibc 64-bit layouts that PythoC targets.
"""

from ..builtin_entities import i32, i64, u16, u32, u64, ptr, void
from ..forward_ref import mark_type_defined
from ._platform import IS_MACOS, IS_LINUX, IS_WINDOWS, IS_ARM64


# time_t is a 64-bit signed integer on all 64-bit platforms.
time_t = i64


if IS_MACOS:
    dev_t = i32
    ino_t = u64
    mode_t = u16
    nlink_t = u16
    uid_t = u32
    gid_t = u32
    off_t = i64
    blkcnt_t = i64
    blksize_t = i32
    pid_t = i32
    pthread_t = ptr[void]
elif IS_LINUX and IS_ARM64:
    # glibc aarch64 (asm-generic): narrower nlink_t / blksize_t than x86_64.
    dev_t = u64
    ino_t = u64
    mode_t = u32
    nlink_t = u32
    uid_t = u32
    gid_t = u32
    off_t = i64
    blkcnt_t = i64
    blksize_t = i32
    pid_t = i32
    pthread_t = u64
elif IS_LINUX:
    # glibc x86_64
    dev_t = u64
    ino_t = u64
    mode_t = u32
    nlink_t = u64
    uid_t = u32
    gid_t = u32
    off_t = i64
    blkcnt_t = i64
    blksize_t = i64
    pid_t = i32
    pthread_t = u64
else:
    # Windows placeholders (not used by the POSIX code paths).
    dev_t = i32
    ino_t = u64
    mode_t = u16
    nlink_t = u16
    uid_t = u32
    gid_t = u32
    off_t = i64
    blkcnt_t = i64
    blksize_t = i32
    pid_t = i32
    pthread_t = ptr[void]


for _name in (
    "dev_t", "ino_t", "mode_t", "nlink_t", "uid_t", "gid_t",
    "off_t", "blkcnt_t", "blksize_t", "pid_t", "pthread_t", "time_t",
):
    mark_type_defined(_name, globals()[_name])


__all__ = [
    "dev_t", "ino_t", "mode_t", "nlink_t", "uid_t", "gid_t",
    "off_t", "blkcnt_t", "blksize_t", "pid_t", "pthread_t", "time_t",
]
