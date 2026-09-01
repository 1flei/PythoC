"""
POSIX system types (<sys/types.h>).

These typedefs are introduced by system headers that the C preprocessor
expands.  The concrete widths are platform-specific; this module follows
the macOS / glibc 64-bit layouts that PythoC targets.
"""

from ..builtin_entities import i8, i16, i32, i64, u8, u16, u32, u64, ptr, void
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
    # macOS sys/types.h fixed-width aliases.
    __int8_t = i8
    __uint8_t = u8
    __int16_t = i16
    __uint16_t = u16
    __int32_t = i32
    __uint32_t = u32
    __int64_t = i64
    __uint64_t = u64
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


if IS_LINUX:
    # Kernel fixed-width/byte-order typedefs (asm-generic/int-ll64.h),
    # reachable through <sys/types.h> and headers such as bits/statx.h.
    # The byte-order aliases assume little-endian, which holds for the
    # x86_64 / aarch64 targets PythoC supports.
    __s8 = i8
    __u8 = u8
    __s16 = i16
    __u16 = u16
    __s32 = i32
    __u32 = u32
    __s64 = i64
    __u64 = u64
    __le16 = __u16
    __be16 = __u16
    __le32 = __u32
    __be32 = __u32
    __le64 = __u64
    __be64 = __u64
    __sum16 = __u16
    __wsum = __u32

_KERNEL_TYPEDEFS = [
    "__s8", "__u8", "__s16", "__u16", "__s32", "__u32", "__s64", "__u64",
    "__le16", "__be16", "__le32", "__be32", "__le64", "__be64",
    "__sum16", "__wsum",
]


for _name in (
    "dev_t", "ino_t", "mode_t", "nlink_t", "uid_t", "gid_t",
    "off_t", "blkcnt_t", "blksize_t", "pid_t", "pthread_t", "time_t",
    "__int8_t", "__uint8_t", "__int16_t", "__uint16_t",
    "__int32_t", "__uint32_t", "__int64_t", "__uint64_t",
    *_KERNEL_TYPEDEFS,
):
    if _name in globals():
        mark_type_defined(_name, globals()[_name])


__all__ = [
    "dev_t", "ino_t", "mode_t", "nlink_t", "uid_t", "gid_t",
    "off_t", "blkcnt_t", "blksize_t", "pid_t", "pthread_t", "time_t",
    "__int8_t", "__uint8_t", "__int16_t", "__uint16_t",
    "__int32_t", "__uint32_t", "__int64_t", "__uint64_t",
]
if IS_LINUX:
    __all__ = __all__ + _KERNEL_TYPEDEFS
