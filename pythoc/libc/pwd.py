"""
User account database (<pwd.h>).
"""

from ..decorators import compile, extern
from ..builtin_entities import i8, i32, u32, i64, ptr
from ..forward_ref import mark_type_defined
from ._platform import IS_MACOS


if IS_MACOS:
    @compile
    class passwd:
        pw_name: ptr[i8]
        pw_passwd: ptr[i8]
        pw_uid: u32
        pw_gid: u32
        pw_change: i64
        pw_class: ptr[i8]
        pw_gecos: ptr[i8]
        pw_dir: ptr[i8]
        pw_shell: ptr[i8]
        pw_expire: i64
else:
    # glibc layout.
    @compile
    class passwd:
        pw_name: ptr[i8]
        pw_passwd: ptr[i8]
        pw_uid: u32
        pw_gid: u32
        pw_gecos: ptr[i8]
        pw_dir: ptr[i8]
        pw_shell: ptr[i8]


@extern(lib='c')
def getpwuid(uid: u32) -> ptr[passwd]:
    """Look up a user by ID."""
    pass


mark_type_defined("passwd", passwd)

__all__ = ["passwd", "getpwuid"]
