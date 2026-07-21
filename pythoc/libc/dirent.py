"""
Directory entry functions and types (<dirent.h>).

``DIR`` is opaque by standard; ``struct dirent`` layout is
platform-specific (macOS / glibc 64-bit).  <dirent.h> does not exist on
Windows (MSVC/UCRT), so this module refuses to import there.
"""

from ..decorators import compile, extern
from ..builtin_entities import i8, i32, i64, u8, u16, u64, ptr, array
from ..forward_ref import mark_type_defined
from ._platform import IS_MACOS, IS_LINUX


if IS_MACOS:
    @compile
    class dirent:
        """struct dirent from <dirent.h>.  Layout matches macOS."""
        d_ino: u64
        d_seekoff: u64
        d_reclen: u16
        d_namlen: u16
        d_type: u8
        d_name: array[i8, 1024]
elif IS_LINUX:
    @compile
    class dirent:
        """struct dirent from <dirent.h>.  Layout matches glibc; d_name is last."""
        d_ino: u64
        d_off: i64
        d_reclen: u16
        d_type: u8
        d_name: array[i8, 256]
else:
    raise NotImplementedError(
        "<dirent.h> bindings are only available on macOS and Linux"
    )


# DIR is an opaque directory stream type (same idiom as stdio's FILE):
# exposing it as a named alias lets code use ``ptr[DIR]`` while keeping the
# actual representation an opaque byte pointer.
DIR = i8


@extern(lib="c")
def opendir(name: ptr[i8]) -> ptr[DIR]:
    """Open a directory stream."""
    pass


@extern(lib="c")
def readdir(dirp: ptr[DIR]) -> ptr[dirent]:
    """Read the next directory entry."""
    pass


@extern(lib="c")
def closedir(dirp: ptr[DIR]) -> i32:
    """Close a directory stream."""
    pass


for _name in ("dirent", "DIR"):
    mark_type_defined(_name, globals()[_name])


__all__ = ["dirent", "DIR", "opendir", "readdir", "closedir"]
