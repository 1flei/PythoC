"""
POSIX character set conversion API (iconv.h).

glibc and libSystem expose iconv in libc / libiconv respectively; the
darwin case needs an explicit lib name because libSystem.tbd does not
re-export the symbols.  ``iconv_t`` is an opaque handle (``void *`` on
both glibc and darwin).
"""

from ..decorators import extern
from ..builtin_entities import ptr, i8, i32, void
from ._platform import IS_LINUX, IS_MACOS
from .stddef import size_t

if not (IS_MACOS or IS_LINUX):
    raise NotImplementedError(
        '<iconv.h> bindings are only available on macOS and Linux'
    )

# glibc has iconv in libc; on Darwin libSystem.tbd does not re-export it,
# so link libiconv explicitly.
_iconv_lib = 'iconv' if IS_MACOS else 'c'

# Opaque conversion descriptor (``typedef void *iconv_t;``).
iconv_t = ptr[void]


@extern(lib=_iconv_lib)
def iconv_open(tocode: ptr[i8], fromcode: ptr[i8]) -> iconv_t:
    """Allocate a conversion descriptor for tocode/fromcode."""
    pass


@extern(lib=_iconv_lib)
def iconv(cd: iconv_t, inbuf: ptr[ptr[i8]], inbytesleft: ptr[size_t],
          outbuf: ptr[ptr[i8]], outbytesleft: ptr[size_t]) -> size_t:
    """Convert characters, updating cursors and remaining byte counts."""
    pass


@extern(lib=_iconv_lib)
def iconv_close(cd: iconv_t) -> i32:
    """Free a conversion descriptor."""
    pass


__all__ = ["iconv_t", "iconv", "iconv_open", "iconv_close"]
