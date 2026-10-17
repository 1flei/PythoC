"""
Locale API (locale.h)

setlocale() plus the LC_* category constants.  The constant values differ
between glibc and macOS; both sets are selected at import time.
"""

from ..decorators import extern
from ..builtin_entities import i32, ptr, i8
from ._platform import IS_MACOS, IS_LINUX

if IS_MACOS:
    LC_ALL = 0
    LC_COLLATE = 1
    LC_CTYPE = 2
    LC_MONETARY = 3
    LC_NUMERIC = 4
    LC_TIME = 5
    LC_MESSAGES = 6
elif IS_LINUX:
    LC_CTYPE = 0
    LC_NUMERIC = 1
    LC_TIME = 2
    LC_COLLATE = 3
    LC_MONETARY = 4
    LC_MESSAGES = 5
    LC_ALL = 6
else:
    raise NotImplementedError(
        "<locale.h> bindings are only available on macOS and Linux"
    )

__all__ = [
    'setlocale',
    'LC_ALL', 'LC_COLLATE', 'LC_CTYPE', 'LC_MONETARY', 'LC_NUMERIC',
    'LC_TIME', 'LC_MESSAGES',
]


@extern(lib='c')
def setlocale(category: i32, locale: ptr[i8]) -> ptr[i8]:
    """Set or query the current locale for a category."""
    pass
