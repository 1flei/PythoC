"""
Wide Character String Functions (wchar.h)
"""

from ..decorators import extern
from ..builtin_entities import ptr, i32, i64
from .stddef import wchar_t, size_t
from ._platform import IS_WINDOWS


_c_long = i32 if IS_WINDOWS else i64


@extern(lib='c')
def wcscmp(s1: ptr[wchar_t], s2: ptr[wchar_t]) -> i32:
    """Compare two wide strings"""
    pass


@extern(lib='c')
def wcsncmp(s1: ptr[wchar_t], s2: ptr[wchar_t], n: size_t) -> i32:
    """Compare up to n wide characters of two wide strings"""
    pass


@extern(lib='c')
def wcslen(s: ptr[wchar_t]) -> size_t:
    """Get wide string length"""
    pass


@extern(lib='c')
def wmemchr(s: ptr[wchar_t], c: wchar_t, n: size_t) -> ptr[wchar_t]:
    """Locate a wide character in a wide buffer"""
    pass


@extern(lib='c')
def wmemcmp(s1: ptr[wchar_t], s2: ptr[wchar_t], n: size_t) -> i32:
    """Compare two wide buffers"""
    pass


@extern(lib='c')
def wcschr(s: ptr[wchar_t], c: wchar_t) -> ptr[wchar_t]:
    """Locate the first occurrence of a wide character in a wide string"""
    pass


@extern(lib='c')
def wcstol(nptr: ptr[wchar_t], endptr: ptr[ptr[wchar_t]], base: i32) -> _c_long:
    """Convert a wide string to a long"""
    pass


@extern(lib='c')
def wcstok(
    s: ptr[wchar_t], delim: ptr[wchar_t], ptr_: ptr[ptr[wchar_t]],
) -> ptr[wchar_t]:
    """Split a wide string into tokens"""
    pass
