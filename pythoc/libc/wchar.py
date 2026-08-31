"""
Wide Character String Functions (wchar.h)
"""

from ..decorators import extern
from ..builtin_entities import ptr, i32
from .stddef import wchar_t, size_t


@extern(lib='c')
def wcscmp(s1: ptr[wchar_t], s2: ptr[wchar_t]) -> i32:
    """Compare two wide strings"""
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
