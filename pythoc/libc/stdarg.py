"""
Variable argument list support (stdarg.h).

`va_list` is an opaque pointer type on the x86_64 ABI used here.
"""

from ..builtin_entities import ptr, i8

va_list = ptr[i8]

__all__ = ['va_list']
