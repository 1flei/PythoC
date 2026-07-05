"""
Non-local jumps API (setjmp.h).

`jmp_buf` is an opaque array type.  The size below matches the macOS
x86_64 definition; it is embedded by value in translated structs, so the
size must be correct for the target ABI.
"""

from ..decorators import extern
from ..builtin_entities import array, i32, void, ptr


# macOS x86_64: _JBLEN == (9 * 2) + 3 + 16 == 37 ints
jmp_buf = array[i32, 37]
sigjmp_buf = array[i32, 38]


__all__ = ['jmp_buf', 'sigjmp_buf', 'setjmp', 'longjmp']


@extern(lib='c')
def setjmp(env: ptr[void]) -> i32:
    """Save calling environment for a non-local jump."""
    pass


@extern(lib='c')
def longjmp(env: ptr[void], val: i32) -> void:
    """Perform a non-local jump to a saved environment."""
    pass
