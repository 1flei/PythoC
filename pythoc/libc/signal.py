"""
Signal handling API (signal.h).

Only the symbols referenced by translated C code are exposed.  Complex structs
are kept opaque until field access is required.
"""

from ..decorators import extern
from ..builtin_entities import i8, i32, i64, ptr, func, void
from ..forward_ref import mark_type_defined

siginfo_t = i8
mark_type_defined("siginfo_t", siginfo_t)

__all__ = [
    'siginfo_t',
    'signal',
    'raise_',
    'sigaction',
    'sigemptyset',
    'sigfillset',
]


@extern(lib='c')
def signal(signum: i32, handler: ptr[void]) -> ptr[void]:
    """Install a signal handler."""
    pass


@extern(lib='c')
def raise_(signum: i32) -> i32:
    """Send a signal to the current process."""
    pass


@extern(lib='c')
def sigaction(signum: i32, act: ptr[void], oldact: ptr[void]) -> i32:
    """Examine/change a signal action."""
    pass


@extern(lib='c')
def sigemptyset(set_: ptr[void]) -> i32:
    """Initialize an empty signal set."""
    pass


@extern(lib='c')
def sigfillset(set_: ptr[void]) -> i32:
    """Initialize a full signal set."""
    pass
