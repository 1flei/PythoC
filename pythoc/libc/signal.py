"""
Signal handling API (signal.h).

Only the symbols referenced by translated C code are exposed.  Complex structs
are kept opaque until field access is required.  Layouts are platform-specific
because signal structures differ across OS and architecture.
"""

from .. import compile
from ..decorators import extern
from ..builtin_entities import array, func, i16, i32, i64, i8, ptr, u32, u64, union, void
from ..forward_ref import mark_type_defined
from ._platform import IS_MACOS, IS_LINUX, IS_WINDOWS, IS_X86_64, IS_ARM64


if IS_MACOS:
    # macOS siginfo_t layout (struct __siginfo from <sys/signal.h>).
    # The layout is the same on x86_64 and ARM64.
    @compile
    class siginfo_t:
        si_signo: i32
        si_errno: i32
        si_code: i32
        si_pid: i32
        si_uid: u32
        si_status: i32
        _pad_after_status: i32
        si_addr: ptr[void]
        si_value_int: i32
        si_value_ptr: ptr[void]
        si_band: i64
        __pad: array[u64, 7]

    # __darwin_sigset_t is a 32-bit mask on macOS
    sigset_t = u32

    # struct sigaction layout from <sys/signal.h>.  The C ``sigaction``
    # function is renamed to ``sigaction_`` in PythoC because the struct and
    # the function share the same name.
    @compile
    class sigaction:
        __sigaction_u: union[
            "__sa_handler": func[i32, void],
            "__sa_sigaction": func[i32, ptr[siginfo_t], ptr[void], void],
        ]
        sa_mask: sigset_t
        sa_flags: i32

elif IS_LINUX:
    # glibc siginfo_t layout (bits/types/siginfo_t.h).  The layout is the
    # same on x86_64 and ARM64 because both use the asm-generic definition.
    # Total size is 128 bytes.

    @compile
    class _siginfo_kill:
        si_pid: i32
        si_uid: u32

    @compile
    class _siginfo_sigfault:
        si_addr: ptr[void]
        si_addr_lsb: i16
        __pad: array[i8, 6]

    _sifields = union[
        "_pad": array[i32, 28],
        "_kill": _siginfo_kill,
        "_sigfault": _siginfo_sigfault,
    ]

    @compile
    class siginfo_t:
        si_signo: i32
        si_errno: i32
        si_code: i32
        __pad0: i32
        _sifields: _sifields

    # glibc sigset_t is unsigned long int __val[16] (128 bytes).
    sigset_t = array[u64, 16]

    # glibc struct sigaction (bits/sigaction.h).  The C ``sigaction`` function
    # is renamed to ``sigaction_`` because the struct and function share the
    # same name.
    @compile
    class sigaction:
        __sigaction_handler: union[
            "sa_handler": func[i32, void],
            "sa_sigaction": func[i32, ptr[siginfo_t], ptr[void], void],
        ]
        sa_mask: sigset_t
        sa_flags: i32
        sa_restorer: func[void, void]

else:
    # Windows and unsupported platforms: keep the types defined so that
    # declarations compile, but field access will fail until a proper layout
    # is added.  Windows does not provide POSIX signal structures.
    siginfo_t = i8
    sigset_t = i8
    sigaction = i8

mark_type_defined("siginfo_t", siginfo_t)
mark_type_defined("sigset_t", sigset_t)
mark_type_defined("sigaction", sigaction)

__all__ = [
    'siginfo_t',
    'sigset_t',
    'sigaction',
    'signal',
    'raise_',
    'sigaction_',
    'sigemptyset',
    'sigfillset',
    'sigprocmask',
]


@extern(lib='c')
def signal(signum: i32, handler: ptr[void]) -> ptr[void]:
    """Install a signal handler."""
    pass


@extern(lib='c', name='raise')
def raise_(signum: i32) -> i32:
    """Send a signal to the current process."""
    pass


@extern(lib='c', name='sigaction')
def sigaction_(signum: i32, act: ptr[void], oldact: ptr[void]) -> i32:
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


@extern(lib='c')
def sigprocmask(how: i32, set_: ptr[void], oldset: ptr[void]) -> i32:
    """Examine and change blocked signals."""
    pass
