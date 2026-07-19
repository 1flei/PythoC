"""
User context type (ucontext.h / sys/ucontext.h).

Bindings are platform-specific because the layout of ucontext_t and its
nested machine-context structures varies across OS/architecture.  The
definitions below cover macOS (ARM64/x86_64) and glibc Linux (ARM64/x86_64);
other platforms fall back to an opaque type.
"""

from .. import compile
from ..builtin_entities import array, i8, i32, i64, ptr, u16, u32, u64, u8, void
from ..forward_ref import mark_type_defined
from ._platform import IS_MACOS, IS_LINUX, IS_WINDOWS, IS_X86_64, IS_ARM64


if IS_MACOS:
    # ------------------------------------------------------------------
    # macOS layout (verified against <sys/ucontext.h>)
    # The outer ucontext_t and sigaltstack layouts are the same on
    # x86_64 and ARM64; only the machine context differs.
    # ------------------------------------------------------------------

    @compile
    class _stack_t:
        ss_sp: ptr[void]
        ss_size: u64
        ss_flags: i32
        _padding: i32

    if IS_ARM64:
        @compile
        class _arm_thread_state64:
            __x: array[u64, 29]
            __fp: u64
            __lr: u64
            __sp: u64
            __pc: u64
            __cpsr: u32
            __pad: u32

        @compile
        class _arm_exception_state64:
            __far: u64
            __esr: u32
            __exception: i32

        @compile
        class _mcontext64:
            __es: _arm_exception_state64
            __ss: _arm_thread_state64
            # __ns (_STRUCT_ARM_NEON_STATE64) is omitted: it is not accessed
            # by translated code and its absence does not affect the offsets
            # of the fields that are accessed.  If code begins to touch NEON
            # state fields this struct must be extended.

    elif IS_X86_64:
        @compile
        class _x86_exception_state64:
            __trapno: u16
            __cpu: u16
            __err: u32
            __faultvaddr: u64

        @compile
        class _x86_thread_state64:
            __rax: u64
            __rbx: u64
            __rcx: u64
            __rdx: u64
            __rdi: u64
            __rsi: u64
            __rbp: u64
            __rsp: u64
            __r8: u64
            __r9: u64
            __r10: u64
            __r11: u64
            __r12: u64
            __r13: u64
            __r14: u64
            __r15: u64
            __rip: u64
            __rflags: u64
            __cs: u64
            __fs: u64
            __gs: u64

        @compile
        class _mcontext64:
            __es: _x86_exception_state64
            __ss: _x86_thread_state64
            # __fs (_STRUCT_X86_FLOAT_STATE64) is omitted for the same reason
            # as the NEON state on ARM64.  Extend when translated code
            # accesses floating-point state fields.

    else:
        _mcontext64 = i8

    @compile
    class ucontext_t:
        uc_onstack: i32
        uc_sigmask: u32
        uc_stack: _stack_t
        uc_link: ptr[void]
        uc_mcsize: u64
        uc_mcontext: ptr[_mcontext64]

elif IS_LINUX:
    # ------------------------------------------------------------------
    # glibc layout (sys/ucontext.h)
    # ------------------------------------------------------------------

    @compile
    class stack_t:
        ss_sp: ptr[void]
        ss_flags: i32
        _padding: i32
        ss_size: u64

    if IS_X86_64:
        # glibc x86_64 mcontext_t
        @compile
        class mcontext_t:
            gregs: array[i64, 23]
            fpregs: ptr[void]
            __reserved1: array[u64, 8]

        # glibc x86_64 ucontext_t (includes inline fpstate and __ssp)
        @compile
        class ucontext_t:
            uc_flags: u64
            uc_link: ptr[void]
            uc_stack: stack_t
            uc_mcontext: mcontext_t
            uc_sigmask: array[u64, 16]
            __fpregs_mem: array[u64, 64]
            __ssp: array[u64, 4]

    elif IS_ARM64:
        # glibc aarch64 mcontext_t
        @compile
        class mcontext_t:
            fault_address: u64
            regs: array[u64, 31]
            sp: u64
            pc: u64
            pstate: u64
            __reserved: array[u8, 4096]

        # glibc aarch64 ucontext_t (sigmask precedes mcontext)
        @compile
        class ucontext_t:
            uc_flags: u64
            uc_link: ptr[void]
            uc_stack: stack_t
            uc_sigmask: array[u64, 16]
            uc_mcontext: mcontext_t

    else:
        ucontext_t = i8

else:
    # Windows and unsupported platforms: keep the type defined so that
    # declarations compile, but field access will fail until a proper layout
    # is added.  Windows does not provide POSIX ucontext.
    ucontext_t = i8

mark_type_defined("ucontext_t", ucontext_t)

__all__ = ['ucontext_t']
