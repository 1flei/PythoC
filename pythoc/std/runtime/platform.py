"""
Platform Abstractions: Cross-platform primitives for the N:M runtime.

Supports: Linux / macOS / Windows x x86_64 / AArch64

Strategy:
    Context switch:  Hand-written assembly per (OS, arch) pair.
                     Only saves callee-saved registers + stack pointer.
                     Much faster than ucontext (no signal mask save/restore).

    Threading:       Thin abstraction over pthread (Linux/macOS) and
                     Win32 threads (Windows).

    Atomics:         LLVM IR atomics (PythoC compiles to LLVM, which has
                     target-independent atomic instructions).

    Notification:    eventfd (Linux), pipe (macOS), Event (Windows).

Design:
    The platform layer is split into:
    1. Portable types + constants (this file, always compiled)
    2. Platform-specific implementations selected at compile-time
       via PythoC's Python-level metaprogramming (if/else at import time)

    Context switch uses a minimal "MiniCtx" struct: just the saved
    registers needed for a cooperative (voluntary) switch.  This is
    6 registers on x86_64, 12 on AArch64.  Total struct size: 56-104 bytes.
    Compare to ucontext_t's 936 bytes.
"""
from __future__ import annotations

from .policy import bind_mem
bind_mem()

import sys
import os
import platform as _platform_mod

from pythoc import (
    compile, effect, extern, i32, i64, u32, u64, u8, ptr, void, struct, func, array,
    nullptr, sizeof,
)
from pythoc.builtin_entities import (
    atomic_load_i64 as _atomic_load_i64,
    atomic_store_i64 as _atomic_store_i64,
    atomic_fetch_add_i64 as _atomic_fetch_add_i64,
    atomic_cas_i64 as _atomic_cas_i64,
    atomic_load_i32 as _atomic_load_i32,
    atomic_store_i32 as _atomic_store_i32,
    llvm_asm,
)
from pythoc.libc.string import memset


# ============================================================
# Compile-time platform detection (Python level = compile time)
# ============================================================

_os = sys.platform        # 'linux', 'darwin', 'win32'
_arch = _platform_mod.machine()  # 'x86_64', 'aarch64', 'arm64', 'AMD64'

# Normalize
if _arch in ('arm64', 'ARM64'):
    _arch = 'aarch64'
if _arch in ('AMD64', 'x86_64'):
    _arch = 'x86_64'

IS_LINUX   = (_os == 'linux')
IS_MACOS   = (_os == 'darwin')
IS_WINDOWS = (_os == 'win32')
IS_X86_64  = (_arch == 'x86_64')
IS_AARCH64 = (_arch == 'aarch64')

if IS_X86_64:
    _CTX_ASM_NAME = 'ctx_x86_64_win.S' if IS_WINDOWS else 'ctx_x86_64_sysv.S'
else:
    _CTX_ASM_NAME = 'ctx_aarch64.S'

_CTX_ASM_PATH = os.path.join(os.path.dirname(__file__), 'asm', _CTX_ASM_NAME)


# ============================================================
# Atomics: LLVM-native atomic operations
#
# PythoC compiles to LLVM IR, which has target-independent atomics:
#   atomicrmw, cmpxchg, fence, load atomic, store atomic
#
# We emit LLVM atomic instructions directly via @compile.
# These work on ALL targets LLVM supports. No platform #ifdefs needed.
# ============================================================

ATOMIC_RELAXED = i32(0)
ATOMIC_CONSUME = i32(1)
ATOMIC_ACQUIRE = i32(2)
ATOMIC_RELEASE = i32(3)
ATOMIC_ACQ_REL = i32(4)
ATOMIC_SEQ_CST = i32(5)


@compile
def atomic_load_i64(p: ptr[i64]) -> i64:
    return _atomic_load_i64(p)


@compile
def atomic_store_i64(p: ptr[i64], val: i64) -> void:
    _atomic_store_i64(p, val)


@compile
def atomic_fetch_add_i64(p: ptr[i64], val: i64) -> i64:
    return _atomic_fetch_add_i64(p, val)


@compile
def atomic_cas_i64(p: ptr[i64], expected: ptr[i64], desired: i64) -> i32:
    return _atomic_cas_i64(p, expected, desired)


@compile
def atomic_load_i32(p: ptr[i32]) -> i32:
    return _atomic_load_i32(p)


@compile
def atomic_store_i32(p: ptr[i32], val: i32) -> void:
    _atomic_store_i32(p, val)


# ============================================================
# Spin hint
#
# This is a performance hint only. The atomic operations carry correctness.
# ============================================================

if IS_X86_64:
    @compile
    def spin_hint() -> void:
        llvm_asm("pause", "~{memory}")

elif IS_AARCH64:
    @compile
    def spin_hint() -> void:
        llvm_asm("yield", "~{memory}")

else:
    @compile
    def spin_hint() -> void:
        pass


# ============================================================
# Spinlock (portable: uses LLVM atomics + spin hint)
# ============================================================

@compile
class SpinLock:
    state: i64   # 0 = unlocked, 1 = locked


@compile
def spinlock_init(lock: ptr[SpinLock]) -> void:
    lock.state = i64(0)


@compile
def spinlock_lock(lock: ptr[SpinLock]) -> void:
    """Acquire spinlock (CAS loop with architecture-specific backoff)."""
    expected: i64 = 0
    while True:
        expected = i64(0)
        if atomic_cas_i64(
            ptr[i64](ptr[void](lock)),
            ptr[i64](ptr[void](ptr(expected))),
            i64(1)
        ) != 0:
            return
        spin_hint()  # PAUSE on x86, YIELD on ARM


@compile
def spinlock_unlock(lock: ptr[SpinLock]) -> void:
    """Release spinlock."""
    atomic_store_i64(ptr[i64](ptr[void](lock)), i64(0))


# ============================================================
# Context switching: minimal cooperative context ("MiniCtx")
#
# Only saves callee-saved registers for a COOPERATIVE switch.
# This is what the calling convention guarantees the caller
# already saved; we just need to preserve the callee's state.
#
# x86_64 (System V ABI / Win64):
#   Callee-saved: rbx, rbp, r12-r15, rsp (+ rip via return addr)
#   + on Win64: rdi, rsi, xmm6-xmm15  (we skip XMM for integer tasks)
#   Total: 7 registers = 56 bytes
#
# AArch64 (AAPCS64):
#   Callee-saved: x19-x28, x29(fp), x30(lr), sp
#   + d8-d15 (SIMD) - skip for integer tasks
#   Total: 13 registers = 104 bytes
#
# The context switch function is `ctx_swap(from: ptr[MiniCtx], to: ptr[MiniCtx])`
# It saves current state into `from`, loads state from `to`, and jumps.
# ============================================================

if IS_X86_64 and IS_WINDOWS:
    # Win64 ABI callee-saved set is wider than System V:
    #   rsp, rbp, rbx, r12-r15, rdi, rsi (9 GP regs) + rip
    #   xmm6-xmm15 (10 SIMD regs, 128-bit each)
    # Layout (must match asm/ctx_x86_64_win.S):
    #   [0..72]   GP regs + rip   (10 u64 slots)
    #   [80..240) xmm6..xmm15     (10 * 16 bytes, movups)
    #   [240..256) pad
    # Total: 32 u64 slots = 256 bytes.
    MINI_CTX_SIZE = 256

    @compile
    class MiniCtx:
        regs: array[u64, 32]

elif IS_X86_64:
    # System V ABI: rsp, rbp, rbx, r12-r15, rip
    MINI_CTX_SIZE = 64  # 8 slots x 8 bytes

    @compile
    class MiniCtx:
        regs: array[u64, 8]

elif IS_AARCH64:
    # AAPCS64 callee-saved set:
    #   sp, x19-x28, x29(fp), x30(lr)  (13 GP slots)
    #   d8-d15 (8 SIMD slots, lower 64 bits)
    # Layout (must match asm/ctx_aarch64.S):
    #   [0..104)  GP regs              (13 u64 slots)
    #   [104..168) d8..d15             (8 u64 slots)
    #   [168..176) pad                 (1 u64 slot)
    # Total: 22 u64 slots = 176 bytes.
    MINI_CTX_SIZE = 176

    @compile
    class MiniCtx:
        regs: array[u64, 22]

else:
    # Fallback: large context (will use C setjmp/longjmp)
    MINI_CTX_SIZE = 256

    @compile
    class MiniCtx:
        regs: array[u64, 32]


# ---- Assembly for context switch ----
#
# These are linked as external symbols.  The actual .s files are
# provided per platform in pythoc/std/runtime/asm/
#
# Signature: void ctx_swap(MiniCtx* from, MiniCtx* to)
# Signature: void ctx_make(MiniCtx* ctx, void* stack_top, void (*entry)(void*), void* arg)

@extern(lib=_CTX_ASM_PATH)
def ctx_swap(from_ctx: ptr[MiniCtx], to_ctx: ptr[MiniCtx]) -> void:
    """Save current context into from_ctx, restore from to_ctx."""
    pass

@extern(lib=_CTX_ASM_PATH)
def ctx_make(
    ctx: ptr[MiniCtx],
    stack_top: ptr[void],
    entry: ptr[void],
    arg: ptr[void]
) -> void:
    """Initialize ctx to start at entry(arg) on the given stack."""
    pass


# ============================================================
# Threading abstraction
# ============================================================

if IS_WINDOWS:
    # ---- Windows: Win32 threads ----

    @compile
    class ThreadHandle:
        handle: u64   # HANDLE (void* on Win64)
        id: u64       # OS thread id

    @compile
    class Mutex:
        _opaque: array[u64, 5]   # CRITICAL_SECTION (40 bytes on Win64)

    @compile
    class CondVar:
        _opaque: array[u64, 1]   # CONDITION_VARIABLE (8 bytes)

    @extern(lib='kernel32')
    def CreateThread(
        attrs: ptr[void], stack_size: u64,
        start: ptr[void], arg: ptr[void],
        flags: u32, thread_id: ptr[u32]
    ) -> u64:
        pass

    @extern(lib='kernel32')
    def WaitForSingleObject(handle: u64, milliseconds: u32) -> u32:
        pass

    @extern(lib='kernel32')
    def CloseHandle(handle: u64) -> i32:
        pass

    @extern(lib='kernel32')
    def GetCurrentThreadId() -> u32:
        pass

    @extern(lib='kernel32')
    def InitializeCriticalSection(cs: ptr[Mutex]) -> void:
        pass

    @extern(lib='kernel32')
    def EnterCriticalSection(cs: ptr[Mutex]) -> void:
        pass

    @extern(lib='kernel32')
    def LeaveCriticalSection(cs: ptr[Mutex]) -> void:
        pass

    @extern(lib='kernel32')
    def DeleteCriticalSection(cs: ptr[Mutex]) -> void:
        pass

    @extern(lib='kernel32')
    def InitializeConditionVariable(cv: ptr[CondVar]) -> void:
        pass

    @extern(lib='kernel32')
    def SleepConditionVariableCS(cv: ptr[CondVar], cs: ptr[Mutex], ms: u32) -> i32:
        pass

    @extern(lib='kernel32')
    def WakeConditionVariable(cv: ptr[CondVar]) -> void:
        pass

    @extern(lib='kernel32')
    def WakeAllConditionVariable(cv: ptr[CondVar]) -> void:
        pass

    INFINITE = u32(0xFFFFFFFF)

    # ---- Portable wrappers ----

    @compile
    def thread_create(start: ptr[void], arg: ptr[void]) -> ThreadHandle:
        h: ThreadHandle
        tid: u32 = 0
        h.handle = CreateThread(
            nullptr, u64(0), start, arg, u32(0), ptr[u32](ptr[void](ptr(tid)))
        )
        h.id = u64(tid)
        return h

    @compile
    def thread_join(t: ThreadHandle) -> void:
        WaitForSingleObject(t.handle, INFINITE)
        CloseHandle(t.handle)

    @compile
    def thread_current() -> ThreadHandle:
        h: ThreadHandle
        h.handle = u64(0)
        h.id = u64(GetCurrentThreadId())
        return h

    @compile
    def thread_equal(a: ThreadHandle, b: ThreadHandle) -> i32:
        if a.id == b.id:
            return i32(1)
        return i32(0)

    @compile
    def mutex_init(m: ptr[Mutex]) -> void:
        InitializeCriticalSection(m)

    @compile
    def mutex_lock(m: ptr[Mutex]) -> void:
        EnterCriticalSection(m)

    @compile
    def mutex_unlock(m: ptr[Mutex]) -> void:
        LeaveCriticalSection(m)

    @compile
    def mutex_destroy(m: ptr[Mutex]) -> void:
        DeleteCriticalSection(m)

    @compile
    def condvar_init(cv: ptr[CondVar]) -> void:
        InitializeConditionVariable(cv)

    @compile
    def condvar_wait(cv: ptr[CondVar], m: ptr[Mutex]) -> void:
        SleepConditionVariableCS(cv, m, INFINITE)

    @compile
    def condvar_signal(cv: ptr[CondVar]) -> void:
        WakeConditionVariable(cv)

    @compile
    def condvar_broadcast(cv: ptr[CondVar]) -> void:
        WakeAllConditionVariable(cv)

    @compile
    def condvar_destroy(cv: ptr[CondVar]) -> void:
        pass  # Windows CONDITION_VARIABLE needs no destruction

else:
    # ---- POSIX: pthread (Linux + macOS) ----

    @compile
    class ThreadHandle:
        handle: u64   # pthread_t
        id: u64       # same value, kept for portable equality

    # pthread_mutex_t sizes:
    #   Linux x86_64: 40 bytes, Linux aarch64: 48 bytes
    #   macOS: 64 bytes (both architectures)
    _MUTEX_SIZE = 64 if IS_MACOS else 48

    @compile
    class Mutex:
        _opaque: array[u64, 8]  # use max size for portability

    # pthread_cond_t sizes:
    #   Linux: 48 bytes, macOS: 48 bytes
    @compile
    class CondVar:
        _opaque: array[u64, 6]

    @extern(lib='pthread')
    def pthread_create(
        thread: ptr[u64], attr: ptr[void],
        start_routine: ptr[void], arg: ptr[void]
    ) -> i32:
        pass

    @extern(lib='pthread')
    def pthread_join(thread: u64, retval: ptr[ptr[void]]) -> i32:
        pass

    @extern(lib='pthread')
    def pthread_self() -> u64:
        pass

    @extern(lib='pthread')
    def pthread_mutex_init(mutex: ptr[Mutex], attr: ptr[void]) -> i32:
        pass

    @extern(lib='pthread')
    def pthread_mutex_lock(mutex: ptr[Mutex]) -> i32:
        pass

    @extern(lib='pthread')
    def pthread_mutex_unlock(mutex: ptr[Mutex]) -> i32:
        pass

    @extern(lib='pthread')
    def pthread_mutex_destroy(mutex: ptr[Mutex]) -> i32:
        pass

    @extern(lib='pthread')
    def pthread_cond_init(cond: ptr[CondVar], attr: ptr[void]) -> i32:
        pass

    @extern(lib='pthread')
    def pthread_cond_wait(cond: ptr[CondVar], mutex: ptr[Mutex]) -> i32:
        pass

    @extern(lib='pthread')
    def pthread_cond_signal(cond: ptr[CondVar]) -> i32:
        pass

    @extern(lib='pthread')
    def pthread_cond_broadcast(cond: ptr[CondVar]) -> i32:
        pass

    @extern(lib='pthread')
    def pthread_cond_destroy(cond: ptr[CondVar]) -> i32:
        pass

    # ---- Portable wrappers ----

    @compile
    def thread_create(start: ptr[void], arg: ptr[void]) -> ThreadHandle:
        h: ThreadHandle
        pthread_create(
            ptr[u64](ptr[void](ptr(h.handle))),
            nullptr, start, arg
        )
        h.id = h.handle
        return h

    @compile
    def thread_join(t: ThreadHandle) -> void:
        pthread_join(t.handle, nullptr)

    @compile
    def thread_current() -> ThreadHandle:
        h: ThreadHandle
        h.handle = pthread_self()
        h.id = h.handle
        return h

    @compile
    def thread_equal(a: ThreadHandle, b: ThreadHandle) -> i32:
        if a.id == b.id:
            return i32(1)
        return i32(0)

    @compile
    def mutex_init(m: ptr[Mutex]) -> void:
        pthread_mutex_init(m, nullptr)

    @compile
    def mutex_lock(m: ptr[Mutex]) -> void:
        pthread_mutex_lock(m)

    @compile
    def mutex_unlock(m: ptr[Mutex]) -> void:
        pthread_mutex_unlock(m)

    @compile
    def mutex_destroy(m: ptr[Mutex]) -> void:
        pthread_mutex_destroy(m)

    @compile
    def condvar_init(cv: ptr[CondVar]) -> void:
        pthread_cond_init(cv, nullptr)

    @compile
    def condvar_wait(cv: ptr[CondVar], m: ptr[Mutex]) -> void:
        pthread_cond_wait(cv, m)

    @compile
    def condvar_signal(cv: ptr[CondVar]) -> void:
        pthread_cond_signal(cv)

    @compile
    def condvar_broadcast(cv: ptr[CondVar]) -> void:
        pthread_cond_broadcast(cv)

    @compile
    def condvar_destroy(cv: ptr[CondVar]) -> void:
        pthread_cond_destroy(cv)


# ============================================================
# Cross-thread notification
# ============================================================

if IS_LINUX:
    @extern(lib='c')
    def eventfd(initval: u32, flags: i32) -> i32:
        pass

    EFD_NONBLOCK  = i32(2048)
    EFD_SEMAPHORE = i32(1)

# Common: read/write/close (POSIX) or ReadFile/WriteFile (Windows)
if not IS_WINDOWS:
    @extern(lib='c')
    def read(fd: i32, buf: ptr[void], count: u64) -> i64:
        pass

    @extern(lib='c')
    def write(fd: i32, buf: ptr[void], count: u64) -> i64:
        pass

    @extern(lib='c')
    def close(fd: i32) -> i32:
        pass

    @extern(lib='c')
    def pipe(fds: ptr[i32]) -> i32:
        pass


# ============================================================
# Stack allocation (platform-aware)
#
# On Linux/macOS: mmap with guard page
# On Windows: VirtualAlloc with guard page
# For simplicity, start with malloc (no guard page).
# Production: add mmap/VirtualAlloc + guard page.
# ============================================================

@compile
def stack_alloc(size: u64) -> ptr[void]:
    """Allocate a coroutine stack.  Returns bottom of usable stack.

    Note: stacks grow downward on x86_64 and ARM64.
    The returned pointer is the LOW address.
    stack_top = returned_ptr + size
    """
    return effect.mem.malloc(size)


@compile
def stack_free(stack_bottom: ptr[void], size: u64) -> void:
    """Free a stack allocated by stack_alloc."""
    effect.mem.free(stack_bottom)
