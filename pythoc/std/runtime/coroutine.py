"""
Coroutine: Stackful lightweight execution context (cross-platform).

A Coroutine is a pair of (stack allocation, saved CPU context).
It can be suspended and resumed with zero kernel involvement.

Design:
    Each coroutine owns a fixed-size stack (default 64KB, configurable).
    Context switch uses hand-written assembly (ctx_swap) that only
    saves/restores callee-saved registers — much faster than ucontext.

    Supported platforms:
    - x86_64 Linux (System V ABI)
    - x86_64 macOS (System V ABI)
    - x86_64 Windows (Win64 ABI)
    - AArch64 Linux (AAPCS64)
    - AArch64 macOS / Apple Silicon (AAPCS64)

Lifecycle:
    coro_alloc()   -> allocate stack + context
    coro_init()    -> setup entry function via ctx_make
    coro_switch()  -> ctx_swap between two coroutines
    coro_free()    -> deallocate stack

Memory layout:
    Coroutine struct:
    +----------+
    | ctx      |  <- MiniCtx (56-112 bytes depending on arch)
    | stack    |  <- ptr to allocated stack memory (low address)
    | stack_sz |  <- stack size in bytes
    | state    |  <- READY / RUNNING / SUSPENDED / DONE
    +----------+
"""
from __future__ import annotations

from .policy import bind_mem
bind_mem()

from pythoc import compile, effect, i32, i64, u64, u8, ptr, void, struct, nullptr, sizeof
from pythoc.libc.string import memset

from .platform import (
    MiniCtx, ctx_swap, ctx_make,
    stack_alloc, stack_free,
)


# ============================================================
# Coroutine states
# ============================================================

CORO_READY     = i32(0)   # Created, not yet started
CORO_RUNNING   = i32(1)   # Currently executing on a worker
CORO_SUSPENDED = i32(2)   # Yielded, waiting to be resumed
CORO_DONE      = i32(3)   # Entry function returned


# ============================================================
# Coroutine struct
# ============================================================

DEFAULT_STACK_SIZE = u64(65536)  # 64KB per coroutine stack

@compile
class Coroutine:
    ctx: MiniCtx           # saved execution context (callee-saved registers)
    stack_bottom: ptr[void]  # allocated stack memory (low address)
    stack_size: u64        # size of allocated stack
    state: i32             # CORO_READY | CORO_RUNNING | CORO_SUSPENDED | CORO_DONE


# ============================================================
# Coroutine lifecycle functions
# ============================================================

@compile
def coro_alloc(stack_size: u64) -> ptr[Coroutine]:
    """Allocate a coroutine with the given stack size.

    Returns ptr to Coroutine struct.  Caller owns the memory.
    Must call coro_free() when done.
    """
    coro: ptr[Coroutine] = ptr[Coroutine](effect.mem.malloc(u64(sizeof(Coroutine))))
    memset(ptr[void](coro), 0, i64(sizeof(Coroutine)))

    # Allocate stack
    stack: ptr[void] = stack_alloc(stack_size)
    coro.stack_bottom = stack
    coro.stack_size = stack_size
    coro.state = CORO_READY

    return coro


@compile
def coro_init(coro: ptr[Coroutine], entry: ptr[void], arg: ptr[void]) -> void:
    """Initialize coroutine to start at entry(arg).

    After this call, coro_switch() to this coroutine will begin
    executing entry(arg) on the coroutine's own stack.

    Note: stack grows downward on both x86_64 and AArch64.
    stack_top = stack_bottom + stack_size
    """
    # Calculate stack top (high address, where SP starts)
    stack_top: ptr[void] = ptr[void](
        ptr[u8](coro.stack_bottom) + i64(coro.stack_size)
    )

    # Setup context via assembly helper
    ctx_make(
        ptr[MiniCtx](ptr[void](ptr(coro.ctx))),
        stack_top,
        entry,
        arg
    )


@compile
def coro_switch(from_coro: ptr[Coroutine], to_coro: ptr[Coroutine]) -> void:
    """Context switch: suspend from_coro, resume to_coro.

    Saves current callee-saved registers into from_coro.ctx,
    restores registers from to_coro.ctx.
    Returns when someone switches back to from_coro.
    """
    from_coro.state = CORO_SUSPENDED
    to_coro.state = CORO_RUNNING

    ctx_swap(
        ptr[MiniCtx](ptr[void](ptr(from_coro.ctx))),
        ptr[MiniCtx](ptr[void](ptr(to_coro.ctx)))
    )


@compile
def coro_free(coro: ptr[Coroutine]) -> void:
    """Deallocate coroutine stack and struct."""
    if coro.stack_bottom != nullptr:
        stack_free(coro.stack_bottom, coro.stack_size)
    effect.mem.free(ptr[void](coro))
