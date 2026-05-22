# ============================================================
# ctx_swap / ctx_make for x86_64 System V ABI (Linux / macOS)
#
# MiniCtx layout (8 × u64 = 64 bytes):
#   [0]  rsp
#   [8]  rbp
#   [16] rbx
#   [24] r12
#   [32] r13
#   [40] r14
#   [48] r15
#   [56] rip (return address)
#
# Calling convention:
#   rdi = first arg  (from_ctx / ctx)
#   rsi = second arg (to_ctx / stack_top)
#   rdx = third arg  (entry)
#   rcx = fourth arg (arg)
# ============================================================

.text

# ---- ctx_swap(from_ctx: *MiniCtx, to_ctx: *MiniCtx) ----
#
# Saves current callee-saved regs into from_ctx,
# restores from to_ctx, effectively "jumping" to to_ctx's saved state.

.globl ctx_swap
#ifdef __APPLE__
.globl _ctx_swap
_ctx_swap:
#endif
ctx_swap:
    # Save callee-saved registers into from_ctx (rdi)
    mov     %rsp,   0(%rdi)
    mov     %rbp,   8(%rdi)
    mov     %rbx,  16(%rdi)
    mov     %r12,  24(%rdi)
    mov     %r13,  32(%rdi)
    mov     %r14,  40(%rdi)
    mov     %r15,  48(%rdi)
    # Save return address (on stack) as rip
    lea     .Lret_from(%rip), %rax
    mov     %rax,  56(%rdi)

    # Restore callee-saved registers from to_ctx (rsi)
    mov      0(%rsi), %rsp
    mov      8(%rsi), %rbp
    mov     16(%rsi), %rbx
    mov     24(%rsi), %r12
    mov     32(%rsi), %r13
    mov     40(%rsi), %r14
    mov     48(%rsi), %r15
    # Jump to saved rip
    jmp    *56(%rsi)

.Lret_from:
    ret


# ---- ctx_make(ctx: *MiniCtx, stack_top: *void, entry: fn(*void), arg: *void) ----
#
# Sets up ctx so that ctx_swap(..., ctx) will begin executing entry(arg)
# on the provided stack.
#
# stack_top is the HIGH address of the allocated stack (stack grows down).

.globl ctx_make
#ifdef __APPLE__
.globl _ctx_make
_ctx_make:
#endif
ctx_make:
    # rdi = ctx, rsi = stack_top, rdx = entry, rcx = arg

    # Align stack_top to 16 bytes (ABI requirement) then subtract 8
    # for the fake return address (so on entry, rsp % 16 == 8, which
    # is the state after a CALL instruction).
    and     $-16, %rsi
    sub     $8,   %rsi

    # Place a "return address" that points to ctx_entry_trampoline
    lea     .Lctx_entry_trampoline(%rip), %rax
    mov     %rax, (%rsi)

    # Setup MiniCtx fields
    mov     %rsi,  0(%rdi)    # rsp = adjusted stack_top
    xor     %rax, %rax
    mov     %rax,  8(%rdi)    # rbp = 0 (no frame)
    mov     %rcx, 16(%rdi)    # rbx = arg (preserved across call)
    mov     %rdx, 24(%rdi)    # r12 = entry (preserved across call)
    mov     %rax, 32(%rdi)    # r13 = 0
    mov     %rax, 40(%rdi)    # r14 = 0
    mov     %rax, 48(%rdi)    # r15 = 0
    lea     .Lctx_entry_trampoline(%rip), %rax
    mov     %rax, 56(%rdi)    # rip = trampoline
    ret

# Trampoline: called when ctx_swap jumps to a fresh ctx.
# At this point: r12 = entry, rbx = arg (set up by ctx_make).
.Lctx_entry_trampoline:
    mov     %rbx, %rdi        # arg -> first parameter (System V: rdi)
    call    *%r12             # call entry(arg)
    # If entry returns, halt (shouldn't happen - task should explicitly finish)
    ud2


# Mark stack as non-executable (security hardening)
#if defined(__linux__)
.section .note.GNU-stack,"",%progbits
#endif
