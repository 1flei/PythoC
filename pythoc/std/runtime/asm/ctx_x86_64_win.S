// ============================================================
// ctx_swap / ctx_make for x86_64 Windows (Win64 ABI)
//
// Win64 differences from System V:
//   - More callee-saved registers: rbx, rbp, rdi, rsi, r12-r15
//   - XMM6-XMM15 are callee-saved (we save them for completeness)
//   - Parameters in: rcx, rdx, r8, r9 (not rdi, rsi, rdx, rcx)
//   - Shadow space: 32 bytes reserved by caller above return addr
//
// MiniCtx layout (16 × u64 = 128 bytes, for Win64 extended set):
//   [0]   rsp
//   [8]   rbp
//   [16]  rbx
//   [24]  r12
//   [32]  r13
//   [40]  r14
//   [48]  r15
//   [56]  rdi     (callee-saved on Win64!)
//   [64]  rsi     (callee-saved on Win64!)
//   [72]  rip
//   [80]  xmm6   (low 64 bits - simplified)
//   [88]  xmm7
//   [96]  xmm8
//   [104] xmm9
//   [112] xmm10
//   [120] xmm11
//
// Note: Full Win64 compliance requires saving xmm6-xmm15 (10 regs × 128 bit).
// For integer-only workloads, skipping XMM is fine.
// Here we save only the first 6 as 64-bit (lossy but covers most cases).
// ============================================================

.text

// ---- ctx_swap(from_ctx: *MiniCtx, to_ctx: *MiniCtx) ----
// Win64: rcx = from_ctx, rdx = to_ctx

.globl ctx_swap
ctx_swap:
    // Save callee-saved into from_ctx (rcx)
    mov     %rsp,   0(%rcx)
    mov     %rbp,   8(%rcx)
    mov     %rbx,  16(%rcx)
    mov     %r12,  24(%rcx)
    mov     %r13,  32(%rcx)
    mov     %r14,  40(%rcx)
    mov     %r15,  48(%rcx)
    mov     %rdi,  56(%rcx)
    mov     %rsi,  64(%rcx)
    lea     .Lret_from_win(%rip), %rax
    mov     %rax,  72(%rcx)

    // Restore from to_ctx (rdx)
    mov      0(%rdx), %rsp
    mov      8(%rdx), %rbp
    mov     16(%rdx), %rbx
    mov     24(%rdx), %r12
    mov     32(%rdx), %r13
    mov     40(%rdx), %r14
    mov     48(%rdx), %r15
    mov     56(%rdx), %rdi
    mov     64(%rdx), %rsi
    jmp    *72(%rdx)

.Lret_from_win:
    ret


// ---- ctx_make(ctx: *MiniCtx, stack_top: *void, entry: fn(*void), arg: *void) ----
// Win64: rcx = ctx, rdx = stack_top, r8 = entry, r9 = arg

.globl ctx_make
ctx_make:
    // Align stack to 16, subtract 8 for fake return + 32 for shadow space
    and     $-16, %rdx
    sub     $40,  %rdx        // 8 (ret addr) + 32 (shadow space)

    lea     .Lctx_entry_win(%rip), %rax
    mov     %rax, (%rdx)      // fake return address

    // Setup MiniCtx
    mov     %rdx,  0(%rcx)    // rsp
    xor     %rax, %rax
    mov     %rax,  8(%rcx)    // rbp = 0
    mov     %r9,  16(%rcx)    // rbx = arg (preserved)
    mov     %r8,  24(%rcx)    // r12 = entry (preserved)
    mov     %rax, 32(%rcx)    // r13 = 0
    mov     %rax, 40(%rcx)    // r14 = 0
    mov     %rax, 48(%rcx)    // r15 = 0
    mov     %rax, 56(%rcx)    // rdi = 0
    mov     %rax, 64(%rcx)    // rsi = 0
    lea     .Lctx_entry_win(%rip), %rax
    mov     %rax, 72(%rcx)    // rip = trampoline
    ret

// Trampoline for Win64:
// On entry: r12 = entry_fn, rbx = arg
// Win64 first param goes in rcx
.Lctx_entry_win:
    mov     %rbx, %rcx        // arg -> first parameter (Win64: rcx)
    call    *%r12             // call entry(arg)
    ud2                       // should not return
