// ============================================================
// ctx_swap / ctx_make for x86_64 Windows (Win64 ABI)
//
// Win64 differences from System V:
//   - More callee-saved registers: rbx, rbp, rdi, rsi, r12-r15
//   - XMM6-XMM15 are callee-saved (saved/restored here in full 128-bit)
//   - Parameters in: rcx, rdx, r8, r9 (not rdi, rsi, rdx, rcx)
//   - Shadow space: 32 bytes reserved by caller above return addr
//
// MiniCtx layout (must match platform.py: 32 x u64 = 256 bytes):
//   [0]   rsp
//   [8]   rbp
//   [16]  rbx
//   [24]  r12
//   [32]  r13
//   [40]  r14
//   [48]  r15
//   [56]  rdi     (callee-saved on Win64)
//   [64]  rsi     (callee-saved on Win64)
//   [72]  rip
//   [80]  xmm6   (16 bytes, full 128-bit, movups - no alignment assumption)
//   [96]  xmm7
//   [112] xmm8
//   [128] xmm9
//   [144] xmm10
//   [160] xmm11
//   [176] xmm12
//   [192] xmm13
//   [208] xmm14
//   [224] xmm15
//   [240] pad (16 bytes, unused)
// ============================================================

.text

// ---- ctx_swap(from_ctx: *MiniCtx, to_ctx: *MiniCtx) ----
// Win64: rcx = from_ctx, rdx = to_ctx

.globl ctx_swap
ctx_swap:
    // Save callee-saved integer regs into from_ctx (rcx)
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

    // Save callee-saved xmm regs (full 128-bit each)
    movups  %xmm6,   80(%rcx)
    movups  %xmm7,   96(%rcx)
    movups  %xmm8,  112(%rcx)
    movups  %xmm9,  128(%rcx)
    movups  %xmm10, 144(%rcx)
    movups  %xmm11, 160(%rcx)
    movups  %xmm12, 176(%rcx)
    movups  %xmm13, 192(%rcx)
    movups  %xmm14, 208(%rcx)
    movups  %xmm15, 224(%rcx)

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

    movups   80(%rdx), %xmm6
    movups   96(%rdx), %xmm7
    movups  112(%rdx), %xmm8
    movups  128(%rdx), %xmm9
    movups  144(%rdx), %xmm10
    movups  160(%rdx), %xmm11
    movups  176(%rdx), %xmm12
    movups  192(%rdx), %xmm13
    movups  208(%rdx), %xmm14
    movups  224(%rdx), %xmm15

    jmp    *72(%rdx)

.Lret_from_win:
    ret


// ---- ctx_make(ctx: *MiniCtx, stack_top: *void, entry: fn(*void), arg: *void) ----
// Win64: rcx = ctx, rdx = stack_top, r8 = entry, r9 = arg

.globl ctx_make
ctx_make:
    // Align stack to 16, then reserve the Win64 caller shadow space for
    // the trampoline's call to entry(arg).  ctx_swap jumps to the trampoline
    // rather than calling it, so the trampoline itself must start in caller
    // state: rsp % 16 == 0 with 32 bytes of home space available.
    and     $-16, %rdx
    sub     $32,  %rdx        // 32-byte shadow space for entry(arg)

    // Setup integer slots in MiniCtx
    mov     %rdx,  0(%rcx)    // rsp
    xor     %rax, %rax
    mov     %rax,  8(%rcx)    // rbp = 0
    mov     %r9,  16(%rcx)    // rbx = arg (preserved across call)
    mov     %r8,  24(%rcx)    // r12 = entry (preserved across call)
    mov     %rax, 32(%rcx)    // r13 = 0
    mov     %rax, 40(%rcx)    // r14 = 0
    mov     %rax, 48(%rcx)    // r15 = 0
    mov     %rax, 56(%rcx)    // rdi = 0
    mov     %rax, 64(%rcx)    // rsi = 0
    lea     .Lctx_entry_win(%rip), %rax
    mov     %rax, 72(%rcx)    // rip = trampoline

    // Zero xmm6-xmm15 slots (10 * 16 = 160 bytes starting at offset 80)
    // Use two GP stores per slot to avoid touching xmm regs here.
    xor     %rax, %rax
    mov     %rax,  80(%rcx)
    mov     %rax,  88(%rcx)
    mov     %rax,  96(%rcx)
    mov     %rax, 104(%rcx)
    mov     %rax, 112(%rcx)
    mov     %rax, 120(%rcx)
    mov     %rax, 128(%rcx)
    mov     %rax, 136(%rcx)
    mov     %rax, 144(%rcx)
    mov     %rax, 152(%rcx)
    mov     %rax, 160(%rcx)
    mov     %rax, 168(%rcx)
    mov     %rax, 176(%rcx)
    mov     %rax, 184(%rcx)
    mov     %rax, 192(%rcx)
    mov     %rax, 200(%rcx)
    mov     %rax, 208(%rcx)
    mov     %rax, 216(%rcx)
    mov     %rax, 224(%rcx)
    mov     %rax, 232(%rcx)
    ret

// Trampoline for Win64:
// On entry: r12 = entry_fn, rbx = arg
// Win64 first param goes in rcx
.Lctx_entry_win:
    mov     %rbx, %rcx        // arg -> first parameter (Win64: rcx)
    call    *%r12             // call entry(arg)
    ud2                       // should not return

