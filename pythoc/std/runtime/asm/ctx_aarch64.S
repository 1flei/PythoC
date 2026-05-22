// ============================================================
// ctx_swap / ctx_make for AArch64 (AAPCS64)
// Works on: Linux ARM64, macOS ARM64 (Apple Silicon)
//
// AAPCS64 callee-saved registers:
//   x19-x28 (general purpose)
//   x29 (frame pointer / fp)
//   x30 (link register / lr)
//   sp (stack pointer)
//   d8-d15 (SIMD - lower 64 bits of v8-v15)
//
// MiniCtx layout (14 × u64 = 112 bytes, integer only):
//   [0]   sp
//   [8]   x19
//   [16]  x20
//   [24]  x21
//   [32]  x22
//   [40]  x23
//   [48]  x24
//   [56]  x25
//   [64]  x26
//   [72]  x27
//   [80]  x28
//   [88]  x29 (fp)
//   [96]  x30 (lr)  -- "return address"
//   [104] padding / reserved
//
// Parameters: x0, x1, x2, x3
// ============================================================

.text

// ---- ctx_swap(from_ctx: *MiniCtx, to_ctx: *MiniCtx) ----
// x0 = from_ctx, x1 = to_ctx

.globl ctx_swap
#ifdef __APPLE__
.globl _ctx_swap
_ctx_swap:
#endif
ctx_swap:
    // Save callee-saved into from_ctx (x0)
    mov     x2, sp
    str     x2,  [x0, #0]      // sp
    stp     x19, x20, [x0, #8]
    stp     x21, x22, [x0, #24]
    stp     x23, x24, [x0, #40]
    stp     x25, x26, [x0, #56]
    stp     x27, x28, [x0, #72]
    str     x29, [x0, #88]      // fp
    str     x30, [x0, #96]      // lr (return address)

    // Restore from to_ctx (x1)
    ldr     x2,  [x1, #0]
    mov     sp, x2              // restore sp
    ldp     x19, x20, [x1, #8]
    ldp     x21, x22, [x1, #24]
    ldp     x23, x24, [x1, #40]
    ldp     x25, x26, [x1, #56]
    ldp     x27, x28, [x1, #72]
    ldr     x29, [x1, #88]      // restore fp
    ldr     x30, [x1, #96]      // restore lr
    ret                          // branch to x30


// ---- ctx_make(ctx: *MiniCtx, stack_top: *void, entry: fn(*void), arg: *void) ----
// x0 = ctx, x1 = stack_top, x2 = entry, x3 = arg

.globl ctx_make
#ifdef __APPLE__
.globl _ctx_make
_ctx_make:
#endif
ctx_make:
    // Align stack_top to 16 bytes (AAPCS64 requirement)
    and     x1, x1, #~0xF

    // Store into MiniCtx
    str     x1,  [x0, #0]       // sp = stack_top (aligned)
    str     x3,  [x0, #8]       // x19 = arg (preserved across call)
    str     x2,  [x0, #16]      // x20 = entry (preserved across call)

    // Zero out remaining callee-saved
    stp     xzr, xzr, [x0, #24]  // x21, x22
    stp     xzr, xzr, [x0, #40]  // x23, x24
    stp     xzr, xzr, [x0, #56]  // x25, x26
    stp     xzr, xzr, [x0, #72]  // x27, x28
    str     xzr, [x0, #88]       // fp = 0

    // lr = trampoline address
    adr     x4, .Lctx_entry_aarch64
    str     x4, [x0, #96]        // x30 (lr) = trampoline

    ret

// Trampoline: entered via ctx_swap when a fresh context is first run.
// At this point: x19 = arg, x20 = entry (set by ctx_make).
.Lctx_entry_aarch64:
    mov     x0, x19              // arg -> first parameter
    blr     x20                  // call entry(arg)
    // If entry returns - should not happen, task must explicitly finish
    brk     #0x1                 // trap


// Mark stack as non-executable (Linux)
#if defined(__linux__)
.section .note.GNU-stack,"",%progbits
#endif
