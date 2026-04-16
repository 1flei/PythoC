"""
Platform-specific va_arg lowering for C ABI varargs.

Clang does NOT rely on the LLVM generic va_arg instruction for most targets.
Instead, the frontend emits explicit GEP/load/store sequences that match
each target's va_list layout and register save area conventions.

We replicate the same approach here so that pythoc's bare *args functions
produce correct code on every platform the CI covers:

- Win64 (x86_64-windows-gnu):  va_list = i8*, slot = 8 bytes
- AArch64 Windows:             same as Win64
- x86_64 SysV (Linux/macOS):   va_list = { i32, i32, i8*, i8* }
- AArch64 AAPCS (Linux):       va_list = { i8*, i8*, i8*, i32, i32 }

Reference: clang/lib/CodeGen/Targets/{X86,AArch64}.cpp
"""

from abc import ABC, abstractmethod
from typing import Any

from llvmlite import ir


# ---- helpers shared by all lowerings ----

_i8 = ir.IntType(8)
_i32 = ir.IntType(32)
_i64 = ir.IntType(64)
_i8ptr = ir.PointerType(_i8)


def _get_or_declare(module, name, fn_ty):
    """Get an existing global or declare a new function."""
    try:
        return module.get_global(name)
    except KeyError:
        return ir.Function(module, fn_ty, name)


def _call_va_start(builder, ap_ptr):
    fn = _get_or_declare(
        builder.module, "llvm.va_start",
        ir.FunctionType(ir.VoidType(), [_i8ptr]),
    )
    builder.call(fn, [ap_ptr])


def _call_va_end(builder, ap_ptr):
    fn = _get_or_declare(
        builder.module, "llvm.va_end",
        ir.FunctionType(ir.VoidType(), [_i8ptr]),
    )
    builder.call(fn, [ap_ptr])


def _type_size(llvm_type):
    """Approximate size of an LLVM type in bytes (used for slot bumping)."""
    if isinstance(llvm_type, ir.IntType):
        return (llvm_type.width + 7) // 8
    if isinstance(llvm_type, ir.FloatType):
        return 4
    if isinstance(llvm_type, ir.DoubleType):
        return 8
    if isinstance(llvm_type, ir.PointerType):
        return 8
    if isinstance(llvm_type, ir.HalfType):
        return 2
    return 8


def _is_fp_type(llvm_type):
    """True for float/double/vector-of-float (SSE class on x86_64)."""
    return isinstance(llvm_type, (ir.FloatType, ir.DoubleType))


# ---- abstract base ----

class VAArgLowering(ABC):
    """Abstract base for platform-specific va_arg code emission."""

    @abstractmethod
    def emit_va_start(self, builder) -> Any:
        """Emit va_start. Returns an i8* pointer to the va_list storage."""

    @abstractmethod
    def emit_va_arg(self, builder, va_list_ptr, target_type, name="") -> Any:
        """Emit va_arg. Returns the loaded value of *target_type*."""

    def emit_va_end(self, builder, va_list_ptr) -> None:
        """Emit va_end. Default implementation calls llvm.va_end."""
        _call_va_end(builder, va_list_ptr)


# =========================================================================
# Win64 / AArch64-Windows
# =========================================================================

class VoidPtrVAArgLowering(VAArgLowering):
    """Win64 and AArch64-Windows va_arg lowering.

    va_list is a simple ``i8*``.  Each slot is 8-byte aligned.
    Types > 8 bytes (or not power-of-2 sized) are passed indirectly
    (the slot holds a pointer to the actual value).

    Replicates clang's ``emitVoidPtrDirectVAArg`` / ``WinX86_64ABIInfo::EmitVAArg``.
    """

    SLOT_SIZE = 8  # every slot is 8-byte aligned on Win64

    def emit_va_start(self, builder) -> Any:
        # Win64 va_list is just an i8** (pointer to the current arg pointer).
        alloca = builder.alloca(_i8ptr, name="va_list")
        ap = builder.bitcast(alloca, _i8ptr)
        _call_va_start(builder, ap)
        return ap

    def emit_va_arg(self, builder, va_list_ptr, target_type, name="") -> Any:
        slot_size = self.SLOT_SIZE

        # Load current arg pointer
        ap_addr = builder.bitcast(va_list_ptr, ir.PointerType(_i8ptr))
        cur_ptr = builder.load(ap_addr, name="argp.cur")

        # Advance by max(type_size, slot_size) rounded up to slot_size
        type_sz = _type_size(target_type)
        advance = max(type_sz, slot_size)
        # Align up to slot_size
        advance = (advance + slot_size - 1) // slot_size * slot_size

        next_ptr = builder.gep(cur_ptr, [ir.Constant(_i32, advance)],
                               name="argp.next")
        builder.store(next_ptr, ap_addr)

        # Types that don't fit in 8 bytes or aren't 1/2/4/8 are indirect:
        # the slot holds a pointer to the real value.
        is_indirect = type_sz > 8 or type_sz not in (1, 2, 4, 8)

        if is_indirect:
            ptr_to_val = builder.bitcast(
                cur_ptr, ir.PointerType(ir.PointerType(target_type)))
            actual_ptr = builder.load(ptr_to_val, name="indirect.ptr")
            return builder.load(actual_ptr, name=name or "va.arg")

        typed_ptr = builder.bitcast(cur_ptr, ir.PointerType(target_type))
        return builder.load(typed_ptr, name=name or "va.arg")

    # va_end is the default (calls llvm.va_end)


# =========================================================================
# x86_64 System V (Linux / macOS)
# =========================================================================

class X86_64SysVVAArgLowering(VAArgLowering):
    """x86_64 System V ABI va_arg lowering.

    va_list = [1 x { i32, i32, i8*, i8* }]
      field 0: gp_offset   (i32)
      field 1: fp_offset   (i32)
      field 2: overflow_arg_area (i8*)
      field 3: reg_save_area    (i8*)

    GP args live at reg_save_area + gp_offset  (6 regs, 0..47)
    FP args live at reg_save_area + fp_offset  (8 regs, 48..175)
    Stack args live at overflow_arg_area (bump forward by 8).

    Replicates clang's ``X86_64ABIInfo::EmitVAArg`` (simplified for scalar
    types that pythoc currently supports: integers, floats, pointers).
    """

    # va_list struct layout
    _va_list_tag = ir.LiteralStructType([_i32, _i32, _i8ptr, _i8ptr])
    _va_list_type = ir.ArrayType(_va_list_tag, 1)

    # Offsets into register save area
    GP_MAX_OFFSET = 48   # 6 GP regs * 8 bytes
    FP_MAX_OFFSET = 176  # GP_MAX_OFFSET + 8 XMM regs * 16 bytes

    def emit_va_start(self, builder) -> Any:
        alloca = builder.alloca(self._va_list_type, name="va_list")
        ap = builder.bitcast(alloca, _i8ptr)
        _call_va_start(builder, ap)
        return ap

    def emit_va_arg(self, builder, va_list_ptr, target_type, name="") -> Any:
        # Get pointer to the first element of the [1 x struct] array.
        va_list_struct_ptr = builder.bitcast(
            va_list_ptr, ir.PointerType(self._va_list_tag))

        is_fp = _is_fp_type(target_type)

        if is_fp:
            return self._emit_fp_va_arg(builder, va_list_struct_ptr,
                                        target_type, name)
        return self._emit_gp_va_arg(builder, va_list_struct_ptr,
                                    target_type, name)

    def _emit_gp_va_arg(self, builder, va_struct_ptr, target_type, name):
        """Emit va_arg for a GP-class type (int, pointer)."""
        fn = builder.function

        # Load gp_offset
        gp_offset_p = builder.gep(
            va_struct_ptr, [ir.Constant(_i32, 0), ir.Constant(_i32, 0)],
            name="gp_offset_p")
        gp_offset = builder.load(gp_offset_p, name="gp_offset")

        # Check if arg fits in registers
        fits = builder.icmp_unsigned(
            "<=", gp_offset,
            ir.Constant(_i32, self.GP_MAX_OFFSET - 8),
            name="fits_in_gp")

        in_reg_bb = fn.append_basic_block("vaarg.gp.in_reg")
        in_mem_bb = fn.append_basic_block("vaarg.gp.in_mem")
        end_bb = fn.append_basic_block("vaarg.gp.end")

        builder.cbranch(fits, in_reg_bb, in_mem_bb)

        # ---- in_reg path ----
        builder.position_at_end(in_reg_bb)
        reg_save = builder.load(
            builder.gep(va_struct_ptr,
                        [ir.Constant(_i32, 0), ir.Constant(_i32, 3)]),
            name="reg_save_area")
        reg_addr = builder.gep(reg_save, [gp_offset], name="reg_addr")
        reg_val = builder.load(
            builder.bitcast(reg_addr, ir.PointerType(target_type)),
            name="reg_val")

        # Update gp_offset
        new_gp = builder.add(gp_offset, ir.Constant(_i32, 8))
        builder.store(new_gp, gp_offset_p)
        builder.branch(end_bb)
        in_reg_final = builder.block

        # ---- in_mem path ----
        builder.position_at_end(in_mem_bb)
        overflow_p = builder.gep(
            va_struct_ptr, [ir.Constant(_i32, 0), ir.Constant(_i32, 2)],
            name="overflow_p")
        overflow = builder.load(overflow_p, name="overflow")
        mem_val = builder.load(
            builder.bitcast(overflow, ir.PointerType(target_type)),
            name="mem_val")

        # Advance overflow by 8
        new_overflow = builder.gep(overflow, [ir.Constant(_i32, 8)],
                                   name="new_overflow")
        builder.store(new_overflow, overflow_p)
        builder.branch(end_bb)
        in_mem_final = builder.block

        # ---- merge ----
        builder.position_at_end(end_bb)
        phi = builder.phi(target_type, name=name or "va.arg")
        phi.add_incoming(reg_val, in_reg_final)
        phi.add_incoming(mem_val, in_mem_final)
        return phi

    def _emit_fp_va_arg(self, builder, va_struct_ptr, target_type, name):
        """Emit va_arg for an SSE-class type (float, double)."""
        fn = builder.function

        # Load fp_offset
        fp_offset_p = builder.gep(
            va_struct_ptr, [ir.Constant(_i32, 0), ir.Constant(_i32, 1)],
            name="fp_offset_p")
        fp_offset = builder.load(fp_offset_p, name="fp_offset")

        fits = builder.icmp_unsigned(
            "<=", fp_offset,
            ir.Constant(_i32, self.FP_MAX_OFFSET - 16),
            name="fits_in_fp")

        in_reg_bb = fn.append_basic_block("vaarg.fp.in_reg")
        in_mem_bb = fn.append_basic_block("vaarg.fp.in_mem")
        end_bb = fn.append_basic_block("vaarg.fp.end")

        builder.cbranch(fits, in_reg_bb, in_mem_bb)

        # ---- in_reg path ----
        builder.position_at_end(in_reg_bb)
        reg_save = builder.load(
            builder.gep(va_struct_ptr,
                        [ir.Constant(_i32, 0), ir.Constant(_i32, 3)]),
            name="reg_save_area")
        reg_addr = builder.gep(reg_save, [fp_offset], name="fp_reg_addr")
        reg_val = builder.load(
            builder.bitcast(reg_addr, ir.PointerType(target_type)),
            name="fp_reg_val")

        # Advance fp_offset by 16 (XMM slots are 16 bytes apart)
        new_fp = builder.add(fp_offset, ir.Constant(_i32, 16))
        builder.store(new_fp, fp_offset_p)
        builder.branch(end_bb)
        in_reg_final = builder.block

        # ---- in_mem path ----
        builder.position_at_end(in_mem_bb)
        overflow_p = builder.gep(
            va_struct_ptr, [ir.Constant(_i32, 0), ir.Constant(_i32, 2)],
            name="overflow_p")
        overflow = builder.load(overflow_p, name="overflow")
        mem_val = builder.load(
            builder.bitcast(overflow, ir.PointerType(target_type)),
            name="fp_mem_val")

        # Advance overflow by 8
        new_overflow = builder.gep(overflow, [ir.Constant(_i32, 8)],
                                   name="new_overflow")
        builder.store(new_overflow, overflow_p)
        builder.branch(end_bb)
        in_mem_final = builder.block

        # ---- merge ----
        builder.position_at_end(end_bb)
        phi = builder.phi(target_type, name=name or "va.arg")
        phi.add_incoming(reg_val, in_reg_final)
        phi.add_incoming(mem_val, in_mem_final)
        return phi


# =========================================================================
# AArch64 AAPCS (Linux)
# =========================================================================

class AArch64AAPCSVAArgLowering(VAArgLowering):
    """AArch64 AAPCS (Linux/generic) va_arg lowering.

    va_list = { i8* __stack, i8* __gr_top, i8* __vr_top,
                i32 __gr_offs, i32 __vr_offs }

    GP regs saved below __gr_top; __gr_offs starts negative and grows to 0.
    FP/SIMD regs saved below __vr_top; __vr_offs starts negative and grows to 0.
    Stack args live at __stack (bump forward by align(size, 8)).

    Replicates clang's ``AArch64ABIInfo::EmitAAPCSVAArg`` (simplified for
    scalar types pythoc currently supports).
    """

    _va_list_type = ir.LiteralStructType([_i8ptr, _i8ptr, _i8ptr, _i32, _i32])

    def emit_va_start(self, builder) -> Any:
        alloca = builder.alloca(self._va_list_type, name="va_list")
        ap = builder.bitcast(alloca, _i8ptr)
        _call_va_start(builder, ap)
        return ap

    def emit_va_arg(self, builder, va_list_ptr, target_type, name="") -> Any:
        va_struct_ptr = builder.bitcast(
            va_list_ptr, ir.PointerType(self._va_list_type))

        is_fp = _is_fp_type(target_type)
        type_sz = _type_size(target_type)
        reg_size = 16 if is_fp else max(type_sz, 8)
        # Round up to 8-byte multiple for GP
        if not is_fp:
            reg_size = (reg_size + 7) // 8 * 8

        # Offsets:
        #   __stack=0, __gr_top=1, __vr_top=2, __gr_offs=3, __vr_offs=4
        offs_idx = ir.Constant(_i32, 4 if is_fp else 3)
        top_idx = ir.Constant(_i32, 2 if is_fp else 1)

        fn = builder.function

        # Load current offset
        offs_p = builder.gep(
            va_struct_ptr, [ir.Constant(_i32, 0), offs_idx],
            name="offs_p")
        offs = builder.load(offs_p, name="reg_offs")

        # If offset >= 0, already on stack
        on_stack_check = builder.icmp_signed(
            ">=", offs, ir.Constant(_i32, 0), name="using_stack")

        maybe_reg_bb = fn.append_basic_block("vaarg.maybe_reg")
        on_stack_bb = fn.append_basic_block("vaarg.on_stack")
        in_reg_bb = fn.append_basic_block("vaarg.in_reg")
        end_bb = fn.append_basic_block("vaarg.end")

        builder.cbranch(on_stack_check, on_stack_bb, maybe_reg_bb)

        # ---- maybe_reg: try to allocate from register save area ----
        builder.position_at_end(maybe_reg_bb)
        new_offs = builder.add(offs, ir.Constant(_i32, reg_size),
                               name="new_offs")
        builder.store(new_offs, offs_p)

        in_regs = builder.icmp_signed(
            "<=", new_offs, ir.Constant(_i32, 0), name="inreg")
        builder.cbranch(in_regs, in_reg_bb, on_stack_bb)

        # ---- in_reg: fetch from register save area ----
        builder.position_at_end(in_reg_bb)
        top = builder.load(
            builder.gep(va_struct_ptr,
                        [ir.Constant(_i32, 0), top_idx]),
            name="reg_top")
        # reg_top points PAST the save area; offset is negative
        reg_addr = builder.gep(top, [offs], name="reg_addr")
        reg_val = builder.load(
            builder.bitcast(reg_addr, ir.PointerType(target_type)),
            name="reg_val")
        builder.branch(end_bb)
        in_reg_final = builder.block

        # ---- on_stack: fetch from __stack ----
        builder.position_at_end(on_stack_bb)
        stack_p = builder.gep(
            va_struct_ptr, [ir.Constant(_i32, 0), ir.Constant(_i32, 0)],
            name="stack_p")
        stack_ptr = builder.load(stack_p, name="stack")
        stack_val = builder.load(
            builder.bitcast(stack_ptr, ir.PointerType(target_type)),
            name="stack_val")

        # Advance __stack by align(type_size, 8)
        stack_advance = max(type_sz, 8)
        stack_advance = (stack_advance + 7) // 8 * 8
        new_stack = builder.gep(
            stack_ptr, [ir.Constant(_i32, stack_advance)],
            name="new_stack")
        builder.store(new_stack, stack_p)
        builder.branch(end_bb)
        on_stack_final = builder.block

        # ---- merge ----
        builder.position_at_end(end_bb)
        phi = builder.phi(target_type, name=name or "va.arg")
        phi.add_incoming(reg_val, in_reg_final)
        phi.add_incoming(stack_val, on_stack_final)
        return phi


# =========================================================================
# Factory
# =========================================================================

def get_va_arg_lowering(triple: str = None) -> VAArgLowering:
    """Get the appropriate va_arg lowering for the given target triple.

    Args:
        triple: Target triple string. If None, uses llvmlite default.

    Returns:
        A VAArgLowering instance.
    """
    if triple is None:
        from llvmlite import binding
        triple = binding.get_default_triple()

    triple_lower = triple.lower()
    arch = triple.split('-')[0]
    is_windows = (
        'windows' in triple_lower
        or 'win32' in triple_lower
        or 'mingw' in triple_lower
    )

    if is_windows:
        # Both x86_64-windows and aarch64-windows use the simple
        # void-pointer va_list model (MS ABI).
        return VoidPtrVAArgLowering()

    if arch in ('aarch64', 'arm64'):
        return AArch64AAPCSVAArgLowering()

    # x86_64 SysV (Linux, macOS, FreeBSD, ...)
    # Also used as the fallback for unknown architectures.
    return X86_64SysVVAArgLowering()
