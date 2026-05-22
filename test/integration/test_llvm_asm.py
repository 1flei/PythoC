#!/usr/bin/env python3
"""
Test llvm_asm intrinsic: direct LLVM inline assembly emission.

Tests cover:
- Basic void asm (nop, pause, memory barrier)
- Multiple asm calls in one function
- Asm inside control flow (loops, if)
- Asm with input operands
- Asm with output (return value)
- Error cases (non-constant args, missing args)
"""

import sys
import os
import platform
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import unittest
from pythoc.decorators.compile import compile
from pythoc.builtin_entities import void, i32, i64, u64, ptr, llvm_asm
from pythoc.build.output_manager import flush_all_pending_outputs

from test.utils.test_utils import DeferredTestCase


# ============================================================
# Detect current platform for conditional tests
# ============================================================

_arch = platform.machine()
IS_X86_64 = _arch in ('x86_64', 'AMD64', 'x86_64')
IS_AARCH64 = _arch in ('aarch64', 'arm64', 'ARM64')


# ============================================================
# Test functions: basic void asm
# ============================================================

@compile(suffix="asm_nop")
def test_fn_asm_nop() -> i32:
    """Simplest case: emit a NOP instruction."""
    llvm_asm("nop", "")
    return i32(42)


@compile(suffix="asm_memory_barrier")
def test_fn_asm_memory_barrier() -> i32:
    """Empty asm with memory clobber = compiler memory barrier."""
    x: i32 = 10
    llvm_asm("", "~{memory}")
    x = x + 1
    return x


@compile(suffix="asm_no_sideeffect")
def test_fn_asm_no_sideeffect() -> i32:
    """Asm with side_effect=False (optimizer may remove/reorder)."""
    llvm_asm("nop", "", False)
    return i32(1)


# ============================================================
# Test functions: multiple asm calls
# ============================================================

@compile(suffix="asm_multiple")
def test_fn_asm_multiple() -> i32:
    """Multiple asm calls in sequence."""
    llvm_asm("nop", "")
    llvm_asm("nop", "")
    llvm_asm("nop", "")
    return i32(3)


@compile(suffix="asm_in_branch")
def test_fn_asm_in_branch(flag: i32) -> i32:
    """Asm inside conditional branch."""
    result: i32 = 0
    if flag > 0:
        llvm_asm("nop", "")
        result = 1
    else:
        llvm_asm("nop", "")
        result = 2
    return result


@compile(suffix="asm_in_loop")
def test_fn_asm_in_loop(n: i32) -> i32:
    """Asm inside a loop body."""
    i: i32 = 0
    while i < n:
        llvm_asm("", "~{memory}")  # barrier each iteration
        i = i + 1
    return i


# ============================================================
# Platform-specific tests (only compiled on matching arch)
# ============================================================

if IS_X86_64:
    @compile(suffix="asm_pause_x86")
    def test_fn_asm_pause() -> i32:
        """x86 PAUSE instruction for spin loops."""
        llvm_asm("pause", "~{memory}")
        return i32(1)

    @compile(suffix="asm_mfence_x86")
    def test_fn_asm_mfence() -> i32:
        """x86 MFENCE — full memory fence."""
        llvm_asm("mfence", "~{memory}")
        return i32(1)

elif IS_AARCH64:
    @compile(suffix="asm_yield_arm")
    def test_fn_asm_yield() -> i32:
        """ARM YIELD for spin loops."""
        llvm_asm("yield", "~{memory}")
        return i32(1)

    @compile(suffix="asm_dmb_arm")
    def test_fn_asm_dmb() -> i32:
        """ARM DMB ISH — data memory barrier."""
        llvm_asm("dmb ish", "~{memory}")
        return i32(1)


# ============================================================
# Test functions: asm with inputs
# ============================================================

if IS_X86_64:
    @compile(suffix="asm_with_input_x86")
    def test_fn_asm_with_input(val: i32) -> i32:
        """Asm with an input operand (x86 example)."""
        # bsf: bit scan forward — find lowest set bit
        # constraint "r" = any general register
        result: i32 = llvm_asm("bsf $1, $0", "=r,r", i32, val)
        return result


# ============================================================
# Test functions: asm returning a value
# ============================================================

if IS_X86_64:
    @compile(suffix="asm_rdtsc")
    def test_fn_asm_rdtsc() -> u64:
        """Read TSC (time-stamp counter) — returns u64."""
        # rdtsc puts low 32 bits in eax, high in edx
        # =A constraint means edx:eax pair as 64-bit
        return llvm_asm(
            "rdtsc",
            "=A,~{dirflag},~{fpsr},~{flags}",
            u64
        )


# ============================================================
# Error cases (these should produce compile errors)
# ============================================================

# Note: Error tests use the @expect_error pattern from DeferredTestCase.
# Since llvm_asm validates at compile time, these functions should fail
# during the @compile phase.

# These are tested via the DeferredTestCase error mechanism:
# We define them but expect compilation to fail.

# test_fn_asm_no_args: missing arguments
# test_fn_asm_runtime_str: non-constant string

# For now, we test that valid cases work. Error cases require
# the @expect_error infrastructure which catches compile errors.


# ============================================================
# Test class
# ============================================================

class TestLlvmAsm(DeferredTestCase):
    """Tests for the llvm_asm intrinsic."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        flush_all_pending_outputs()

    # --- Basic void asm ---

    def test_asm_nop(self):
        """NOP compiles and function returns correctly."""
        self.assertEqual(test_fn_asm_nop(), 42)

    def test_asm_memory_barrier(self):
        """Memory barrier doesn't affect computation but compiles."""
        self.assertEqual(test_fn_asm_memory_barrier(), 11)

    def test_asm_no_sideeffect(self):
        """Non-side-effect asm compiles (optimizer may elide)."""
        self.assertEqual(test_fn_asm_no_sideeffect(), 1)

    # --- Multiple / control flow ---

    def test_asm_multiple(self):
        """Multiple asm calls in sequence."""
        self.assertEqual(test_fn_asm_multiple(), 3)

    def test_asm_in_branch_true(self):
        """Asm in taken branch."""
        self.assertEqual(test_fn_asm_in_branch(1), 1)

    def test_asm_in_branch_false(self):
        """Asm in else branch."""
        self.assertEqual(test_fn_asm_in_branch(0), 2)

    def test_asm_in_loop(self):
        """Asm inside loop body, N iterations."""
        self.assertEqual(test_fn_asm_in_loop(5), 5)

    def test_asm_in_loop_zero(self):
        """Asm in loop with 0 iterations (loop never entered)."""
        self.assertEqual(test_fn_asm_in_loop(0), 0)

    # --- Platform-specific ---

    @unittest.skipUnless(IS_X86_64, "x86_64 only")
    def test_asm_pause_x86(self):
        """x86 PAUSE compiles and runs."""
        self.assertEqual(test_fn_asm_pause(), 1)

    @unittest.skipUnless(IS_X86_64, "x86_64 only")
    def test_asm_mfence_x86(self):
        """x86 MFENCE compiles and runs."""
        self.assertEqual(test_fn_asm_mfence(), 1)

    @unittest.skipUnless(IS_AARCH64, "aarch64 only")
    def test_asm_yield_arm(self):
        """ARM YIELD compiles and runs."""
        self.assertEqual(test_fn_asm_yield(), 1)

    @unittest.skipUnless(IS_AARCH64, "aarch64 only")
    def test_asm_dmb_arm(self):
        """ARM DMB ISH compiles and runs."""
        self.assertEqual(test_fn_asm_dmb(), 1)

    # --- With inputs ---

    @unittest.skipUnless(IS_X86_64, "x86_64 only")
    def test_asm_with_input(self):
        """Asm with input operand (bsf: find lowest set bit)."""
        # bsf(0b1000) = 3 (bit 3 is lowest set bit)
        result = test_fn_asm_with_input(8)
        self.assertEqual(result, 3)

    @unittest.skipUnless(IS_X86_64, "x86_64 only")
    def test_asm_with_input_power_of_two(self):
        """BSF on various powers of 2."""
        self.assertEqual(test_fn_asm_with_input(1), 0)
        self.assertEqual(test_fn_asm_with_input(2), 1)
        self.assertEqual(test_fn_asm_with_input(4), 2)
        self.assertEqual(test_fn_asm_with_input(16), 4)

    # --- With output ---

    @unittest.skipUnless(IS_X86_64, "x86_64 only")
    def test_asm_rdtsc(self):
        """RDTSC returns a non-zero timestamp."""
        tsc = test_fn_asm_rdtsc()
        self.assertGreater(tsc, 0)

    @unittest.skipUnless(IS_X86_64, "x86_64 only")
    def test_asm_rdtsc_monotonic(self):
        """RDTSC is monotonically increasing."""
        t1 = test_fn_asm_rdtsc()
        t2 = test_fn_asm_rdtsc()
        self.assertGreaterEqual(t2, t1)


if __name__ == '__main__':
    unittest.main()
