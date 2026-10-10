#!/usr/bin/env python3
"""CPU SIMD builtin smoke tests."""

import ctypes
import platform
import unittest

from llvmlite import binding

from pythoc import (
    compile,
    f32,
    i32,
    ptr,
    simd,
    simd_bf16_dot,
    simd_bf16_dot_packed,
    simd_f32_to_bf16,
    simd_fma,
    simd_load,
    simd_reduce_add,
    simd_reduce_max,
    simd_store,
    u16,
    u32,
)
from pythoc.compiler import target_machine_identity

_MACHINE = platform.machine().lower()
_HOST_FEATURES = binding.get_host_cpu_features()
_TARGET_FEATURES = target_machine_identity()[1]
HAS_AVX512_BF16 = (
    _MACHINE in {"x86_64", "amd64"}
    and bool(_HOST_FEATURES.get("avx512bf16", False))
    and "+avx512bf16" in _TARGET_FEATURES.split(",")
)


@compile
def test_simd_fma_reduce() -> i32:
    v: simd[f32, 4] = simd[f32, 4](f32(2.0))
    one: simd[f32, 4] = simd[f32, 4](f32(1.0))
    out: simd[f32, 4] = simd_fma(v, v, one)
    return i32(simd_reduce_add(out))


@compile
def test_simd_int_reduce_and_extract() -> i32:
    v: simd[i32, 4] = simd[i32, 4](3)
    return simd_reduce_add(v) + v[2]


@compile
def test_simd_load_max_and_unsigned_div(values: ptr[u32]) -> u32:
    v: simd[u32, 4] = simd_load(values, simd[u32, 4])
    half: simd[u32, 4] = v / u32(2)
    return simd_reduce_max(half)


if HAS_AVX512_BF16:
    @compile
    def test_simd_bf16_dot() -> i32:
        x0: simd[f32, 16] = simd[f32, 16](f32(1.0))
        x1: simd[f32, 16] = simd[f32, 16](f32(1.0))
        w: simd[u16, 32] = simd[u16, 32](u16(0x3f80))
        acc: simd[f32, 16] = simd[f32, 16](f32(0.0))
        out: simd[f32, 16] = simd_bf16_dot(acc, x0, x1, w)
        return i32(simd_reduce_add(out))


    @compile
    def test_simd_bf16_dot_packed() -> i32:
        x0: simd[f32, 16] = simd[f32, 16](f32(1.0))
        x1: simd[f32, 16] = simd[f32, 16](f32(1.0))
        xb: simd[u16, 32] = simd_f32_to_bf16(x0, x1)
        w: simd[u16, 32] = simd[u16, 32](u16(0x3f80))
        acc: simd[f32, 16] = simd[f32, 16](f32(0.0))
        out: simd[f32, 16] = simd_bf16_dot_packed(acc, xb, w)
        return i32(simd_reduce_add(out))


@compile
def test_simd_store(out: ptr[f32]) -> None:
    v: simd[f32, 4] = simd[f32, 4](f32(3.0))
    simd_store(out, v)


class TestCpuSimd(unittest.TestCase):
    def test_fma_reduce(self):
        self.assertEqual(test_simd_fma_reduce(), 20)

    def test_int_reduce_and_extract(self):
        self.assertEqual(test_simd_int_reduce_and_extract(), 15)

    def test_load_max_and_unsigned_div(self):
        values = (ctypes.c_uint32 * 4)(2, 8, 0xFFFFFFFE, 4)
        result = test_simd_load_max_and_unsigned_div(
            ctypes.addressof(values))
        self.assertEqual(result, 0x7FFFFFFF)

    @unittest.skipUnless(HAS_AVX512_BF16, "AVX-512 BF16 required")
    def test_bf16_dot(self):
        self.assertEqual(test_simd_bf16_dot(), 32)

    @unittest.skipUnless(HAS_AVX512_BF16, "AVX-512 BF16 required")
    def test_bf16_dot_packed(self):
        self.assertEqual(test_simd_bf16_dot_packed(), 32)

    def test_store(self):
        out = (ctypes.c_float * 4)()
        test_simd_store(ctypes.addressof(out))
        self.assertEqual(list(out), [3.0, 3.0, 3.0, 3.0])


if __name__ == "__main__":
    unittest.main()
