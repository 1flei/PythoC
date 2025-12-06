"""
Builder abstraction layer for multi-target code generation.

This module provides an abstract interface for code generation backends,
enabling:
1. LLVM IR generation (current default)
2. Compile-time evaluation (NullBuilder for type resolution)
3. Future: C code, MLIR, GPU kernels, etc.
"""

from .abstract import AbstractBuilder
from .llvm_builder import LLVMBuilder
from .null_builder import NullBuilder

__all__ = ['AbstractBuilder', 'LLVMBuilder', 'NullBuilder']
