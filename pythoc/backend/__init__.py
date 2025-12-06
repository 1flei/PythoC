"""
Backend abstraction layer for multi-target code generation.

This module provides an abstract interface for code generation backends,
enabling:
1. LLVM IR generation (current default)
2. Compile-time evaluation (ConstexprBackend for type resolution)
3. Future: C code, MLIR, GPU kernels, etc.

Architecture:
    Backend provides:
    - Builder: IR generation operations (add, load, store, etc.)
    - Module context: Type resolution and global management
    - Variable allocation: Backend-specific storage allocation

    BaseVisitor uses Backend through a clean interface, making it
    backend-agnostic and enabling constexpr evaluation without LLVM.
"""

from .abstract import AbstractBackend
from .llvm_backend import LLVMBackend
from .constexpr_backend import ConstexprBackend

__all__ = ['AbstractBackend', 'LLVMBackend', 'ConstexprBackend']
