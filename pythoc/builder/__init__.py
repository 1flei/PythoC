"""
Builder abstraction layer for multi-target code generation.

This module provides an abstract interface for code generation backends,
enabling:
1. LLVM IR generation with C ABI (LLVMCBuilder - default, handles struct coercion)
2. LLVM IR without ABI coercion (LLVMSimpleBuilder - thin wrapper)
3. Compile-time evaluation (NullBuilder for type resolution)
4. Future: C code, MLIR, GPU kernels, etc.

For C-compatible code generation (interop with C libraries), use LLVMCBuilder.
For pure LLVM IR without ABI coercion, use LLVMSimpleBuilder.

Note: LLVMBuilder is an alias for LLVMCBuilder for backward compatibility.
"""

from .abstract import AbstractBuilder
from .llvm_simple_builder import LLVMBuilder as LLVMSimpleBuilder
from .llvm_c_builder import LLVMCBuilder, FunctionWrapper

# For backward compatibility, LLVMBuilder refers to LLVMCBuilder (with C ABI support)
# This is because all existing code expects LLVMBuilder to have declare_function,
# set_return_abi_context, etc.
LLVMBuilder = LLVMCBuilder

from .null_builder import NullBuilder

__all__ = [
    'AbstractBuilder',
    'LLVMBuilder',        # Alias for LLVMCBuilder (backward compatible)
    'LLVMCBuilder',       # LLVM builder with C ABI support
    'LLVMSimpleBuilder',  # Simple LLVM builder without ABI coercion
    'FunctionWrapper',
    'NullBuilder',
]
