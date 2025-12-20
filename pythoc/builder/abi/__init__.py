"""
ABI (Application Binary Interface) handling module.

This module implements C ABI conventions for different target architectures.
The key responsibility is to coerce struct types for function arguments and
return values to match what C compilers (like clang/gcc) expect.

LLVM does NOT automatically handle struct ABI - the frontend must do it.
For example, on x86-64, a struct {i32, i32} must be coerced to i64 for
return values, otherwise the generated code will be ABI-incompatible with C.
"""

from .x86_64 import X86_64ABI
from .base import ABIInfo, CoercedType

__all__ = ['ABIInfo', 'CoercedType', 'X86_64ABI', 'get_target_abi']


def get_target_abi(triple: str = None) -> ABIInfo:
    """Get ABI handler for target triple.
    
    Args:
        triple: Target triple (e.g., 'x86_64-unknown-linux-gnu')
                If None, uses default triple from llvmlite.
    
    Returns:
        ABIInfo instance for the target.
    """
    if triple is None:
        from llvmlite import binding
        triple = binding.get_default_triple()
    
    triple_lower = triple.lower()
    
    # Parse architecture from triple
    arch = triple.split('-')[0] if triple else 'x86_64'
    
    # Check if Windows target
    is_windows = 'windows' in triple_lower or 'win32' in triple_lower or 'mingw' in triple_lower
    
    if arch in ('x86_64', 'x86-64', 'amd64'):
        # Windows x64 ABI: max 8 bytes in registers
        # System V ABI (Linux/macOS): max 16 bytes in registers
        max_reg_size = 8 if is_windows else 16
        return X86_64ABI(max_register_size=max_reg_size)
    elif arch in ('aarch64', 'arm64'):
        # TODO: Implement AArch64 ABI
        from .aarch64 import AArch64ABI
        return AArch64ABI()
    else:
        # Default to x86_64 for now
        return X86_64ABI()
