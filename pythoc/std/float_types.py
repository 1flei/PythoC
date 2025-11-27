"""
Extended Floating Point Types

These types require libgcc for software floating point support.
"""

from ..builtin_entities.base import BuiltinType
from llvmlite import ir

# Import LLVM extensions to enable BFloatType and FP128Type
try:
    from .. import llvm_extensions
except ImportError:
    pass


class f16(BuiltinType):
    """16-bit floating point type (IEEE 754 half precision)"""
    _llvm_type = ir.HalfType()
    _size_bytes = 2
    _is_signed = True
    _is_float = True
    _lib_registered = False
    
    @classmethod
    def get_name(cls) -> str:
        return 'f16'
    
    @classmethod
    def get_llvm_type(cls, module_context=None) -> ir.Type:
        """Get LLVM type and register library dependency"""
        if not cls._lib_registered:
            from ..registry import get_unified_registry
            get_unified_registry().add_link_library('gcc_s')
            cls._lib_registered = True
        return cls._llvm_type


class bf16(BuiltinType):
    """16-bit brain floating point type (bfloat16)
    
    Uses monkey-patched BFloatType if available, otherwise falls back to HalfType.
    LLVM natively supports bfloat starting from LLVM 11.
    """
    _size_bytes = 2
    _is_signed = True
    _is_float = True
    _lib_registered = False
    
    @classmethod
    def get_name(cls) -> str:
        return 'bf16'
    
    @classmethod
    def get_llvm_type(cls, module_context=None) -> ir.Type:
        """Get LLVM type for bf16 and register library dependency"""
        if not cls._lib_registered:
            from ..registry import get_unified_registry
            get_unified_registry().add_link_library('gcc_s')
            cls._lib_registered = True
        return ir.BFloatType()


# Set _llvm_type after class definition (will call get_llvm_type once)
bf16._llvm_type = ir.BFloatType()


class f128(BuiltinType):
    """128-bit floating point type (IEEE 754 quad precision)
    
    Uses monkey-patched FP128Type if available, otherwise falls back to DoubleType.
    LLVM natively supports fp128 for quad precision.
    """
    _size_bytes = 16
    _is_signed = True
    _is_float = True
    _lib_registered = False
    
    @classmethod
    def get_name(cls) -> str:
        return 'f128'
    
    @classmethod
    def get_llvm_type(cls, module_context=None) -> ir.Type:
        """Get LLVM type for f128 and register library dependency"""
        if not cls._lib_registered:
            from ..registry import get_unified_registry
            get_unified_registry().add_link_library('gcc_s')
            cls._lib_registered = True
        return ir.FP128Type()


# Set _llvm_type after class definition
f128._llvm_type = ir.FP128Type()

