"""
LLVM IR type extensions for llvmlite

This module provides monkey-patching to add missing float types (bfloat, fp128)
to llvmlite by creating custom type classes that generate the correct LLVM IR.
"""

from llvmlite import ir
from llvmlite.ir import types
import struct


def _format_float_as_hex(value, format_char, pack_char, hex_width):
    """
    Format a float value as hexadecimal IEEE representation.
    Borrowed from llvmlite's internal implementation.
    
    Args:
        value: The float value to format
        format_char: Format character for string conversion (e.g., 'd', 'f')
        pack_char: Struct pack character (e.g., 'Q' for double, 'I' for float, 'H' for half)
        hex_width: Width of hex output in characters
    """
    # Convert to float if not already
    try:
        fval = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"Cannot convert {value} to float")
    
    # Pack as binary and unpack as integer
    packed = struct.pack(f'<{format_char}', fval)
    int_val = struct.unpack(f'<{pack_char}', packed)[0]
    
    # Format as hex with appropriate width
    return f'0x{int_val:0{hex_width}x}'


def _format_double(value):
    """Format value as IEEE 754 double precision hex (64-bit)"""
    return _format_float_as_hex(value, 'd', 'Q', 16)


def _format_bfloat(value):
    """Format value as bfloat16 hex (16-bit)
    
    Bfloat16 uses 'R' prefix in LLVM IR and is 16-bit (1 sign + 8 exponent + 7 mantissa).
    It truncates the mantissa of float32.
    """
    # Convert to float and pack as 32-bit float
    fval = float(value)
    packed = struct.pack('<f', fval)
    int_val = struct.unpack('<I', packed)[0]
    
    # Bfloat16 is the top 16 bits of float32
    bfloat_int = (int_val >> 16) & 0xFFFF
    
    # Format with 'R' prefix for bfloat
    return f'0xR{bfloat_int:04x}'


def _format_half(value):
    """Format value as IEEE 754 half precision hex (16-bit)
    
    Half precision uses 'H' prefix in LLVM IR and is 16-bit (1 sign + 5 exponent + 10 mantissa).
    """
    # For half precision, we need to manually convert
    fval = float(value)
    # Pack as float first, then manually convert to half
    # This is a simplified version - for full IEEE 754 half, use numpy or manual conversion
    # For now, reuse double format (llvmlite does this for half too)
    return _format_double(fval)


class BFloatType(types.Type):
    """Brain Float 16-bit floating point type (bfloat16)
    
    LLVM natively supports bfloat starting from LLVM 11.
    This type generates 'bfloat' in LLVM IR.
    Bfloat16 uses the same exponent width as float32 but with reduced mantissa.
    """
    null = "0.0"
    intrinsic_name = "bf16"
    
    def __str__(self):
        return "bfloat"
    
    def format_constant(self, val):
        """Format a constant value as hex with 'R' prefix for bfloat"""
        return _format_bfloat(val)


class FP128Type(types.Type):
    """128-bit floating point type (quad precision)
    
    LLVM supports fp128 for IEEE 754 quad precision.
    This type generates 'fp128' in LLVM IR.
    """
    null = "0.0"
    intrinsic_name = "f128"
    
    def __str__(self):
        return "fp128"
    
    def format_constant(self, val):
        """Format a constant value as hex for fp128
        
        fp128 constants in LLVM IR use the format: 0xL<32_hex_digits>
        For simplicity, we pad the double representation to 128-bit width.
        """
        # Get the 64-bit hex representation
        hex_val = _format_double(val)
        # fp128 uses 'L' prefix and needs 32 hex digits (128 bits)
        # Pad the 64-bit value to 128 bits by adding zeros
        hex_digits = hex_val[2:]  # Remove '0x' prefix
        padded = hex_digits.ljust(32, '0')  # Pad to 32 hex digits
        return f'0xL{padded}'


def patch_llvmlite():
    """Monkey patch llvmlite to add BFloatType and FP128Type
    
    This adds the missing float types to the ir module so they can
    be used like ir.BFloatType() and ir.FP128Type().
    """
    # Add to ir module
    ir.BFloatType = BFloatType
    ir.FP128Type = FP128Type
    
    # Also add factory functions (singleton pattern like other types)
    _bfloat_singleton = BFloatType()
    _fp128_singleton = FP128Type()
    
    # Override the class to return singleton
    def _bfloat_new(cls):
        return _bfloat_singleton
    
    def _fp128_new(cls):
        return _fp128_singleton
    
    BFloatType.__new__ = staticmethod(_bfloat_new)
    FP128Type.__new__ = staticmethod(_fp128_new)


# Auto-patch on import
patch_llvmlite()
