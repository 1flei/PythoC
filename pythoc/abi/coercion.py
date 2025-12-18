"""
ABI coercion helpers for struct values.

This module provides functions to:
1. Pack a struct value into a coerced type (for returns)
2. Unpack a coerced value back to a struct (at call sites)
3. Unpack coerced parameters back to struct (at function entry)
"""

from typing import Optional
from llvmlite import ir

from .base import CoercedType, PassingKind


def pack_struct_for_return(builder: ir.IRBuilder, struct_value: ir.Value,
                          coercion: CoercedType) -> ir.Value:
    """Pack a struct value into the coerced return type.
    
    For example, {i32, i32} -> i64
    
    Args:
        builder: LLVM IR builder
        struct_value: The struct value to pack
        coercion: ABI coercion info
        
    Returns:
        The packed value in the coerced type
    """
    if not coercion.needs_coercion:
        return struct_value
    
    if coercion.is_indirect:
        # sret: caller provides pointer, we store to it
        # This is handled differently - the function signature changes
        raise ValueError("Indirect (sret) returns should be handled at function level")
    
    original_type = coercion.original_type
    coerced_type = coercion.coerced_type
    
    # Strategy: alloca original struct, store value, bitcast pointer, load as coerced
    # This works because memory layout matches
    
    # Allocate space for the struct
    alloca = builder.alloca(original_type, name="struct.coerce")
    builder.store(struct_value, alloca)
    
    # Load as the coerced type
    # For opaque pointers, we can load any type from any pointer
    # Cast pointer to coerced type pointer first
    coerced_ptr = builder.bitcast(alloca, ir.PointerType(coerced_type), name="coerce.ptr")
    coerced_value = builder.load(coerced_ptr, name="coerced")
    
    return coerced_value


def unpack_struct_from_return(builder: ir.IRBuilder, coerced_value: ir.Value,
                             coercion: CoercedType) -> ir.Value:
    """Unpack a coerced return value back to a struct.
    
    For example, i64 -> {i32, i32}
    
    Args:
        builder: LLVM IR builder
        coerced_value: The coerced value from function call
        coercion: ABI coercion info
        
    Returns:
        The unpacked struct value
    """
    if not coercion.needs_coercion:
        return coerced_value
    
    if coercion.is_indirect:
        # sret: we already have the struct in memory
        raise ValueError("Indirect (sret) returns should be handled at call site")
    
    original_type = coercion.original_type
    coerced_type = coercion.coerced_type
    
    # Strategy: alloca coerced type, store value, load as original struct
    # This works because memory layout matches
    
    # Allocate space for the coerced value
    alloca = builder.alloca(coerced_type, name="coerce.unpack")
    builder.store(coerced_value, alloca)
    
    # Load as the original struct type
    # Cast pointer to struct type pointer first
    struct_ptr = builder.bitcast(alloca, ir.PointerType(original_type), name="struct.ptr")
    struct_value = builder.load(struct_ptr, name="struct.unpacked")
    
    return struct_value


def create_sret_alloca(builder: ir.IRBuilder, struct_type: ir.Type) -> ir.AllocaInstr:
    """Create an alloca for sret (indirect return).
    
    Args:
        builder: LLVM IR builder (should be positioned at function entry)
        struct_type: The struct type being returned
        
    Returns:
        Alloca instruction for the sret buffer
    """
    return builder.alloca(struct_type, name="sret.buf")


def pack_struct_for_argument(builder: ir.IRBuilder, struct_value: ir.Value,
                            coercion: CoercedType) -> ir.Value:
    """Pack a struct value into the coerced argument type.
    
    For example, {i32, i32} -> i64 when passing as argument
    
    Args:
        builder: LLVM IR builder
        struct_value: The struct value to pack (may be value or pointer)
        coercion: ABI coercion info
        
    Returns:
        The packed value in the coerced type
    """
    if not coercion.needs_coercion:
        return struct_value
    
    original_type = coercion.original_type
    
    # If struct_value is a pointer to the aggregate, load it first
    # This handles the case where we have an address-kind ValueRef
    if isinstance(struct_value.type, ir.PointerType):
        pointee = struct_value.type.pointee
        if pointee == original_type:
            # It's a pointer to the aggregate, load it
            struct_value = builder.load(struct_value, name="arg.load")
        elif isinstance(pointee, ir.PointerType):
            # Double pointer - load once to get the pointer, then load again
            struct_value = builder.load(struct_value, name="arg.load.ptr")
            if struct_value.type == ir.PointerType(original_type):
                struct_value = builder.load(struct_value, name="arg.load.val")
    
    if coercion.is_indirect:
        # byval: pass pointer to copy
        # Allocate and store, return pointer
        alloca = builder.alloca(original_type, name="arg.byval")
        builder.store(struct_value, alloca)
        return alloca
    
    coerced_type = coercion.coerced_type
    
    # Strategy: alloca original struct, store value, bitcast pointer, load as coerced
    alloca = builder.alloca(original_type, name="arg.coerce")
    builder.store(struct_value, alloca)
    
    coerced_ptr = builder.bitcast(alloca, ir.PointerType(coerced_type), name="arg.coerce.ptr")
    coerced_value = builder.load(coerced_ptr, name="arg.coerced")
    
    return coerced_value


def unpack_coerced_parameter(builder: ir.IRBuilder, coerced_value: ir.Value,
                            original_type: ir.Type, coerced_type: ir.Type,
                            is_byval: bool) -> ir.Value:
    """Unpack a coerced parameter back to its original struct type.
    
    This is used at function entry to convert ABI-coerced parameters
    back to their original struct types.
    
    For example:
    - i64 -> {i32, i32} (small struct coercion)
    - ptr -> load from ptr (byval)
    
    Args:
        builder: LLVM IR builder
        coerced_value: The coerced parameter value
        original_type: The original struct type
        coerced_type: The coerced type (i64, {i64, i64}, or ptr)
        is_byval: Whether this is a byval parameter (pointer to copy)
        
    Returns:
        The unpacked struct value
    """
    if is_byval:
        # byval: coerced_value is a pointer to the struct, just load it
        return builder.load(coerced_value, name="param.byval.load")
    
    # Coerced: alloca coerced type, store value, bitcast, load as struct
    alloca = builder.alloca(coerced_type, name="param.coerce.unpack")
    builder.store(coerced_value, alloca)
    
    struct_ptr = builder.bitcast(alloca, ir.PointerType(original_type), name="param.struct.ptr")
    struct_value = builder.load(struct_ptr, name="param.struct")
    
    return struct_value
