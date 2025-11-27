# -*- coding: utf-8 -*-
"""
Type identity system for generating unique type identifiers.

This module provides a canonical way to identify types for function mangling.
Two types are considered identical if and only if they produce the same type ID.

Design: Each type implements its own get_type_id() classmethod.
"""

from typing import Any


def get_type_id(pc_type: Any) -> str:
    """
    Get the unique type ID for a PC type.
    
    Delegates to the type's get_type_id() method if available.
    For LLVM IR types, looks up the corresponding Python class in registry.
    
    Returns a compact string that uniquely identifies the type.
    """
    if pc_type is None:
        return 'v'
    
    # Delegate to type's own get_type_id() method
    if hasattr(pc_type, 'get_type_id'):
        return pc_type.get_type_id()
    
    # Handle LLVM IR types by looking up Python class in registry
    from llvmlite import ir
    if isinstance(pc_type, ir.IdentifiedStructType):
        from .registry import get_unified_registry
        registry = get_unified_registry()
        struct_info = registry.get_struct(pc_type.name)
        if struct_info and struct_info.python_class:
            return struct_info.python_class.get_type_id()
        # No Python class found, use LLVM type name
        return f'{len(pc_type.name)}{pc_type.name}'
    
    raise TypeError(f"Type {pc_type} (type={type(pc_type)}) does not have a get_type_id() method")
