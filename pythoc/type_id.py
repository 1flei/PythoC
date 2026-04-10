# -*- coding: utf-8 -*-
"""
Type identity system for generating unique type identifiers.

This module provides a canonical way to identify types for function mangling.
Two types are considered identical if and only if they produce the same type ID.

Design: Each type implements its own get_type_id() classmethod.
"""

from typing import Any, Optional, Set


def _recursive_type_token(pc_type: Any) -> str:
    type_name = getattr(pc_type, '_canonical_name', None) or getattr(pc_type, '__name__', None)
    if not type_name and hasattr(pc_type, 'get_name'):
        try:
            type_name = pc_type.get_name()
        except Exception:
            type_name = None
    if not type_name:
        type_name = type(pc_type).__name__
    return f'R{len(type_name)}{type_name}'


def get_type_id(pc_type: Any, _visited: Optional[Set[int]] = None) -> str:
    """
    Get the unique type ID for a PC type.
    
    Delegates to the type's get_type_id() method if available.
    For LLVM IR types, looks up the corresponding Python class in registry.
    
    Returns a compact string that uniquely identifies the type.
    """
    if _visited is None:
        _visited = set()

    if pc_type is None:
        return 'v'

    type_key = id(pc_type)
    if type_key in _visited:
        return _recursive_type_token(pc_type)
    
    if hasattr(pc_type, 'get_type_id'):
        _visited.add(type_key)
        try:
            return pc_type.get_type_id(_visited)
        finally:
            _visited.remove(type_key)
    
    from llvmlite import ir
    if isinstance(pc_type, ir.IdentifiedStructType):
        from .registry import get_unified_registry
        registry = get_unified_registry()
        struct_info = registry.get_struct(pc_type.name)
        if struct_info and struct_info.python_class:
            return get_type_id(struct_info.python_class, _visited)
        return f'{len(pc_type.name)}{pc_type.name}'
    
    raise TypeError(f"Type {pc_type} (type={type(pc_type)}) does not have a get_type_id() method")
