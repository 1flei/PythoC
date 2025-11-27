# -*- coding: utf-8 -*-
"""
Builtin Entities Registry and Utility Functions

This module provides registry and query functions for builtin entities,
type maps, sizes/alignments, and helpers to get LLVM types and parse annotations.

Content migrated exactly from entities_impl.py (utility sections).
"""

from llvmlite import ir
from typing import Any, Optional
import ast

# Import base classes and basic types from extracted modules
from .base import BuiltinEntity, BuiltinType, BuiltinFunction, BuiltinEntityMeta, _get_unified_registry
from .types import i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool, void, ptr

# Ensure any pending entities from metaclass are registered
BuiltinEntityMeta._register_pending()


# ============================================================================
# Registry and Query Functions
# ============================================================================

def get_builtin_entity(name: str) -> Optional[type]:
    """Get a builtin entity class by name"""
    registry = _get_unified_registry()
    return registry.get_builtin_entity(name)


def has_builtin_entity(name: str) -> bool:
    """Check if a builtin entity exists"""
    registry = _get_unified_registry()
    return registry.has_builtin_entity(name)


def is_builtin_type(name: str) -> bool:
    """Check if name is a builtin type"""
    entity = get_builtin_entity(name)
    return entity and entity.can_be_type()


def is_builtin_function(name: str) -> bool:
    """Check if name is a builtin function"""
    entity = get_builtin_entity(name)
    return entity and entity.can_be_called()


def get_builtin_type_info(name: str) -> Optional[dict]:
    """Get type information for a builtin type"""
    entity = get_builtin_entity(name)
    if entity and entity.can_be_type():
        return {
            'name': entity.get_name(),
            'llvm_type': entity.get_llvm_type(),
            'size_bytes': entity.get_size_bytes(),
            'is_signed': entity.is_signed(),
        }
    return None


def list_builtin_entities() -> list:
    """List all registered builtin entities"""
    registry = _get_unified_registry()
    return registry.list_builtin_entities()


def list_builtin_types() -> list:
    """List all builtin types"""
    registry = _get_unified_registry()
    return registry.list_builtin_types()


def list_builtin_functions() -> list:
    """List all builtin functions"""
    registry = _get_unified_registry()
    return registry.list_builtin_functions()


# ============================================================================
# Type Mapping and Registry
# ============================================================================

# Create TYPE_MAP for backward compatibility
def _build_type_map():
    """Build TYPE_MAP from unified registry"""
    registry = _get_unified_registry()
    type_map = {}
    for name in registry.list_builtin_types():
        entity_cls = registry.get_builtin_entity(name)
        if entity_cls:
            type_map[entity_cls] = name
            type_map[name] = name
    return type_map

TYPE_MAP = _build_type_map()

# Type name to type class mapping (replaces old PC_TYPE_MAP)
def _build_type_registry():
    """Build TYPE_REGISTRY from unified registry"""
    registry = _get_unified_registry()
    type_registry = {}
    for name in registry.list_builtin_types():
        entity_cls = registry.get_builtin_entity(name)
        if entity_cls:
            type_registry[name] = entity_cls
    return type_registry

TYPE_REGISTRY = _build_type_registry()

# Backward compatibility: PC_TYPE_MAP points to TYPE_REGISTRY
PC_TYPE_MAP = TYPE_REGISTRY

# Import extended float types from std
from ..std.float_types import f16, bf16, f128

# Type sets for backward compatibility
SIGNED_INT_TYPES = {i8, i16, i32, i64}
UNSIGNED_INT_TYPES = {u8, u16, u32, u64}
INT_TYPES = SIGNED_INT_TYPES | UNSIGNED_INT_TYPES
FLOAT_TYPES = {f16, bf16, f32, f64, f128}
NUMERIC_TYPES = INT_TYPES | FLOAT_TYPES


def is_signed_int(type_hint):
    """Check if type is a signed integer"""
    if isinstance(type_hint, type) and issubclass(type_hint, BuiltinType):
        return type_hint.is_signed() and type_hint in SIGNED_INT_TYPES
    return type_hint in SIGNED_INT_TYPES


def is_unsigned_int(type_hint):
    """Check if type is an unsigned integer"""
    if isinstance(type_hint, type) and issubclass(type_hint, BuiltinType):
        return not type_hint.is_signed() and type_hint in UNSIGNED_INT_TYPES
    return type_hint in UNSIGNED_INT_TYPES


# ============================================================================
# Unified Type Size and Alignment Mapping
# ============================================================================

# Type sizes mapping (in bytes)
def _build_type_sizes():
    """Build TYPE_SIZES from unified registry"""
    registry = _get_unified_registry()
    sizes = {}
    for name in registry.list_builtin_types():
        entity_cls = registry.get_builtin_entity(name)
        if entity_cls:
            sizes[entity_cls] = entity_cls.get_size_bytes()
            sizes[name] = entity_cls.get_size_bytes()
    return sizes


def _build_type_alignments():
    """Build TYPE_ALIGNMENTS from unified registry"""
    registry = _get_unified_registry()
    alignments = {}
    for name in registry.list_builtin_types():
        entity_cls = registry.get_builtin_entity(name)
        if entity_cls:
            # Alignment = size for basic types
            alignments[entity_cls] = entity_cls.get_size_bytes()
            alignments[name] = entity_cls.get_size_bytes()
    return alignments


TYPE_SIZES = _build_type_sizes()
TYPE_ALIGNMENTS = _build_type_alignments()

# Add Python builtin type mappings
TYPE_SIZES[int] = 8  # Python int maps to i64
TYPE_SIZES[float] = 8  # Python float maps to f64
TYPE_ALIGNMENTS[int] = 8
TYPE_ALIGNMENTS[float] = 8


def get_type_size(type_hint) -> int:
    """
    Get size of a type in bytes.
    """
    # Handle builtin entity classes
    if isinstance(type_hint, type) and issubclass(type_hint, BuiltinEntity):
        if type_hint.can_be_type():
            return type_hint.get_size_bytes()
    
    # Handle string type names
    if isinstance(type_hint, str):
        entity = get_builtin_entity(type_hint)
        if entity and entity.can_be_type():
            return entity.get_size_bytes()
        # Fallback to TYPE_SIZES dict
        return TYPE_SIZES.get(type_hint, 4)
    
    # Handle direct lookup in TYPE_SIZES
    if type_hint in TYPE_SIZES:
        return TYPE_SIZES[type_hint]
    
    # Default fallback
    return 4


def get_type_alignment(type_hint) -> int:
    """
    Get alignment requirement of a type in bytes.
    """
    # Handle builtin entity classes
    if isinstance(type_hint, type) and issubclass(type_hint, BuiltinEntity):
        if type_hint.can_be_type():
            # For basic types, alignment = size (up to 8 bytes)
            return min(type_hint.get_size_bytes(), 8)
    
    # Handle string type names
    if isinstance(type_hint, str):
        entity = get_builtin_entity(type_hint)
        if entity and entity.can_be_type():
            return min(entity.get_size_bytes(), 8)
        # Fallback to TYPE_ALIGNMENTS dict
        return TYPE_ALIGNMENTS.get(type_hint, 4)
    
    # Handle direct lookup in TYPE_ALIGNMENTS
    if type_hint in TYPE_ALIGNMENTS:
        return TYPE_ALIGNMENTS[type_hint]
    
    # Default fallback
    return 4


def get_llvm_type_by_name(type_name: str) -> Optional[ir.Type]:
    """
    Get LLVM type by name.
    """
    entity = get_builtin_entity(type_name)
    if entity and entity.can_be_type():
        return entity.get_llvm_type()
    
    # Handle some common aliases
    if type_name.lower() == 'int':
        return ir.IntType(32)
    elif type_name.lower() in ('float', 'double'):
        return ir.DoubleType()
    
    return None
