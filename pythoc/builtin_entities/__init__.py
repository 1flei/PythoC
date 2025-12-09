"""
Builtin Entities Module for PC Compiler

This module has been refactored into a directory structure:
- base.py: Base classes (BuiltinEntity, BuiltinType, BuiltinFunction) - EXTRACTED
- types.py: Basic type entities (i8-i64, u8-u64, f32, f64, bool, void, ptr) - EXTRACTED
- entities_impl.py: Complex types and functions (array, func, struct, union, sizeof, etc.)
- functions.py: Function entities (placeholder for future extraction)
- qualifiers.py: Type qualifiers (placeholder for future extraction)
- registry.py: Registry functions (placeholder for future extraction)
"""

# Import base classes from base.py
from .base import (
    BuiltinEntity,
    BuiltinEntityMeta,
    BuiltinType,
    BuiltinFunction,
)

# Import basic types from types.py
from .types import (
    # Signed integer types
    i8, i16, i32, i64,
    # Unsigned integer types
    u8, u16, u32, u64,
    # Floating point types (f32, f64 only; f16/bf16/f128 moved to std)
    f32, f64,
    # Special types
    bool, void, ptr,
)

# Extended float types (f16, bf16, f128) are in pythoc.std.float_types
# Import them here for backward compatibility
from ..std.float_types import f16, bf16, f128

# Linear token type
from .linear import linear

# Move intrinsic for linear ownership transfer
from .move import move

# Import complex types split into modules
from .array import array
from .func import func
from .struct import struct
from .union import union
from .enum import enum
from .refined import refined, RefinedType

# Type qualifiers migrated
from .qualifiers import const, static, volatile

# Built-in functions
from .intrinsics import sizeof, nullptr, seq, consume, assume, refine, typeof, char

# Python type wrapper
from .python_type import PythonType, is_python_type, pyconst

# Registry and query functions
from .utility import (
    get_builtin_entity,
    has_builtin_entity,
    is_builtin_type,
    is_builtin_function,
    get_builtin_type_info,
    list_builtin_entities,
    list_builtin_types,
    list_builtin_functions,
    TYPE_MAP,
    TYPE_REGISTRY,
    PC_TYPE_MAP,
    TYPE_SIZES,
    TYPE_ALIGNMENTS,
    SIGNED_INT_TYPES,
    UNSIGNED_INT_TYPES,
    INT_TYPES,
    FLOAT_TYPES,
    NUMERIC_TYPES,
    is_signed_int,
    is_unsigned_int,
    get_type_size,
    get_type_alignment,
    get_llvm_type_by_name
)

__all__ = [
    # Base classes
    'BuiltinEntity',
    'BuiltinEntityMeta',
    'BuiltinType',
    'BuiltinFunction',
    
    # Signed integer types
    'i8', 'i16', 'i32', 'i64',
    
    # Unsigned integer types
    'u8', 'u16', 'u32', 'u64',
    
    # Floating point types
    'f16', 'bf16', 'f32', 'f64', 'f128',
    
    # Boolean type
    'bool',
    
    # Void type
    'void',
    
    # Pointer type
    'ptr',
    
    # Linear token type
    'linear',
    
    # Move intrinsic
    'move',
    
    # Array type
    'array',
    
    # Function type
    'func',
    
    # Struct and union types
    'struct',
    'union',
    'enum',
    'refined',
    'RefinedType',
    
    # Type qualifiers
    'const',
    'static',
    'volatile',
    
    # Built-in functions
    'sizeof',
    'typeof',
    'nullptr',
    'getptr',
    'seq',
    'consume',
    'assume',
    'refine',
    'char',
    
    # Python type wrapper
    'PythonType',
    'is_python_type',
    'pyconst',
    
    # Registry and query functions
    'get_builtin_entity',
    'has_builtin_entity',
    'is_builtin_type',
    'is_builtin_function',
    'get_builtin_type_info',
    'list_builtin_entities',
    'list_builtin_types',
    'list_builtin_functions',
    
    # Type mapping and registry
    'TYPE_MAP',
    'TYPE_REGISTRY',
    'PC_TYPE_MAP',
    'TYPE_SIZES',
    'TYPE_ALIGNMENTS',
    
    # Type sets
    'SIGNED_INT_TYPES',
    'UNSIGNED_INT_TYPES',
    'INT_TYPES',
    'FLOAT_TYPES',
    'NUMERIC_TYPES',
    
    # Type query functions
    'is_signed_int',
    'is_unsigned_int',
    'get_type_size',
    'get_type_alignment',
    'get_llvm_type_by_name',
]
