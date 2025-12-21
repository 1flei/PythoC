"""
Varargs detection for PC functions.

Detects *args type annotations and determines how they should be handled:
- struct varargs: compile-time expansion into individual parameters
- union/enum/none varargs: LLVM va_list runtime handling
"""

import ast
from typing import List, Tuple, Optional, Any


def detect_varargs(func_node: ast.FunctionDef, type_resolver) -> Tuple[str, Optional[List[Any]], Optional[str]]:
    """Detect if function has *args with type annotation.
    
    Args:
        func_node: Function AST node
        type_resolver: TypeResolver for resolving type annotations
    
    Returns:
        Tuple of (kind, element_types, param_name):
        - kind: 'struct', 'union', 'enum', or 'none'
        - element_types: List of type annotations (AST nodes or PC types)
        - param_name: Name of the *args parameter (e.g., 'args')
    
    Examples:
        def f(*args: struct[i32, f64]) -> void
        Returns: ('struct', [ast.Name('i32'), ast.Name('f64')], 'args')
        
        def f(*args: union[i32, f64]) -> void
        Returns: ('union', [ast.Name('i32'), ast.Name('f64')], 'args')
        
        def f(*args) -> void
        Returns: ('none', None, 'args')
    """
    if not func_node.args.vararg:
        return ('none', None, None)
    
    vararg = func_node.args.vararg
    if not vararg.annotation:
        return ('none', None, vararg.arg)
    
    annotation = vararg.annotation
    parsed_type = type_resolver.parse_annotation(annotation)
    
    from ..builtin_entities.struct import StructType
    from ..builtin_entities.union import UnionType
    from ..builtin_entities.enum import EnumType
    
    if hasattr(parsed_type, '__origin__'):
        base_type = parsed_type.__origin__
    else:
        base_type = type(parsed_type) if not isinstance(parsed_type, type) else parsed_type
    
    kind = 'union'
    if isinstance(base_type, type):
        try:
            if issubclass(base_type, StructType) or getattr(base_type, '_is_struct', False):
                kind = 'struct'
            elif issubclass(base_type, UnionType) or getattr(base_type, '_is_union', False):
                kind = 'union'
            elif issubclass(base_type, EnumType) or getattr(base_type, '_is_enum', False):
                kind = 'enum'
        except TypeError:
            pass
    
    if isinstance(annotation, ast.Subscript):
        slice_node = annotation.slice
        if isinstance(slice_node, ast.Tuple):
            element_types = list(slice_node.elts)
        else:
            element_types = [slice_node]
    elif kind == 'struct' and (hasattr(parsed_type, '_struct_fields') or hasattr(parsed_type, '_field_types')):
        element_types = []
    else:
        element_types = [annotation]
    
    return (kind, element_types, vararg.arg)
