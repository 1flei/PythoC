"""
Common utilities for parsing type subscript syntax like struct[...] and union[...]

Handles both:
- Named fields: struct[x: i32, y: f64] -> Tuple of Slice nodes
- Unnamed fields: struct[i32, f64] -> Tuple of type nodes
- Mixed: struct[i32, y: f64] -> Tuple mixing both
"""

import ast
from typing import List, Tuple, Optional, Any


def parse_type_subscript(node: ast.Subscript, type_resolver) -> Tuple[List[Any], Optional[List[str]]]:
    """Parse type subscript syntax for struct/union field definitions.
    
    Args:
        node: AST Subscript node (e.g., struct[i32, y: f64])
        type_resolver: Type resolver to parse type annotations
        
    Returns:
        Tuple of (field_types, field_names)
        - field_types: List of field types
        - field_names: List of field names (None for unnamed, or list with some None entries)
        
    Examples:
        struct[i32, f64] -> ([i32, f64], None)
        struct[x: i32, y: f64] -> ([i32, f64], ['x', 'y'])
        struct[i32, y: f64] -> ([i32, f64], [None, 'y'])
    """
    slice_node = node.slice
    
    # Handle single item (not a tuple)
    if not isinstance(slice_node, ast.Tuple):
        # Single field
        field_type, field_name = _parse_field_item(slice_node, type_resolver)
        field_names = [field_name] if field_name is not None else None
        return ([field_type], field_names)
    
    # Multiple fields (Tuple)
    field_types = []
    field_names = []
    has_any_name = False
    
    for item in slice_node.elts:
        field_type, field_name = _parse_field_item(item, type_resolver)
        field_types.append(field_type)
        field_names.append(field_name)
        if field_name is not None:
            has_any_name = True
    
    # If no fields have names, return None for field_names
    if not has_any_name:
        field_names = None
    
    return (field_types, field_names)


def _parse_field_item(item: ast.AST, type_resolver) -> Tuple[Any, Optional[str]]:
    """Parse a single field item from type subscript.
    
    Args:
        item: AST node (either Slice for named field, or type node for unnamed)
        type_resolver: Type resolver to parse type annotations
        
    Returns:
        Tuple of (field_type, field_name)
        - field_name is None for unnamed fields
        
    Examples:
        i32 -> (i32, None)
        x: i32 -> (i32, 'x')  (Slice node with lower=x, upper=i32)
        "x": i32 -> (i32, 'x')  (Slice node with lower=Constant("x"), upper=i32)
    """
    if isinstance(item, ast.Slice):
        # Named field: x: i32 or "x": i32
        # lower = field name (ast.Name or ast.Constant)
        # upper = field type
        if isinstance(item.lower, ast.Name):
            field_name = item.lower.id
        elif isinstance(item.lower, ast.Constant) and isinstance(item.lower.value, str):
            field_name = item.lower.value
        else:
            raise TypeError(f"Field name must be an identifier or string, got {ast.dump(item.lower)}")
        field_type = type_resolver.parse_annotation(item.upper)
        return (field_type, field_name)
    else:
        # Unnamed field: just the type
        field_type = type_resolver.parse_annotation(item)
        return (field_type, None)


def is_type_subscript(node: ast.Subscript) -> bool:
    """Check if a subscript is a type subscript (e.g., struct[...]) vs value subscript (e.g., arr[0]).
    
    A subscript is considered a type subscript if:
    - slice is a Tuple (multiple items)
    - slice contains at least one Slice node (named field syntax)
    
    Note: This is a heuristic. For single-item subscripts without names,
    we cannot distinguish type vs value subscripts at AST level alone.
    The caller should check if node.value is a type name (struct/union/etc).
    
    Args:
        node: AST Subscript node
        
    Returns:
        True if definitely a type subscript, False if uncertain
    """
    slice_node = node.slice
    
    # Multiple items -> definitely type subscript
    if isinstance(slice_node, ast.Tuple):
        return True
    
    # Single Slice node (named field) -> type subscript
    if isinstance(slice_node, ast.Slice):
        return True
    
    # Single item, not a Slice -> could be either, return False (uncertain)
    return False
