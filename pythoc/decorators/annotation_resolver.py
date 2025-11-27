# -*- coding: utf-8 -*-
"""
Unified annotation resolver for handling string annotations from __future__ import annotations
"""

import inspect
from typing import Any, Dict, Optional


def build_annotation_namespace(user_globals: dict, is_dynamic: bool = False, 
                               additional_types: Optional[Dict[str, Any]] = None) -> dict:
    """
    Build a namespace for evaluating string type annotations.
    
    This handles both regular annotations and annotations from `from __future__ import annotations`.
    For dynamic functions/classes (defined inside other functions), it attempts to capture
    local variables from the enclosing scope.
    
    Args:
        user_globals: The global namespace of the module where the function/class is defined
        is_dynamic: Whether this is a dynamic function/class (has '<locals>' in __qualname__)
        additional_types: Additional type mappings to include in the namespace
    
    Returns:
        A dictionary that can be used as the namespace for eval() or TypeResolver
    """
    eval_namespace = dict(user_globals)
    
    # Add any additional type mappings
    if additional_types:
        eval_namespace.update(additional_types)
    
    # For dynamic functions/classes, try to capture enclosing scope variables
    if is_dynamic:
        try:
            # Walk up the call stack to find frames with local variables
            frame = inspect.currentframe()
            # Walk up several frames to get past our decorator stack
            for _ in range(10):
                if frame is None:
                    break
                frame = frame.f_back
                if frame and frame.f_locals:
                    # Merge frame locals into eval namespace
                    # Filter to avoid polluting namespace with internal variables
                    for key, value in frame.f_locals.items():
                        # Skip specific internal names but allow _ClassName for nested structs
                        if key in ['self', 'cls', 'func', 'wrapper']:
                            continue
                        
                        # Include type-like objects (for type annotations)
                        # AND integer constants (for array dimensions, etc.)
                        # AND struct classes (even those starting with _)
                        if (hasattr(value, '__name__') or 
                            hasattr(value, 'can_be_type') or 
                            isinstance(value, int) or
                            (isinstance(value, type) and hasattr(value, '_is_struct'))):
                            eval_namespace[key] = value
        except:
            pass  # If we can't get the frame, just use globals
    
    return eval_namespace


def resolve_string_annotation(annotation_str: str, eval_namespace: dict, 
                              type_resolver=None) -> Any:
    """
    Resolve a string type annotation to an actual type.
    
    Tries multiple strategies:
    1. TypeResolver parsing (handles complex types like ptr[T] properly)
    2. Direct eval in the provided namespace (for simple names)
    
    Args:
        annotation_str: The string annotation to resolve
        eval_namespace: The namespace to use for eval
        type_resolver: Optional TypeResolver instance for parsing
    
    Returns:
        The resolved type, or the original string if resolution fails
    """
    from ..builtin_entities import BuiltinEntity
    
    # Strategy 1: Try TypeResolver first (it handles nested types correctly)
    # This ensures that ptr['ListNode'] gets properly resolved with ListNode looked up
    if type_resolver:
        try:
            parsed = type_resolver.parse_annotation(annotation_str)
            if parsed is not None:
                return parsed
        except:
            pass
    
    # Strategy 2: Try direct eval as fallback
    try:
        evaled = eval(annotation_str, eval_namespace)
        # Check if it's a valid PC type
        if isinstance(evaled, type) and issubclass(evaled, BuiltinEntity):
            if evaled.can_be_type():
                return evaled
        # Also accept non-BuiltinEntity types (might be struct classes)
        return evaled
    except:
        pass
    
    # If all strategies fail, return the string (for lazy resolution)
    return annotation_str


def resolve_annotations_dict(annotations: dict, eval_namespace: dict, 
                             type_resolver=None) -> dict:
    """
    Resolve all annotations in a __annotations__ dictionary.
    
    Args:
        annotations: The __annotations__ dictionary from a function or class
        eval_namespace: The namespace to use for eval
        type_resolver: Optional TypeResolver instance for fallback parsing
    
    Returns:
        Dictionary mapping annotation names to resolved types
    """
    from ..builtin_entities import BuiltinEntity
    
    resolved = {}
    
    for name, annotation_value in annotations.items():
        # Handle direct BuiltinEntity types (already evaluated)
        if isinstance(annotation_value, type) and issubclass(annotation_value, BuiltinEntity):
            if annotation_value.can_be_type():
                resolved[name] = annotation_value
                continue
        
        # Handle string annotations (from __future__ import annotations)
        if isinstance(annotation_value, str):
            resolved[name] = resolve_string_annotation(
                annotation_value, eval_namespace, type_resolver
            )
        else:
            # Keep non-string, non-BuiltinEntity values as-is
            resolved[name] = annotation_value
    
    return resolved
