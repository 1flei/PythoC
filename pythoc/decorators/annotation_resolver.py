# -*- coding: utf-8 -*-
"""
Unified annotation resolver for handling string annotations from __future__ import annotations
"""

from typing import Any, Dict, Optional


def build_annotation_namespace(user_globals: dict, is_dynamic: bool = False, 
                               additional_types: Optional[Dict[str, Any]] = None,
                               definition=None) -> dict:
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
    from .visible import AccessibleSymbols
    live = getattr(user_globals, 'live_globals', None)
    eval_namespace = AccessibleSymbols(
        user_globals, live_globals=live if live is not None else user_globals)
    
    # Add any additional type mappings
    if additional_types:
        eval_namespace.update(additional_types)
    
    if is_dynamic and not isinstance(user_globals, AccessibleSymbols):
        from ..type_resolver import TypeResolver
        captured = TypeResolver.capture_type_namespace(definition=definition)
        eval_namespace.update(captured)
        eval_namespace.type_scope = captured.type_scope
        eval_namespace.type_locals = captured.type_locals

    return eval_namespace


def resolve_annotations_dict(annotations: dict, type_resolver) -> dict:
    """
    Resolve all annotations in a __annotations__ dictionary.

    Args:
        annotations: The __annotations__ dictionary from a function or class
        type_resolver: TypeResolver instance used for string annotation resolution
    
    Returns:
        Dictionary mapping annotation names to resolved types
    """
    return {
        name: (type_resolver.parse_annotation(value, defer_unresolved=True)
               if isinstance(value, str) else value)
        for name, value in annotations.items()
    }
