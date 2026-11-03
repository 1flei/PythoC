# -*- coding: utf-8 -*-
"""
Type identity system for generating unique type identifiers.

This module provides a canonical way to identify types for function mangling.
Two types are considered identical if and only if they produce the same type ID.

The public ID is a lossless projection of authoritative Python type objects.
Individual type methods are compatibility entry points and delegate here.
"""

import weakref
from typing import Any, Optional, Set

# Only scalar leaves can be cached across calls. Composite Python type
# objects can gain fields or resolve forward names after their first use, so
# an object-keyed cache would silently return an obsolete identity. The graph
# projection itself processes each reachable node once per call.
_type_id_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

# Bumped whenever a computation emits a name token for a *currently
# unresolved* forward reference (ptr.get_type_id is the only producer).
# Such a result must not be memoized: the name may be defined later in the
# session, changing the answer.
_name_token_misses = 0


def note_unresolved_name_token():
    """Record that a just-computed type ID used an unresolved forward-ref
    name token, so get_type_id must not memoize the enclosing computation."""
    global _name_token_misses
    _name_token_misses += 1


def _cache_lookup(pc_type: Any) -> Optional[str]:
    try:
        return _type_id_cache.get(pc_type)
    except TypeError:
        # Key does not support weak references (e.g. None, int)
        return None


def _cache_store(pc_type: Any, result: str):
    try:
        _type_id_cache[pc_type] = result
    except TypeError:
        # Key does not support weak references; skip caching
        pass


def _cacheable_leaf(pc_type: Any) -> bool:
    if pc_type is None or not isinstance(pc_type, type):
        return False
    if (getattr(pc_type, 'qualified_type', None) is not None
            or getattr(pc_type, '_is_refined', False)
            or getattr(pc_type, 'param_types', None) is not None
            or getattr(pc_type, 'pointee_type', None) is not None
            or getattr(pc_type, 'dimensions', None) is not None
            or getattr(pc_type, 'lanes', None) is not None
            or getattr(pc_type, '_field_types', None) is not None):
        return False
    from .builtin_entities import types as builtin_types
    return vars(builtin_types).get(pc_type.__name__) is pc_type


def get_type_id(pc_type: Any, _visited: Optional[Set[int]] = None) -> str:
    """
    Get the unique type ID for a PC type.

    Return a lossless, unambiguous descriptor encoding. Backend
    representations and short hashes are not PythoC type identities.
    """
    from .type_graph import validation_in_progress
    validating = validation_in_progress()
    cacheable = _cacheable_leaf(pc_type)
    cached = None if validating or not cacheable else _cache_lookup(pc_type)
    if cached is not None:
        return cached
    from .type_descriptor import encode_type_key, type_key
    misses_before = _name_token_misses
    result = encode_type_key(type_key(pc_type))
    if cacheable and not validating and misses_before == _name_token_misses:
        _cache_store(pc_type, result)
    return result
