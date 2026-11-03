import hashlib
import json
import re

from .id_generator import get_next_id
from ..type_descriptor import specialization_key


class NormalizedSuffix(str):
    """Short display/path suffix with its lossless input identity attached."""

    def __new__(cls, value, identity):
        result = str.__new__(cls, value)
        result.identity = identity
        return result


_SYMBOL_SAFE_RE = re.compile(r'[^0-9A-Za-z_]+')
_UNDERSCORE_RE = re.compile(r'_+')


def _symbol_safe_suffix(value):
    text = str(value)
    safe = _SYMBOL_SAFE_RE.sub('_', text)
    safe = _UNDERSCORE_RE.sub('_', safe).strip('_')
    if not safe:
        safe = "suffix"
    if safe == text and len(safe) <= 80:
        return safe

    digest = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
    prefix = safe[:64].rstrip('_')
    if not prefix:
        return f"hash_{digest}"
    return f"{prefix}_hash_{digest}"


def get_anonymous_suffix():
    """
    Generate unique anonymous suffix for dynamic functions.
    
    Returns:
        str: Unique suffix like '_anon1', '_anon2', etc.
    """
    return f'_anon{get_next_id()}'


def normalize_suffix(suffix):
    """
    Convert suffix parameter to a normalized string.
    
    Args:
        suffix: Can be:
            - str: use as-is
            - tuple/list: convert type parameters to string
            - PC type: extract type name
            - None: return None
    
    Returns:
        Normalized suffix string or None
    """
    if suffix is None:
        return None
    if isinstance(suffix, NormalizedSuffix):
        return suffix
    
    if isinstance(suffix, str):
        return NormalizedSuffix(_symbol_safe_suffix(suffix), specialization_key(suffix))
    
    if type(suffix) in (tuple, list, dict):
        identity = specialization_key(suffix)
        text = json.dumps(identity, ensure_ascii=True, separators=(',', ':'))
        return NormalizedSuffix(_symbol_safe_suffix(text), identity)
    
    # Handle single type object (PC type or Python type)
    from ..builtin_entities.base import _is_type_like
    if _is_type_like(suffix):
        # PC type with get_name method (i32, f64, etc.)
        from ..type_id import get_type_id
        return NormalizedSuffix(_symbol_safe_suffix(get_type_id(suffix)),
                                specialization_key(suffix))
    elif isinstance(suffix, type):
        # Python type
        return NormalizedSuffix(_symbol_safe_suffix(suffix.__name__),
                                specialization_key(suffix))
    
    # Fallback: convert to string
    return NormalizedSuffix(_symbol_safe_suffix(suffix), specialization_key(suffix))
