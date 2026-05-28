import hashlib
import re

from .id_generator import get_next_id


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
    
    if isinstance(suffix, str):
        return _symbol_safe_suffix(suffix)
    
    if isinstance(suffix, (tuple, list)):
        # Convert type parameters to string
        parts = []
        for item in suffix:
            if hasattr(item, 'get_name'):
                # PC type with get_name method
                name = item.get_name()
            elif isinstance(item, type):
                # Python type
                name = item.__name__
            elif isinstance(item, (int, str)):
                # Literal value
                name = str(item)
            else:
                # Fallback to str()
                name = str(item)
            parts.append(name)
        
        return _symbol_safe_suffix('_'.join(parts))
    
    # Handle single type object (PC type or Python type)
    if hasattr(suffix, 'get_name'):
        # PC type with get_name method (i32, f64, etc.)
        return _symbol_safe_suffix(suffix.get_name())
    elif isinstance(suffix, type):
        # Python type
        return _symbol_safe_suffix(suffix.__name__)
    
    # Fallback: convert to string
    return _symbol_safe_suffix(suffix)
