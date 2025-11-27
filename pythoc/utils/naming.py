from .id_generator import get_next_id


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
        return suffix
    
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
        
        # Join with underscore and sanitize
        result = '_'.join(parts)
        # Remove special characters that might cause issues in symbol names
        result = result.replace('[', '_').replace(']', '_').replace(',', '_')
        result = result.replace(' ', '').replace('(', '_').replace(')', '_')
        return result
    
    # Handle single type object (PC type or Python type)
    if hasattr(suffix, 'get_name'):
        # PC type with get_name method (i32, f64, etc.)
        return suffix.get_name()
    elif isinstance(suffix, type):
        # Python type
        return suffix.__name__
    
    # Fallback: convert to string
    return str(suffix)
