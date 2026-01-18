import inspect


def find_caller_frame(skip_packages=None):
    """
    Find the first frame outside specified packages.
    
    Args:
        skip_packages: List of package path patterns to skip (e.g., ['/pythoc/decorators/', '/pythoc/std/'])
    
    Returns:
        Frame object or None if not found
    """
    if skip_packages is None:
        skip_packages = ['/pythoc/decorators/', '/pythoc/std/', '/pythoc/builtin_entities/', '/pythoc/ast_visitor/',
                        '/pc/decorators/', '/pc/std/', '/pc/builtin_entities/', '/pc/ast_visitor/']
    
    frame = inspect.currentframe()
    temp_frame = frame.f_back if frame else None
    
    while temp_frame:
        filename = temp_frame.f_code.co_filename
        
        # Check if this frame is in any skip package
        if not any(pkg in filename for pkg in skip_packages):
            return temp_frame
        
        temp_frame = temp_frame.f_back
    
    return None


def get_definition_scope():
    """
    Find the enclosing factory/template function generating compiled functions.
    Walk up stack to find first non-decorator frame outside pythoc/ package.
    
    Returns:
        str: Function name with .<locals> suffix (e.g., 'GenericType.<locals>'),
             or 'module' for module-level, 'unknown' if not found
             
    The .<locals> suffix matches Python's __qualname__ convention and allows
    distinguishing between functions with the same name in different scopes.
    """
    frame = inspect.currentframe()
    
    # Skip frames inside pythoc internals
    temp_frame = frame.f_back if frame else None
    scope_parts = []
    
    while temp_frame:
        code = temp_frame.f_code
        filename = code.co_filename
        func_name = code.co_name
        
        # Skip pythoc package internals (support both /pythoc/ and /pc/ paths)
        if ('/pythoc/decorators/' in filename or 
            '/pythoc/builtin_entities/' in filename or
            '/pythoc/ast_visitor/' in filename or
            '/pythoc/std/' in filename or
            '/pythoc/effect.py' in filename):
            temp_frame = temp_frame.f_back
            continue
        
        # Found user code
        if func_name == '<module>':
            # At module level - return collected scope or 'module'
            if scope_parts:
                return '.<locals>.'.join(reversed(scope_parts)) + '.<locals>'
            return 'module'
        
        # Collect this scope name
        scope_parts.append(func_name)
        
        # Return the innermost user function with .<locals> suffix
        # This matches Python's __qualname__ format: GenericType.<locals>
        return func_name + '.<locals>'
    
    # Fallback
    return 'unknown'
