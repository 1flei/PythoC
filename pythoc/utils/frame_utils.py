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
        filename_norm = filename.replace('\\', '/')
        filename_norm_lower = filename_norm.lower()
        
        # Check if this frame is in any skip package (case-insensitive for Windows)
        if not any(pkg in filename_norm_lower for pkg in skip_packages):
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
        filename_norm = filename.replace('\\', '/')
        filename_norm_lower = filename_norm.lower()
        func_name = code.co_name
        
        # Skip pythoc package internals (support both /pythoc/ and /pc/ paths)
        if ('/pythoc/decorators/' in filename_norm_lower or 
            '/pythoc/builtin_entities/' in filename_norm_lower or
            '/pythoc/ast_visitor/' in filename_norm_lower or
            '/pythoc/std/' in filename_norm_lower or
            '/pythoc/effect.py' in filename_norm_lower):
            temp_frame = temp_frame.f_back
            continue
        
        # Skip Python import machinery frames (importlib, frozen importlib, etc.)
        # These appear in the stack when @compile is triggered during import
        # and would incorrectly set scope to e.g. '_call_with_frames_removed.<locals>'
        if (filename_norm_lower.startswith('<frozen') or
            'importlib' in filename_norm_lower or
            func_name == '_call_with_frames_removed'):
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
