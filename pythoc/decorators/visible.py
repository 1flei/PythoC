"""
Utilities for extracting visible symbols from calling context.

This module provides functions to inspect the calling stack and extract
all visible symbols including:
- Global variables
- Local variables
- Imported modules and names
- Closure variables
- Builtin names
"""

import inspect
import sys
from typing import Dict, Any, Optional, Set


def get_visible_symbols(depth: int = 1, include_builtins: bool = False) -> Dict[str, Any]:
    """Get all visible symbols at a specific stack depth.
    
    This function inspects the calling stack and extracts all symbols that would
    be visible at that point, including:
    - Global variables (module-level)
    - Local variables (function/method locals)
    - Nonlocal/closure variables (from enclosing scopes)
    - Imported names
    - Optionally, builtin names
    
    Args:
        depth: Stack depth to inspect (1 = immediate caller, 2 = caller's caller, etc.)
        include_builtins: Whether to include Python builtin names (__builtins__)
        
    Returns:
        Dictionary mapping symbol names to their values
        
    Examples:
        >>> x = 10
        >>> def foo():
        ...     y = 20
        ...     symbols = get_visible_symbols(depth=1)
        ...     # symbols will contain both x (global) and y (local)
        
        >>> def outer():
        ...     z = 30
        ...     def inner():
        ...         w = 40
        ...         symbols = get_visible_symbols(depth=1)
        ...         # symbols will contain z (nonlocal), w (local), and globals
    """
    frame = inspect.currentframe()
    
    # Navigate to the target frame
    try:
        for _ in range(depth):
            if frame is None:
                raise ValueError(f"Stack depth {depth} exceeds available frames")
            frame = frame.f_back
        
        if frame is None:
            raise ValueError(f"Stack depth {depth} exceeds available frames")
        
        # Start with global variables (module-level)
        symbols = dict(frame.f_globals)
        
        # Add local variables (overrides globals with same name)
        symbols.update(frame.f_locals)
        
        # Optionally remove builtins to reduce noise
        if not include_builtins and '__builtins__' in symbols:
            del symbols['__builtins__']
        
        return symbols
        
    finally:
        # Clean up frame reference to avoid reference cycles
        del frame


def get_closure_variables(func) -> Dict[str, Any]:
    """Extract closure variables from a function.
    
    Args:
        func: Function object to inspect
        
    Returns:
        Dictionary mapping closure variable names to their values
        
    Examples:
        >>> def outer(x):
        ...     def inner():
        ...         return x
        ...     return inner
        >>> f = outer(42)
        >>> get_closure_variables(f)
        {'x': 42}
    """
    if not callable(func):
        raise TypeError(f"Expected callable, got {type(func).__name__}")
    
    closure_vars = {}
    
    if hasattr(func, '__closure__') and func.__closure__ is not None:
        if hasattr(func, '__code__'):
            freevars = func.__code__.co_freevars
            closure_values = [cell.cell_contents for cell in func.__closure__]
            closure_vars = dict(zip(freevars, closure_values))
    
    return closure_vars


def get_imported_names(frame_depth: int = 1) -> Dict[str, Any]:
    """Get all imported names visible at a specific stack depth.
    
    This extracts names that were imported via import statements or from imports.
    
    Args:
        frame_depth: Stack depth to inspect
        
    Returns:
        Dictionary of imported module names and imported symbols
    """
    frame = inspect.currentframe()
    
    try:
        for _ in range(frame_depth):
            if frame is None:
                return {}
            frame = frame.f_back
        
        if frame is None:
            return {}
        
        imports = {}
        
        # Check both globals and locals for imported names
        all_names = {**frame.f_globals, **frame.f_locals}
        
        for name, value in all_names.items():
            # Check if it's a module
            if inspect.ismodule(value):
                imports[name] = value
            # Check if it's imported from a module (heuristic)
            elif hasattr(value, '__module__') and not name.startswith('_'):
                # Include classes, functions imported from other modules
                if value.__module__ != frame.f_globals.get('__name__'):
                    imports[name] = value
        
        return imports
        
    finally:
        del frame


def get_type_annotations(frame_depth: int = 1) -> Dict[str, Any]:
    """Extract type annotations visible at a specific stack depth.
    
    Args:
        frame_depth: Stack depth to inspect
        
    Returns:
        Dictionary of variable names to their type annotations
    """
    frame = inspect.currentframe()
    
    try:
        for _ in range(frame_depth):
            if frame is None:
                return {}
            frame = frame.f_back
        
        if frame is None:
            return {}
        
        annotations = {}
        
        # Check for __annotations__ in locals (function/class level)
        if '__annotations__' in frame.f_locals:
            annotations.update(frame.f_locals['__annotations__'])
        
        # Check for __annotations__ in globals (module level)
        if '__annotations__' in frame.f_globals:
            annotations.update(frame.f_globals['__annotations__'])
        
        return annotations
        
    finally:
        del frame


def get_all_accessible_symbols(
    func,
    include_builtins: bool = False,
    include_closure: bool = True,
    include_annotations: bool = False,
    include_calling_scope: bool = True
) -> Dict[str, Any]:
    """Get all symbols accessible to a function.
    
    This combines:
    - Function's global namespace (func.__globals__)
    - Function's closure variables
    - Calling stack frames (to capture factory function parameters, etc.)
    - Optionally, builtins and annotations
    
    Args:
        func: Function to inspect
        include_builtins: Include Python builtins
        include_closure: Include closure variables
        include_annotations: Include type annotations
        include_calling_scope: Include symbols from calling stack frames
        
    Returns:
        Dictionary of all accessible symbols
        
    Examples:
        >>> def factory(element_type, size_type):
        ...     _ = (element_type, size_type)  # Force closure capture
        ...     def inner():
        ...         pass
        ...     symbols = get_all_accessible_symbols(inner, include_calling_scope=True)
        ...     # symbols will include element_type and size_type from factory's frame
    """
    symbols = {}
    
    # Start with function's globals
    if hasattr(func, '__globals__'):
        symbols.update(func.__globals__)
    
    # Add closure variables
    if include_closure:
        symbols.update(get_closure_variables(func))
    
    # Add symbols from calling stack frames
    # This is critical for capturing factory function parameters and other
    # symbols that may not be in closures (e.g., type annotation names)
    if include_calling_scope:
        frame = inspect.currentframe()
        try:
            # Walk up the stack to find symbols
            # Skip the first frame (this function itself)
            temp_frame = frame.f_back if frame else None
            
            while temp_frame:
                # Add locals from each frame (but don't override existing symbols)
                for name, value in temp_frame.f_locals.items():
                    if name not in symbols:
                        symbols[name] = value
                
                temp_frame = temp_frame.f_back
        finally:
            del frame
    
    # Add annotations if requested
    if include_annotations and hasattr(func, '__annotations__'):
        # Store annotations separately to avoid conflicts
        symbols['__func_annotations__'] = func.__annotations__
    
    # Remove builtins if requested
    if not include_builtins and '__builtins__' in symbols:
        del symbols['__builtins__']
    
    return symbols


def summarize_visible_symbols(depth: int = 1) -> Dict[str, Set[str]]:
    """Get a summary of visible symbols categorized by type.
    
    Args:
        depth: Stack depth to inspect
        
    Returns:
        Dictionary with categories: 'modules', 'functions', 'classes', 
        'variables', 'builtins'
    """
    symbols = get_visible_symbols(depth=depth + 1, include_builtins=True)
    
    summary = {
        'modules': set(),
        'functions': set(),
        'classes': set(),
        'variables': set(),
        'builtins': set(),
    }
    
    # Get builtins for comparison
    import builtins as builtin_module
    builtin_names = set(dir(builtin_module))
    
    for name, value in symbols.items():
        if name.startswith('_') and name != '__builtins__':
            continue  # Skip private/dunder names
            
        if name in builtin_names:
            summary['builtins'].add(name)
        elif inspect.ismodule(value):
            summary['modules'].add(name)
        elif inspect.isclass(value):
            summary['classes'].add(name)
        elif inspect.isfunction(value) or inspect.ismethod(value):
            summary['functions'].add(name)
        else:
            summary['variables'].add(name)
    
    return summary
