"""
Utilities for extracting visible symbols from calling context.

This module provides functions to capture symbols visible at decoration time,
which is critical for resolving type annotation names in @compile decorated
functions.
"""

import inspect
from typing import Dict, Any, Optional


def get_closure_variables(func) -> Dict[str, Any]:
    """Extract closure variables from a function.
    
    Args:
        func: Function object to inspect
        
    Returns:
        Dictionary mapping closure variable names to their values
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


def capture_caller_symbols(depth: int = 1) -> Dict[str, Any]:
    """Capture all visible symbols from the caller's frame at decoration time.
    
    This captures both globals and locals from the caller's frame, providing
    a complete snapshot of all symbols visible at the decoration point.
    
    This should be called immediately when a decorator is applied, to capture
    the correct symbols before the stack frame changes.
    
    Args:
        depth: How many frames to go back (1 = immediate caller)
        
    Returns:
        Dictionary of all visible symbols (globals + locals) from caller's frame.
        Locals take precedence over globals with the same name.
    """
    frame = inspect.currentframe()
    try:
        for _ in range(depth + 1):  # +1 for this function itself
            if frame is None:
                return {}
            frame = frame.f_back
        
        if frame is None:
            return {}
        
        # Capture globals first, then locals (locals override globals)
        symbols = dict(frame.f_globals)
        symbols.update(frame.f_locals)
        return symbols
    finally:
        del frame


def get_all_accessible_symbols(
    func,
    include_builtins: bool = False,
    include_closure: bool = True,
    include_annotations: bool = False,
    captured_symbols: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Get all symbols accessible to a function.
    
    This combines:
    - Function's global namespace (func.__globals__)
    - Function's closure variables
    - Captured symbols from decorator call time (highest priority)
    - Optionally, builtins and annotations
    
    The priority order (lowest to highest):
    1. func.__globals__ - function's module globals
    2. closure variables - captured from enclosing scopes
    3. captured_symbols - all visible symbols at decoration time (HIGHEST)
    
    Args:
        func: Function to inspect
        include_builtins: Include Python builtins
        include_closure: Include closure variables
        include_annotations: Include type annotations
        captured_symbols: Pre-captured symbols from decorator call time.
            Should be captured using capture_caller_symbols() immediately
            when the decorator is applied. These take HIGHEST priority.
        
    Returns:
        Dictionary of all accessible symbols
    """
    symbols = {}
    
    # Start with function's globals
    if hasattr(func, '__globals__'):
        symbols.update(func.__globals__)
    
    # Add closure variables
    if include_closure:
        symbols.update(get_closure_variables(func))
    
    # Add captured symbols (HIGHEST priority)
    if captured_symbols:
        symbols.update(captured_symbols)
    
    # Add annotations if requested
    if include_annotations and hasattr(func, '__annotations__'):
        symbols['__func_annotations__'] = func.__annotations__
    
    # Remove builtins if requested
    if not include_builtins and '__builtins__' in symbols:
        del symbols['__builtins__']
    
    return symbols
