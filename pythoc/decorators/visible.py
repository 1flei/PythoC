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
        Dictionary mapping closure variable names to their values.
        Cells that have not been assigned yet (typical when inspecting a
        method defined inside an as-yet-unbound class body) are silently
        skipped; the consumer is expected to fall back to its own resolution
        path (captured_symbols, type_resolver) for those names.
    """
    if not callable(func):
        raise TypeError(f"Expected callable, got {type(func).__name__}")
    
    closure_vars: Dict[str, Any] = {}
    
    if hasattr(func, '__closure__') and func.__closure__ is not None:
        if hasattr(func, '__code__'):
            freevars = func.__code__.co_freevars
            for name, cell in zip(freevars, func.__closure__):
                try:
                    closure_vars[name] = cell.cell_contents
                except ValueError:
                    # Cell is empty: name is not yet bound at the time of
                    # inspection (e.g. self-reference inside a class body
                    # whose class object is still being constructed).
                    continue
    
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


def capture_user_caller_symbols() -> Dict[str, Any]:
    """Capture symbols from the first user frame, skipping pythoc internals.

    Decorators such as ``@union`` or ``@enum`` go through several pythoc
    layers (factory dispatch, ``DecoratorInstance.__call__``, base-class
    helpers) before reaching the user's module/function frame. Counting
    frames by hand is fragile, so this helper relies on
    :func:`find_caller_frame` to skip pythoc package files and returns the
    first user frame's globals merged with its locals (locals win).

    Returns an empty dict when no user frame is found.
    """
    from ..utils.frame_utils import find_caller_frame

    frame = find_caller_frame()
    if frame is None:
        return {}

    symbols = dict(frame.f_globals)
    symbols.update(frame.f_locals)
    return symbols


class AccessibleSymbols(dict):
    """Decoration-time symbol snapshot with a live reference to the defining
    module's globals.

    The snapshot takes precedence for every name it contains (closure and
    decoration-time captured symbols keep their priority).  Names absent
    from the snapshot -- e.g. module-level aliases defined after decoration
    -- fall back to the live module namespace at compile time, matching the
    lazy-compilation rule: a quoted name resolves if it is defined anywhere
    at module level before the first native call.
    """

    def __init__(self, *args, live_globals: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.live_globals = live_globals
        self.type_scope = getattr(args[0], 'type_scope', None) if args else None
        self.type_owner_name = getattr(args[0], 'type_owner_name', None) if args else None
        self.type_owner_namespace = getattr(args[0], 'type_owner_namespace', None) if args else None
        self.type_locals = getattr(args[0], 'type_locals', None) if args else None


def get_all_accessible_symbols(
    func,
    include_builtins: bool = False,
    include_closure: bool = True,
    include_annotations: bool = False,
    captured_symbols: Optional[Dict[str, Any]] = None,
    names=None,
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
        names: Optional subset to query without copying the whole namespace.
        
    Returns:
        Dictionary of all accessible symbols
    """
    symbols = AccessibleSymbols()

    def add_namespace(namespace):
        if names is None:
            symbols.update(namespace)
        else:
            symbols.update((name, namespace[name]) for name in names
                           if name in namespace)

    from ..type_resolver import TypeResolver
    captured = TypeResolver.capture_type_namespace(definition=func)
    scope = getattr(captured, 'type_scope', None)
    if scope is not None and scope.contains(func.__code__):
        symbols.type_scope = scope
        symbols.type_locals = captured.type_locals

    # Start with function's globals
    if hasattr(func, '__globals__'):
        add_namespace(func.__globals__)
        # Live reference for names defined after decoration (compile is lazy,
        # so by first call the whole module has executed).
        symbols.live_globals = func.__globals__
    
    # Add closure variables
    if include_closure:
        add_namespace(get_closure_variables(func))
    
    # Add captured symbols (HIGHEST priority)
    # This captures locals from decoration time that may not be available later
    if captured_symbols:
        add_namespace(captured_symbols)
        if scope is not None:
            symbols.type_locals = dict(symbols.type_locals or {})
            symbols.type_locals.update(
                (name, value) for name, value in captured_symbols.items()
                if name in scope.local_names)
    
    # Add annotations if requested
    if include_annotations and hasattr(func, '__annotations__'):
        symbols['__func_annotations__'] = func.__annotations__
    
    # Remove builtins if requested
    if not include_builtins and '__builtins__' in symbols:
        del symbols['__builtins__']
    
    return symbols
