# -*- coding: utf-8 -*-
"""
Forward Reference Resolution System

This module provides a callback-based system for resolving forward references
in @compile decorated classes. It handles self-reference, forward reference,
and circular reference uniformly.

Key concepts:
- Types can be marked as "defined" when they are fully constructed
- Callbacks can be registered to be triggered when a type is defined
- This allows field types to be updated after the struct is created

State lives in ForwardRefNamespace instances owned by CompileSession;
the module-level functions delegate to the active session's namespace.
"""

from typing import Dict, List, Callable, Any, Optional
import re
import threading


class ForwardRefNamespace:
    """Per-session forward reference state.

    Holds the defined-type table and the pending-callback table for one
    compilation session.  Mutations are guarded by an RLock so concurrent
    register/mark calls cannot corrupt the tables (same-name semantic
    conflicts across threads are still possible and out of scope here).
    """

    def __init__(self) -> None:
        self._defined_types: Dict[str, Any] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
        self._lock = threading.RLock()

    def register_callback(self, type_name: str, callback: Callable[[Any], None]):
        """Register a callback to be called when type_name is defined

        Args:
            type_name: Name of the type to wait for
            callback: Function to call when type is defined, receives the type object
        """
        with self._lock:
            if type_name in self._defined_types:
                # Type already defined, call immediately
                type_obj = self._defined_types[type_name]
            else:
                # Type not yet defined, register callback
                self._callbacks.setdefault(type_name, []).append(callback)
                return
        callback(type_obj)

    def mark_type_defined(self, type_name: str, type_obj: Any):
        """Mark a type as fully defined and trigger all waiting callbacks

        Args:
            type_name: Name of the type being defined
            type_obj: The type object (usually a Python class)
        """
        with self._lock:
            self._defined_types[type_name] = type_obj
            callbacks = self._callbacks.pop(type_name, [])

        for callback in callbacks:
            try:
                callback(type_obj)
            except Exception as e:
                # Log error but continue with other callbacks
                import traceback
                print(f"Error in forward ref callback for {type_name}: {e}")
                traceback.print_exc()

    def is_type_defined(self, type_name: str) -> bool:
        """Check if a type has been marked as defined"""
        with self._lock:
            return type_name in self._defined_types

    def get_defined_type(self, type_name: str) -> Optional[Any]:
        """Get a defined type by name, or None if not defined"""
        with self._lock:
            return self._defined_types.get(type_name)

    def defined_types_snapshot(self) -> Dict[str, Any]:
        """Return a snapshot copy of the defined-types table"""
        with self._lock:
            return dict(self._defined_types)

    def clear(self):
        """Clear all forward reference state (for testing)"""
        with self._lock:
            self._callbacks.clear()
            self._defined_types.clear()

    def get_pending_callbacks(self) -> Dict[str, int]:
        """Get count of pending callbacks for each type (for debugging)

        Returns:
            Dict mapping type names to number of pending callbacks
        """
        with self._lock:
            return {name: len(callbacks) for name, callbacks in self._callbacks.items()}


def _current_namespace() -> ForwardRefNamespace:
    """Return the active session's forward reference namespace."""
    from .session import CompileSession
    return CompileSession.current().forward_refs


def register_forward_ref_callback(type_name: str, callback: Callable[[Any], None]):
    """Register a callback to be called when type_name is defined

    Args:
        type_name: Name of the type to wait for
        callback: Function to call when type is defined, receives the type object
    """
    _current_namespace().register_callback(type_name, callback)


def mark_type_defined(type_name: str, type_obj: Any):
    """Mark a type as fully defined and trigger all waiting callbacks

    Args:
        type_name: Name of the type being defined
        type_obj: The type object (usually a Python class)
    """
    _current_namespace().mark_type_defined(type_name, type_obj)


def is_type_defined(type_name: str) -> bool:
    """Check if a type has been marked as defined

    Args:
        type_name: Name of the type to check

    Returns:
        True if type is defined, False otherwise
    """
    return _current_namespace().is_type_defined(type_name)


def get_defined_type(type_name: str) -> Optional[Any]:
    """Get a defined type by name

    Args:
        type_name: Name of the type

    Returns:
        The type object if defined, None otherwise
    """
    return _current_namespace().get_defined_type(type_name)


def extract_type_names_from_annotation(annotation_str: str) -> List[str]:
    """Extract all type names from a type annotation string

    Examples:
        "ptr[Node]" -> ["Node"]
        "ptr[ptr[Node]]" -> ["Node"]
        "array[Node, 10]" -> ["Node"]
        "tuple[Node, TreeNode]" -> ["Node", "TreeNode"]

    Args:
        annotation_str: Type annotation string

    Returns:
        List of type names referenced in the annotation
    """
    # Remove builtin type names
    builtins = {'ptr', 'array', 'tuple', 'struct', 'union', 'const', 'static', 'volatile',
                'i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64',
                'f16', 'bf16', 'f32', 'f64', 'f128', 'bool', 'void'}

    # Extract all identifiers (alphanumeric + underscore)
    identifiers = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', annotation_str)

    # Filter out builtins and numbers
    type_names = []
    for ident in identifiers:
        if ident not in builtins and not ident.isdigit():
            if ident not in type_names:  # Avoid duplicates
                type_names.append(ident)

    return type_names


def clear_forward_ref_state():
    """Clear all forward reference state (for testing)"""
    _current_namespace().clear()


def get_pending_callbacks() -> Dict[str, int]:
    """Get count of pending callbacks for each type (for debugging)

    Returns:
        Dict mapping type names to number of pending callbacks
    """
    return _current_namespace().get_pending_callbacks()
