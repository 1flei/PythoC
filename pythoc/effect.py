# -*- coding: utf-8 -*-
"""
Effect System for pythoc

Provides compile-time dependency injection with zero runtime overhead.
Effects are resolved at import time and compiled into static function calls.

Key concepts:
- effect.xxx: Access effect implementation (e.g., effect.rng.next())
- effect.default(): Set module-level defaults (overridable by caller)
- effect.xxx = impl: Direct assignment (NOT overridable by caller)
- with effect(...): Scoped override for imports

Resolution order:
1. Direct assignment in library (effect.xxx = impl) - HIGHEST, not overridable
2. Caller override (with effect(xxx=...) or effect.xxx = ...)
3. Module default (effect.default(xxx=...)) - overridable
4. Global built-in default - LOWEST

Import Override Mechanism:
When code runs `with effect(rng=MyRNG, suffix="custom"): from lib import func`,
the effect system intercepts the import and re-compiles @compile functions
with the overridden effect context. This produces new versions of functions
with the specified suffix (e.g., func_custom).
"""

import threading
import sys
import builtins
import importlib
import importlib.util
from types import SimpleNamespace, ModuleType
from typing import Any, Dict, Optional, Set, Tuple
from contextlib import contextmanager


# Store the original __import__ function
_original_import = builtins.__import__

# Thread-local storage for tracking effect usage during compilation
# This is used to record which effects a function uses (for transitive propagation)
_effect_usage_tracker = threading.local()

# Thread-local storage for tracking current compilation context
# This stores the suffix and effect overrides of the function currently being compiled
_compilation_context = threading.local()


def start_effect_tracking():
    """Start tracking effect usage for the current function being compiled."""
    if not hasattr(_effect_usage_tracker, 'stack'):
        _effect_usage_tracker.stack = []
    _effect_usage_tracker.stack.append(set())


def stop_effect_tracking() -> Set[str]:
    """Stop tracking and return the set of effects used."""
    if not hasattr(_effect_usage_tracker, 'stack') or not _effect_usage_tracker.stack:
        return set()
    return _effect_usage_tracker.stack.pop()


def record_effect_usage(effect_name: str):
    """Record that the current function uses the given effect."""
    if hasattr(_effect_usage_tracker, 'stack') and _effect_usage_tracker.stack:
        _effect_usage_tracker.stack[-1].add(effect_name)


def push_compilation_context(suffix: Optional[str], effect_overrides: Dict[str, Any], 
                            caller_module: Optional[str] = None, group_key: Optional[tuple] = None):
    """Push a compilation context onto the stack.

    Called when starting to compile a function that has effect overrides.
    This allows handle_call to know what suffix and effects to propagate.

    Args:
        suffix: The suffix for the current compilation (e.g., "mock")
        effect_overrides: Dict of effect name -> implementation
        caller_module: Optional caller module name (for import override)
        group_key: Optional group key tuple - transitive functions should use this
                   to be compiled into the same .so file as the caller
    """
    if not hasattr(_compilation_context, 'stack'):
        _compilation_context.stack = []
    _compilation_context.stack.append({
        'suffix': suffix,
        'effect_overrides': effect_overrides,
        'caller_module': caller_module,
        'group_key': group_key,
    })


def pop_compilation_context():
    """Pop the current compilation context."""
    if hasattr(_compilation_context, 'stack') and _compilation_context.stack:
        _compilation_context.stack.pop()


def get_current_compilation_context() -> Optional[Dict[str, Any]]:
    """Get the current compilation context (suffix and effect overrides)."""
    if hasattr(_compilation_context, 'stack') and _compilation_context.stack:
        return _compilation_context.stack[-1]
    return None


def _create_effect_wrapped_function(original_func, suffix: str, caller_module: str):
    """
    Create a new @compile function with the current effect context.

    This re-invokes the compile decorator with the current effect bindings
    captured, producing a new compiled version with the specified suffix.

    The new version is compiled into a separate .so file to avoid symbol conflicts.
    The caller_module is used to group the compiled function with the caller's code.
    """
    # Get the original Python function from __wrapped__
    original_python_func = getattr(original_func, '__wrapped__', None)
    if original_python_func is None:
        return None

    # Re-apply @compile decorator with the new suffix
    # The effect context is already set, so capture_effect_context() will
    # capture the current (overridden) effects
    from .decorators.compile import compile as compile_decorator

    try:
        # Create new compiled version with suffix
        # Pass _effect_caller_module to indicate this is an effect override import
        # This ensures proper grouping and avoids symbol conflicts
        new_wrapper = compile_decorator(
            original_python_func, 
            suffix=suffix,
            _effect_caller_module=caller_module
        )
        return new_wrapper
    except Exception:
        # If compilation fails, return original
        import traceback
        traceback.print_exc()
        return None


class EffectImportHook:
    """
    Import hook that intercepts imports within effect context.

    Uses builtins.__import__ to intercept all imports, including
    `from module import name` for already-cached modules.
    """

    def __init__(self, effect_context: 'EffectContext'):
        self._effect_context = effect_context
        self._wrapped_attrs: Dict[Tuple[str, str, str], Any] = {}  # (module, attr, suffix) -> wrapped func

    def __call__(self, name, globals=None, locals=None, fromlist=(), level=0):
        """Custom __import__ that wraps @compile functions with effect context."""
        # Call original import
        module = _original_import(name, globals, locals, fromlist, level)

        suffix = self._effect_context._suffix
        if suffix is None or not fromlist:
            return module

        # Get caller module name from globals
        caller_module = globals.get('__name__', '__main__') if globals else '__main__'

        # For `from module import name1, name2, ...`
        # We need to wrap @compile functions in the returned module
        for attr_name in fromlist:
            if attr_name == '*':
                continue

            try:
                attr = getattr(module, attr_name)
            except AttributeError:
                continue

            # Check if this is a @compile decorated function
            if not callable(attr) or not getattr(attr, '_is_compiled', False):
                continue

            # Skip private/internal functions
            if attr_name.startswith('_'):
                continue

            # Skip if this function already has the target suffix
            existing_mangled = getattr(attr, '_mangled_name', None)
            if existing_mangled and f'_{suffix}' in existing_mangled:
                continue

            # Check cache first - key includes caller module to allow different callers
            cache_key = (name, attr_name, suffix, caller_module)
            if cache_key in self._wrapped_attrs:
                wrapped = self._wrapped_attrs[cache_key]
            else:
                # Create a new version with the current effect context
                # Pass caller_module so the compiled function is grouped with caller's code
                wrapped = _create_effect_wrapped_function(attr, suffix, caller_module)
                if wrapped is not None:
                    self._wrapped_attrs[cache_key] = wrapped

            if wrapped is not None:
                # Save original attribute before modifying
                # This allows restoration in __exit__
                if name not in self._effect_context._saved_modules:
                    self._effect_context._saved_modules[name] = {}
                if attr_name not in self._effect_context._saved_modules[name]:
                    self._effect_context._saved_modules[name][attr_name] = attr

                # Replace the attribute in the module temporarily
                # so that `from module import func` gets the wrapped version
                setattr(module, attr_name, wrapped)

        return module


class EffectNamespace:
    """
    A namespace object that represents an effect category (e.g., rng, allocator).

    Attributes on this object forward to the current effect implementation.
    At compile time, effect.rng.next() is resolved to the actual function.

    Implements handle_attribute for compiler integration - allows the compiler
    to resolve effect.rng.next() to the actual function at compile time.
    """

    def __init__(self, name: str, implementation: Any = None):
        object.__setattr__(self, '_name', name)
        object.__setattr__(self, '_impl', implementation)

    def __getattr__(self, attr: str) -> Any:
        impl = object.__getattribute__(self, '_impl')
        if impl is None:
            name = object.__getattribute__(self, '_name')
            raise AttributeError(
                f"Effect '{name}' has no implementation bound. "
                f"Use effect.default({name}=impl) or effect.{name} = impl to set one."
            )

        if hasattr(impl, attr):
            return getattr(impl, attr)

        name = object.__getattribute__(self, '_name')
        raise AttributeError(
            f"Effect '{name}' implementation {type(impl).__name__} "
            f"has no attribute '{attr}'"
        )

    def __setattr__(self, attr: str, value: Any) -> None:
        # Setting attributes on the namespace sets them on the implementation
        impl = object.__getattribute__(self, '_impl')
        if impl is None:
            name = object.__getattribute__(self, '_name')
            raise AttributeError(
                f"Effect '{name}' has no implementation bound. "
                f"Cannot set attribute '{attr}'."
            )
        setattr(impl, attr, value)

    def _set_impl(self, impl: Any) -> None:
        """Internal: set the implementation"""
        object.__setattr__(self, '_impl', impl)

    def _get_impl(self) -> Any:
        """Internal: get the implementation"""
        return object.__getattribute__(self, '_impl')

    def handle_attribute(self, visitor, base, attr_name: str, node):
        """Handle attribute access for compiler integration.

        Called by PythonType.handle_attribute when the compiler encounters
        effect.rng.next() - this resolves .next to the actual function.

        Args:
            visitor: AST visitor
            base: Pre-evaluated base object (ValueRef containing this EffectNamespace)
            attr_name: Attribute name (e.g., 'next', 'seed')
            node: ast.Attribute node

        Returns:
            ValueRef wrapping the resolved attribute (usually a @compile function)
        """
        impl = object.__getattribute__(self, '_impl')
        name = object.__getattribute__(self, '_name')

        # Record effect usage for transitive propagation
        # This tracks that the current function being compiled uses this effect
        record_effect_usage(name)

        if impl is None:
            from .logger import logger
            logger.error(
                f"Effect '{name}' has no implementation bound. "
                f"Use effect.default({name}=impl) or effect.{name} = impl to set one.",
                node=node, exc_type=AttributeError
            )

        if not hasattr(impl, attr_name):
            from .logger import logger
            logger.error(
                f"Effect '{name}' implementation {type(impl).__name__} "
                f"has no attribute '{attr_name}'",
                node=node, exc_type=AttributeError
            )

        attr_value = getattr(impl, attr_name)

        # Wrap the attribute value for the compiler
        from .valueref import wrap_value
        from .builtin_entities.python_type import PythonType

        attr_type = PythonType.wrap(attr_value, is_constant=True)
        return wrap_value(attr_value, kind="python", type_hint=attr_type)

    def __repr__(self) -> str:
        name = object.__getattribute__(self, '_name')
        impl = object.__getattribute__(self, '_impl')
        if impl is None:
            return f"<EffectNamespace '{name}' (unbound)>"
        return f"<EffectNamespace '{name}' -> {type(impl).__name__}>"


class EffectContext:
    """
    Context manager for scoped effect overrides.

    Usage:
        with effect(rng=my_rng, suffix="custom"):
            from mylib import func

    When imports occur within this block, @compile functions are re-compiled
    with the overridden effects and the specified suffix.
    """

    def __init__(self, effect_manager: 'Effect', overrides: Dict[str, Any], suffix: Optional[str] = None):
        self._effect = effect_manager
        self._overrides = overrides
        self._suffix = suffix
        self._saved: Dict[str, Any] = {}
        self._saved_direct: Dict[str, bool] = {}
        self._import_hook: Optional[EffectImportHook] = None
        self._saved_modules: Dict[str, Dict[str, Any]] = {}  # module_name -> {attr_name -> original_func}

    def __enter__(self) -> 'EffectContext':
        # Save current state and apply overrides
        for name, impl in self._overrides.items():
            # Save current implementation
            if name in self._effect._effects:
                ns = self._effect._effects[name]
                self._saved[name] = ns._get_impl()
            else:
                self._saved[name] = None

            # Save whether this was a direct assignment
            self._saved_direct[name] = name in self._effect._direct_assignments

            # Apply override (this is a caller override, so it can override defaults
            # but NOT direct assignments in libraries)
            # Note: The actual check happens during compilation when resolving effects
            self._effect._set_effect(name, impl, is_caller_override=True)

        # Push suffix to stack
        if self._suffix is not None:
            self._effect._suffix_stack.append(self._suffix)

        # Install builtins.__import__ hook to intercept all imports
        self._import_hook = EffectImportHook(self)
        builtins.__import__ = self._import_hook

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Restore original __import__ first
        if self._import_hook is not None:
            builtins.__import__ = _original_import

            # Restore modified module attributes
            for module_name, attrs in self._saved_modules.items():
                if module_name in sys.modules:
                    module = sys.modules[module_name]
                    for attr_name, original_func in attrs.items():
                        setattr(module, attr_name, original_func)

            self._import_hook = None

        # Restore saved state
        for name, saved_impl in self._saved.items():
            if saved_impl is None:
                # Remove the effect if it didn't exist before
                if name in self._effect._effects:
                    del self._effect._effects[name]
            else:
                self._effect._effects[name]._set_impl(saved_impl)

            # Restore direct assignment status
            if self._saved_direct.get(name, False):
                self._effect._direct_assignments.add(name)
            elif name in self._effect._direct_assignments:
                self._effect._direct_assignments.discard(name)

        # Pop suffix from stack
        if self._suffix is not None:
            self._effect._suffix_stack.pop()

        return False  # Don't suppress exceptions


class Effect:
    """
    Global effect manager singleton.

    Provides:
    - effect.xxx: Access effect implementations
    - effect.default(**kwargs): Set module-level defaults
    - effect.xxx = impl: Direct assignment (not overridable)
    - with effect(**kwargs): Scoped overrides

    Thread-safe for concurrent compilation.
    """

    # Reserved attribute names that should not be treated as effects
    _RESERVED = frozenset({
        '_effects', '_defaults', '_direct_assignments', '_lock',
        '_suffix_stack', 'default', '_set_effect', '_get_effect',
        '_resolve_effect', '_get_current_suffix', '_RESERVED',
        'get_effect_impl', 'has_effect', 'list_effects',
        'is_direct_assignment', 'handle_attribute',  # Compiler integration methods
        'handle_call',  # Prevent these from being treated as effects
    })

    def __init__(self):
        object.__setattr__(self, '_effects', {})  # name -> EffectNamespace
        object.__setattr__(self, '_defaults', {})  # module -> {name -> impl}
        object.__setattr__(self, '_direct_assignments', set())  # names with direct assignment
        object.__setattr__(self, '_lock', threading.RLock())
        object.__setattr__(self, '_suffix_stack', [])  # Stack of active suffixes

    def __call__(self, suffix: Optional[str] = None, **overrides) -> EffectContext:
        """
        Create a scoped effect override context.

        Usage:
            with effect(rng=my_rng, suffix="custom"):
                from mylib import func

        Args:
            suffix: Required when overriding effects (for C ABI symbol naming)
            **overrides: Effect name -> implementation mappings

        Returns:
            EffectContext for use with 'with' statement
        """
        # Validate: suffix is required when overriding effects
        if overrides and suffix is None:
            raise ValueError(
                "Effect override requires 'suffix' parameter for C ABI symbol naming. "
                "Example: with effect(rng=my_rng, suffix='custom'): ..."
            )

        return EffectContext(self, overrides, suffix)

    def default(self, **kwargs) -> None:
        """
        Set module-level effect defaults.

        These defaults CAN be overridden by caller's with effect() block.
        Use direct assignment (effect.xxx = impl) for non-overridable effects.

        Usage:
            effect.default(rng=my_rng, allocator=my_alloc)
        """
        import inspect

        # Get the caller's module
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back
            caller_module = caller_frame.f_globals.get('__name__', '__main__')
        finally:
            del frame

        with self._lock:
            if caller_module not in self._defaults:
                self._defaults[caller_module] = {}

            for name, impl in kwargs.items():
                # Store as module default
                self._defaults[caller_module][name] = impl

                # Also set as current effect if not already set by direct assignment
                if name not in self._direct_assignments:
                    self._set_effect(name, impl, is_default=True)

    def _set_effect(self, name: str, impl: Any, 
                    is_direct: bool = False, 
                    is_default: bool = False,
                    is_caller_override: bool = False) -> None:
        """
        Internal: Set an effect implementation.

        Args:
            name: Effect name
            impl: Implementation object
            is_direct: True if this is a direct assignment (effect.xxx = impl)
            is_default: True if this is from effect.default()
            is_caller_override: True if this is from caller's with effect() block
        """
        with self._lock:
            # If this effect has a direct assignment, caller overrides cannot change it
            if is_caller_override and name in self._direct_assignments:
                # Direct assignments are immune to caller override
                return

            # Create or update the effect namespace
            if name not in self._effects:
                self._effects[name] = EffectNamespace(name, impl)
            else:
                self._effects[name]._set_impl(impl)

            # Track direct assignments
            if is_direct:
                self._direct_assignments.add(name)

    def _get_effect(self, name: str) -> Optional[EffectNamespace]:
        """Internal: Get an effect namespace"""
        with self._lock:
            return self._effects.get(name)

    def _resolve_effect(self, name: str, caller_module: Optional[str] = None) -> Any:
        """
        Resolve an effect to its current implementation.

        Resolution order:
        1. Direct assignment (effect.xxx = impl) - not overridable
        2. Caller override (from with effect() context)
        3. Module default (from effect.default())
        4. Global built-in default

        Args:
            name: Effect name
            caller_module: Module requesting the effect (for default lookup)

        Returns:
            The resolved implementation, or None if not found
        """
        with self._lock:
            ns = self._effects.get(name)
            if ns is not None:
                impl = ns._get_impl()
                if impl is not None:
                    return impl

            # Check module defaults
            if caller_module and caller_module in self._defaults:
                impl = self._defaults[caller_module].get(name)
                if impl is not None:
                    return impl

            return None

    def _get_current_suffix(self) -> Optional[str]:
        """Get the current suffix from the context stack"""
        with self._lock:
            if self._suffix_stack:
                return self._suffix_stack[-1]
            return None

    def __getattr__(self, name: str) -> Any:
        """
        Access an effect namespace.

        Usage: effect.rng.next()
        """
        if name.startswith('_') or name in Effect._RESERVED:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        with self._lock:
            if name in self._effects:
                return self._effects[name]

            # Create an unbound namespace - will error on access
            ns = EffectNamespace(name, None)
            self._effects[name] = ns
            return ns

    def handle_attribute(self, visitor, base, attr_name: str, node):
        """Handle attribute access for compiler integration.

        Called by PythonType.handle_attribute when the compiler encounters
        effect.rng or effect.MULTIPLIER.

        For namespace effects (e.g., effect.rng with rng=SimpleNamespace(...)):
            Returns the EffectNamespace wrapped for the compiler.

        For simple value effects (e.g., effect.MULTIPLIER with MULTIPLIER=2):
            Returns the value directly wrapped as a Python constant.

        Args:
            visitor: AST visitor
            base: Pre-evaluated base object (ValueRef containing this Effect)
            attr_name: Attribute name (e.g., 'rng', 'allocator', 'MULTIPLIER')
            node: ast.Attribute node

        Returns:
            ValueRef wrapping the effect value
        """
        if attr_name.startswith('_') or attr_name in Effect._RESERVED:
            from .logger import logger
            logger.error(
                f"Cannot access reserved attribute '{attr_name}' on effect",
                node=node, exc_type=AttributeError
            )

        from .valueref import wrap_value
        from .builtin_entities.python_type import PythonType

        with self._lock:
            if attr_name in self._effects:
                ns = self._effects[attr_name]
                impl = ns._get_impl()

                # For simple value types (int, float, bool, str), return value directly
                # This allows effect.MULTIPLIER to be used in expressions like x * effect.MULTIPLIER
                if impl is not None and isinstance(impl, (int, float, bool, str)):
                    value_type = PythonType.wrap(impl, is_constant=True)
                    return wrap_value(impl, kind="python", type_hint=value_type)

                # For namespace effects (SimpleNamespace, modules, etc.), return EffectNamespace
                ns_type = PythonType.wrap(ns, is_constant=True)
                return wrap_value(ns, kind="python", type_hint=ns_type)
            else:
                # Create an unbound namespace - will error on access
                ns = EffectNamespace(attr_name, None)
                self._effects[attr_name] = ns
                ns_type = PythonType.wrap(ns, is_constant=True)
                return wrap_value(ns, kind="python", type_hint=ns_type)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Direct assignment of effect implementation.

        Usage: effect.rng = my_rng

        This creates a non-overridable effect binding.
        """
        if name.startswith('_') or name in Effect._RESERVED:
            object.__setattr__(self, name, value)
            return

        # Direct assignment - mark as non-overridable
        self._set_effect(name, value, is_direct=True)

    def __delattr__(self, name: str) -> None:
        """
        Remove an effect binding.

        Usage: del effect.rng
        """
        if name.startswith('_') or name in Effect._RESERVED:
            object.__delattr__(self, name)
            return

        with self._lock:
            if name in self._effects:
                del self._effects[name]
            self._direct_assignments.discard(name)

    # Public API methods

    def get_effect_impl(self, name: str) -> Any:
        """
        Get the current implementation for an effect.

        Returns None if the effect is not bound.
        """
        with self._lock:
            ns = self._effects.get(name)
            if ns is not None:
                return ns._get_impl()
            return None

    def has_effect(self, name: str) -> bool:
        """Check if an effect is bound"""
        with self._lock:
            ns = self._effects.get(name)
            return ns is not None and ns._get_impl() is not None

    def list_effects(self) -> Dict[str, bool]:
        """
        List all registered effects and their bound status.

        Returns:
            Dict mapping effect name to whether it's bound
        """
        with self._lock:
            return {
                name: ns._get_impl() is not None
                for name, ns in self._effects.items()
            }

    def is_direct_assignment(self, name: str) -> bool:
        """Check if an effect was set via direct assignment (non-overridable)"""
        with self._lock:
            return name in self._direct_assignments


# Global singleton instance
effect = Effect()


def get_current_effect_suffix() -> Optional[str]:
    """
    Get the current effect suffix from the context stack.

    Used by the compiler to determine symbol naming.
    """
    return effect._get_current_suffix()


def capture_effect_context() -> Dict[str, Any]:
    """
    Capture the current effect context for later restoration.

    Returns:
        Dict mapping effect name to implementation
    """
    with effect._lock:
        return {
            name: ns._get_impl()
            for name, ns in effect._effects.items()
            if ns._get_impl() is not None
        }


@contextmanager
def restore_effect_context(captured: Dict[str, Any]):
    """
    Context manager to temporarily restore a captured effect context.

    Args:
        captured: Dict from capture_effect_context()
    """
    from .logger import logger

    # Save current state
    saved = {}
    with effect._lock:
        for name, impl in captured.items():
            if name in effect._effects:
                saved[name] = effect._effects[name]._get_impl()
            else:
                saved[name] = None

            # Set captured implementation
            if name not in effect._effects:
                effect._effects[name] = EffectNamespace(name, impl)
            else:
                effect._effects[name]._set_impl(impl)

        logger.debug(f"restore_effect_context: restored {captured}")

    try:
        yield
    finally:
        # Restore saved state
        with effect._lock:
            for name, saved_impl in saved.items():
                if saved_impl is None:
                    if name in effect._effects:
                        effect._effects[name]._set_impl(None)
                else:
                    effect._effects[name]._set_impl(saved_impl)
