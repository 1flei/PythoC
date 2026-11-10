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

import builtins
import inspect
import os
import sys
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from types import MappingProxyType, ModuleType
from typing import Any, Dict, Mapping, Optional, Set

from .effect_graph import node_id_from_group_key
from .session import CompileSession


@dataclass(frozen=True)
class _EffectScope:
    bindings: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}),
    )
    suffixes: tuple = ()
    override_names: frozenset = frozenset()
    replay: bool = False


_EMPTY_SCOPE = _EffectScope()
# (Effect, import suffix, owning CompileSession). A context never replaces
# the process import function; this request is local to its execution context.
_active_import_context = ContextVar('pythoc_effect_import_context', default=None)
_context_tokens = ContextVar('pythoc_effect_context_tokens', default=())


def _instance_dict(value):
    """Inspect compiler metadata without invoking user __getattr__ hooks."""
    try:
        return object.__getattribute__(value, '__dict__')
    except AttributeError:
        return {}


def _callable_metadata(value):
    if not callable(value):
        return {}
    if not any(inspect.getattr_static(value, name, False) is True for name in (
        '_is_compiled', '_is_yield_generated',
    )):
        return {}
    return _instance_dict(value)


def record_effect_usage(effect_name: str, compile_frame: Optional[Any] = None):
    """Record that the function in ``compile_frame`` uses the given effect.

    Usage is attributed to the ActiveCompileFrame of the compilation whose
    visitor triggered the access (see ActiveCompileFrame.effect_usage).
    Calls without an active frame (e.g. decoration-time evaluation) are
    dropped: there is no function to attribute the usage to.
    """
    if compile_frame is not None:
        compile_frame.effect_usage.add(effect_name)


# Thread-local stack of FunctionBindingState for the functions currently
# being compiled on this thread.
#
# This dynamic scope is kept deliberately: the readers (effect
# specialization materialization in decorators/compile.py) run on planning-time paths
# that cannot reach the active visitor/binding without changing public
# wrapper APIs (get_effect_specialized, materialize_specialization).
# Code inside the visitor chain must read visitor.binding_state instead.
_compilation_context = threading.local()


def push_compilation_context(binding_state):
    """Push the binding state of the function being compiled.

    Called when starting to compile a function that has effect overrides.
    Planning-time code (no visitor at hand) uses this to see the compiling
    function's effect context.
    """
    if not hasattr(_compilation_context, 'stack'):
        _compilation_context.stack = []
    _compilation_context.stack.append(binding_state)


def pop_compilation_context():
    """Pop the current compilation context."""
    if hasattr(_compilation_context, 'stack') and _compilation_context.stack:
        _compilation_context.stack.pop()


def get_current_compilation_context() -> Optional[Dict[str, Any]]:
    """Get the current compilation context (suffix and effect overrides).

    Returns a dict view over the binding state of the innermost function
    being compiled on this thread, or None outside compilation.
    """
    if not hasattr(_compilation_context, 'stack') or not _compilation_context.stack:
        return None
    st = _compilation_context.stack[-1]
    return {
        'compile_suffix': st.compile_suffix,
        'effect_suffix': st.effect_suffix,
        'effect_overrides': st.captured_effect_context,
        'group_key': st.group_key,
        'effect_override_names': st.effect_override_names or set(),
    }


def _get_output_manager():
    """Resolve the output manager through the active compile session.

    The session owns the output manager; the fallback to
    build.output_manager.get_output_manager() lazily creates and
    attaches it on first use.
    """
    session = CompileSession.current()
    om = session.output_manager
    if om is None:
        from .build.output_manager import get_output_manager
        om = get_output_manager()
    return om


def resolve_effect_wrapper(wrapper, caller_group_key=None,
                           suffix=None, effect_overrides=None):
    """Explicit effect-specialization entry point.

    Import/planning code passes ``suffix`` and ``effect_overrides`` explicitly to
    obtain a specialized binding. Codegen/lowering must not infer effect
    specialization from the current compilation context; ordinary wrapper calls
    use their lexical binding. Direct ``effect.xxx`` access still uses the
    captured effect context of the function being compiled.

    Being registered as a provider is a property of an effect binding, not of
    the callable. An explicit request may specialize that same declaration;
    provider calls still use the reference stored in the captured binding.

    Args:
        wrapper: The @compile wrapper to potentially specialize.
        caller_group_key: Optional caller group for dependency recording.
        suffix: Explicit effect suffix. If omitted, no specialization happens.
        effect_overrides: Explicit effect implementation bindings.

    Returns:
        The specialized wrapper for explicit requests, otherwise ``wrapper``.
    """
    if not suffix or not effect_overrides:
        return wrapper

    # --- Resolve canonical wrapper ---
    func_info = getattr(wrapper, '_func_info', None)
    original_wrapper = func_info.wrapper if func_info else wrapper
    original_binding = getattr(
        original_wrapper, '_binding',
        getattr(original_wrapper, '_state', None)
    )
    if original_binding is None:
        return wrapper

    # Already effect-specialized for target suffix.
    if original_binding.effect_suffix is not None:
        return wrapper

    # --- Check group-level effects via EffectGraph ---
    should_specialize = True
    callee_group_key = original_binding.group_key
    if callee_group_key:
        from .effect_graph import node_id_from_group_key
        _om = _get_output_manager()
        _node_id = node_id_from_group_key(callee_group_key)
        if not _om._effect_graph.has_node(_node_id):
            # Node not yet in graph — conservative: specialize
            pass
        elif not _om._effect_graph.has_effects_declared(_node_id):
            # Node exists but effects not yet declared (not yet compiled)
            # — conservative: specialize
            pass
        else:
            # Effects declared (possibly empty set) — definitive answer
            should_specialize = _om._effect_graph.transitive_uses_effects(
                _node_id, frozenset(effect_overrides.keys()),
            )

    if not should_specialize:
        return wrapper

    # --- Produce specialized wrapper ---
    specialized_wrapper = _do_effect_specialize(
        original_wrapper, suffix, effect_overrides
    )
    if specialized_wrapper is None or specialized_wrapper is original_wrapper:
        return wrapper

    from .logger import logger as _logger
    _logger.debug(
        f"Effect specialization: {original_binding.original_name} -> "
        f"{getattr(specialized_wrapper, '_func_info', None) and specialized_wrapper._func_info.mangled_name}"
    )

    # Record dependency: caller_group -> specialized_wrapper_group
    spec_binding = getattr(
        specialized_wrapper, '_binding',
        getattr(specialized_wrapper, '_state', None)
    )
    spec_group_key = spec_binding.group_key if spec_binding else None
    if (caller_group_key and spec_group_key
            and caller_group_key != spec_group_key):
        from .build.deps import get_dependency_tracker
        get_dependency_tracker().record_group_dependency(
            caller_group_key, spec_group_key, "effect_specialized_call"
        )

    return specialized_wrapper


def _do_effect_specialize(original_wrapper, suffix, effect_overrides):
    """Produce an effect-specialized wrapper.

    Prefers get_effect_specialized() (reuses captured symbols correctly).
    Falls back to re-applying @compile for wrappers that don't support it.

    Returns None on failure.
    """
    if hasattr(original_wrapper, 'get_effect_specialized'):
        result = original_wrapper.get_effect_specialized(suffix, effect_overrides)
        if result is not None and result is not original_wrapper:
            return result
    from .logger import logger
    logger.error(f"Failed to specialize effect for {original_wrapper}")


@dataclass(frozen=True)
class _EffectImportRequest:
    manager: 'Effect'
    scope: _EffectScope
    suffix: str
    caller_file: Optional[str]

    def capture(self):
        """Freeze after loading, including the library's pinned declarations."""
        with self.manager._lock:
            bindings = self.manager._capture_context()
            for name in self.manager._direct_assignments:
                bindings[name] = object.__getattribute__(
                    self.manager._effects[name], '_impl',
                )
        scope = replace(
            self.scope, bindings=MappingProxyType(bindings), replay=True,
        )
        return replace(self, scope=scope)

    def bind(self, value):
        metadata = _callable_metadata(value)
        is_yield = metadata.get('_is_yield_generated', False)
        if not metadata.get('_is_compiled', False) and not is_yield:
            return value
        binding = metadata.get('_binding', metadata.get('_state'))
        bound_suffix = (
            binding.effect_suffix if binding is not None
            else metadata.get('_effect_suffix')
        )
        if bound_suffix is not None:
            return value

        self.manager._register_import_bindings(self.suffix, self.scope)
        token = self.manager._scope.set(self.scope)
        try:
            if is_yield:
                from .ast_visitor.yield_transform import specialize_yield_wrapper
                specialized = specialize_yield_wrapper(
                    value, self.suffix, dict(self.scope.bindings),
                )
            else:
                specialized = resolve_effect_wrapper(
                    value, suffix=self.suffix,
                    effect_overrides=dict(self.scope.bindings),
                )
        finally:
            self.manager._scope.reset(token)

        if specialized is not value and binding is not None and binding.group_key:
            graph = _get_output_manager()._effect_graph
            imported_node = node_id_from_group_key(binding.group_key)
            graph.add_layer(imported_node, self.suffix)
            if self.caller_file and self.caller_file != binding.group_key[0]:
                caller_node = (self.caller_file, None, None)
                graph.add_edge(caller_node, imported_node)
                graph.add_layer(caller_node, self.suffix)
        return specialized


class _EffectImportView(ModuleType):
    """One from-import's exports; never installed in sys.modules.

    Ordinary attributes are forwarded exactly when IMPORT_FROM requests them.
    In particular, dynamic Python exports are not probed during planning.
    """

    __slots__ = ('_source_module', '_request', '_requested_names', '_exports')

    def __init__(self, module, request, names, replacements):
        namespace = ModuleType.__getattribute__(module, '__dict__')
        super().__init__(namespace.get('__name__', ''))
        own_namespace = ModuleType.__getattribute__(self, '__dict__')
        for name in ('__doc__', '__package__', '__spec__', '__loader__',
                     '__file__', '__path__'):
            if name in namespace:
                own_namespace[name] = namespace[name]
        object.__setattr__(self, '_source_module', module)
        object.__setattr__(self, '_request', request)
        object.__setattr__(self, '_requested_names', frozenset(names))
        object.__setattr__(self, '_exports', replacements)

    def __getattribute__(self, name):
        names = object.__getattribute__(self, '_requested_names')
        if name not in names:
            return ModuleType.__getattribute__(self, name)
        exports = object.__getattribute__(self, '_exports')
        if name in exports:
            return exports[name]
        source = object.__getattribute__(self, '_source_module')
        request = object.__getattribute__(self, '_request')
        token = request.manager._scope.set(request.scope)
        try:
            value = getattr(source, name)
        finally:
            request.manager._scope.reset(token)
        if not name.startswith('_'):
            specialized = request.bind(value)
            if specialized is not value:
                exports[name] = specialized
                return specialized
        return value


def _bind_import_exports(module, names, request):
    """Build a private export view without touching the canonical module."""
    if not names or '*' in names or not isinstance(module, ModuleType):
        return module
    namespace = ModuleType.__getattribute__(module, '__dict__')
    custom_lookup = type(module).__getattribute__ is not ModuleType.__getattribute__
    replacements = {}
    for name in names:
        if not isinstance(name, str) or name.startswith('_'):
            continue
        if name in namespace and not custom_lookup:
            value = namespace[name]
            specialized = request.bind(value)
            if specialized is not value:
                replacements[name] = specialized
    dynamic = (
        '__getattr__' in namespace
        or custom_lookup
    )
    if not replacements and not dynamic:
        return module
    return _EffectImportView(module, request, names, replacements)


def _caller_module_from_visitor(visitor) -> Optional[str]:
    """Return the Python module name for the function currently being compiled."""
    if visitor is None:
        return None
    ctx = getattr(visitor, 'ctx', None)
    if ctx is None:
        return None
    user_globals = getattr(ctx, 'user_globals', None) or {}
    return user_globals.get('__name__')


def _block_bound_impl(effect_name: str, visitor) -> Any:
    """Innermost block-scoped effect binding, if any.

    Block bindings come from `with effect(x=impl):` in compiled code and
    live on the active compile frame; they win over both caller overrides
    and module defaults because they are the most local binding.
    """
    frame = getattr(visitor, 'func_state', None)
    bindings_stack = (
        getattr(frame, 'block_effect_bindings', None)
        if frame is not None else None
    )
    if not bindings_stack:
        return None
    for bindings in reversed(bindings_stack):
        if effect_name in bindings:
            return bindings[effect_name]
    return None


def _module_default_impl(effect_name: str, caller_module: Optional[str]) -> Any:
    """Lookup per-module effect default registered via effect.default."""
    if not caller_module:
        return None
    eff = _current_effect()
    with eff._lock:
        module_defaults = eff._defaults.get(caller_module, {})
        return module_defaults.get(effect_name)


def _compile_context_has_override(effect_name: str, visitor=None) -> bool:
    """Check whether the compiling function has a caller override for the effect.

    Primary path: the compiling function's binding state, reached
    explicitly through the visitor.  Fallback: the planning-time dynamic
    scope (get_current_compilation_context) for calls without a visitor.
    """
    binding = getattr(visitor, 'binding_state', None) if visitor is not None else None
    if binding is not None:
        if not getattr(binding, 'effect_suffix', None):
            return False
        override_names = getattr(binding, 'effect_override_names', None) or set()
        return effect_name in override_names
    compilation_ctx = get_current_compilation_context()
    if not compilation_ctx:
        return False
    return effect_name in compilation_ctx.get('effect_override_names', set())


def _resolve_compile_implementation(manager, name, visitor):
    """Resolve scalar and namespace effects through the same binding rules."""
    block = _block_bound_impl(name, visitor)
    if block is not None:
        return block
    binding = getattr(visitor, 'binding_state', None)
    captured = getattr(binding, 'captured_effect_context', None) or {}
    if (manager.is_direct_assignment(name)
            or _compile_context_has_override(name, visitor)):
        return captured.get(name, manager._resolve_effect(name))
    default = _module_default_impl(name, _caller_module_from_visitor(visitor))
    return default if default is not None else manager._resolve_effect(name)


class EffectNamespace:
    """
    A namespace object that represents an effect category (e.g., rng, allocator).

    Attributes on this object forward to the current effect implementation.
    At compile time, effect.rng.next() is resolved to the actual function.

    Implements handle_attribute for compiler integration - allows the compiler
    to resolve effect.rng.next() to the actual function at compile time.
    """

    def __init__(self, name: str, implementation: Any = None, manager=None):
        object.__setattr__(self, '_name', name)
        object.__setattr__(self, '_impl', implementation)
        object.__setattr__(self, '_manager', manager)

    def __getattr__(self, attr: str) -> Any:
        impl = self._get_impl()
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
        impl = self._get_impl()
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
        manager = object.__getattribute__(self, '_manager')
        if manager is not None:
            return manager._resolve_effect(object.__getattribute__(self, '_name'))
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
        name = object.__getattribute__(self, '_name')

        # Record effect usage for transitive propagation
        # This tracks that the current function being compiled uses this effect
        record_effect_usage(name, getattr(visitor, 'func_state', None))

        manager = object.__getattribute__(self, '_manager')
        impl = (
            _resolve_compile_implementation(manager, name, visitor)
            if manager is not None else self._get_impl()
        )

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
        impl = self._get_impl()
        if impl is None:
            return f"<EffectNamespace '{name}' (unbound)>"
        return f"<EffectNamespace '{name}' -> {type(impl).__name__}>"


class EffectContext:
    """Context-local effect bindings for definitions and explicit imports.

    Binding an implementation stores its reference without modifying the
    implementation or its members. This applies to default and pinned
    registrations too; a callable can also be an ordinary specialization root.
    """

    def __init__(self, effect_manager: 'Effect', overrides: Dict[str, Any],
                 suffix: Optional[str] = None):
        self._effect = effect_manager
        self._overrides = MappingProxyType(dict(overrides))
        self._suffix = suffix

    def __enter__(self) -> 'EffectContext':
        manager = self._effect
        previous = manager._scope.get()
        bindings = dict(previous.bindings)
        for name, implementation in self._overrides.items():
            bindings[name] = (
                manager.get_effect_impl(name)
                if manager.is_direct_assignment(name) else implementation
            )
        suffixes = previous.suffixes
        if self._suffix is not None:
            suffixes += (self._suffix,)
        scope = _EffectScope(
            MappingProxyType(bindings), suffixes,
            previous.override_names | frozenset(self._overrides), previous.replay,
        )
        scope_token = manager._scope.set(scope)
        try:
            session = CompileSession.current()
        except RuntimeError:
            session = None
        if session is not None and session.effects is not manager:
            session = None
        import_token = _active_import_context.set((manager, self._suffix, session))
        suppression_token = _suppress_compile_suffix.set(False)
        _context_tokens.set(_context_tokens.get() + (
            (self, scope_token, import_token, suppression_token),
        ))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        tokens = _context_tokens.get()
        context, scope_token, import_token, suppression_token = tokens[-1]
        if context is not self:
            raise RuntimeError('Effect contexts must exit in reverse entry order')
        _context_tokens.set(tokens[:-1])
        _suppress_compile_suffix.reset(suppression_token)
        _active_import_context.reset(import_token)
        self._effect._scope.reset(scope_token)
        return False


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
        '_override_stack',
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
        object.__setattr__(self, '_import_binding_keys', {})
        object.__setattr__(self, '_scope', ContextVar(
            'pythoc_effect_scope', default=_EMPTY_SCOPE,
        ))

    @property
    def _suffix_stack(self) -> tuple:
        """Read-only suffixes in this execution context."""
        return self._scope.get().suffixes

    @property
    def _override_stack(self) -> tuple:
        """Read-only override names in this execution context."""
        names = self._scope.get().override_names
        return (names,) if names else ()

    def _capture_context(self):
        with self._lock:
            names = self._effects.keys() | self._scope.get().bindings.keys()
            return {
                name: value for name in names
                if (value := self._resolve_effect(name)) is not None
            }

    def _register_import_bindings(self, suffix, scope):
        """A suffix denotes one explicit import configuration per session."""
        from .type_descriptor import specialization_key
        bindings = {
            name: scope.bindings.get(name) for name in scope.override_names
        }
        key = specialization_key(bindings)
        with self._lock:
            existing = self._import_binding_keys.setdefault(suffix, key)
            if existing != key:
                raise ValueError(
                    f"Effect suffix {suffix!r} already has different import bindings"
                )

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

                if name not in self._direct_assignments:
                    self._set_effect(name, impl)

    def _set_effect(self, name: str, impl: Any,
                    is_direct: bool = False) -> None:
        """
        Internal: Set an effect implementation.

        Args:
            name: Effect name
            impl: Implementation object
            is_direct: True if this is a direct assignment (effect.xxx = impl)
        """
        with self._lock:
            # Create or update the effect namespace
            if name not in self._effects:
                self._effects[name] = EffectNamespace(name, impl, self)
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
            scope = self._scope.get()
            if scope.replay and name in scope.bindings:
                return scope.bindings[name]
            ns = self._effects.get(name)
            if name in self._direct_assignments and ns is not None:
                return object.__getattribute__(ns, '_impl')
            if name in scope.bindings:
                return scope.bindings[name]
            if caller_module:
                defaults = self._defaults.get(caller_module, {})
                if name in defaults:
                    return defaults[name]
            if ns is not None:
                return object.__getattribute__(ns, '_impl')
            return None

    def _get_current_suffix(self) -> Optional[str]:
        """Get the current suffix from the execution-context stack"""
        stack = self._suffix_stack
        if stack:
            return stack[-1]
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
            ns = EffectNamespace(name, None, self)
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

        record_effect_usage(attr_name, getattr(visitor, 'func_state', None))
        impl = _resolve_compile_implementation(self, attr_name, visitor)
        if impl is not None and isinstance(impl, (int, float, bool, str)):
            value_type = PythonType.wrap(impl, is_constant=True)
            return wrap_value(impl, kind="python", type_hint=value_type)
        ns = EffectNamespace(attr_name, impl)
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
        return self._resolve_effect(name)

    def has_effect(self, name: str) -> bool:
        """Check if an effect is bound"""
        return self._resolve_effect(name) is not None

    def list_effects(self) -> Dict[str, bool]:
        """
        List all registered effects and their bound status.

        Returns:
            Dict mapping effect name to whether it's bound
        """
        with self._lock:
            names = self._effects.keys() | self._scope.get().bindings.keys()
            return {name: self._resolve_effect(name) is not None for name in names}

    def is_direct_assignment(self, name: str) -> bool:
        """Check if an effect was set via direct assignment (non-overridable)"""
        with self._lock:
            return name in self._direct_assignments


# Guards lazy check-then-set of the active session's Effect instance.
_effect_creation_lock = threading.RLock()


def _current_effect() -> Effect:
    """Return the active session's Effect, creating and attaching it lazily.

    Raises RuntimeError when no compile session is active.
    """
    session = CompileSession.current()
    eff = session.effects
    if eff is None:
        with _effect_creation_lock:
            eff = session.effects
            if eff is None:
                eff = Effect()
                session.effects = eff
    return eff


class _EffectProxy:
    """Module-level ``effect`` object.

    A thin proxy: every attribute access, assignment, deletion, and call
    is forwarded to the Effect instance owned by the currently active
    CompileSession (created lazily on first access).  This keeps the
    public API (``effect.xxx``, ``effect.default(...)``, ``with
    effect(...)``) unchanged while the actual state is per-session.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        return getattr(_current_effect(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(_current_effect(), name, value)

    def __delattr__(self, name: str) -> None:
        delattr(_current_effect(), name)

    def __call__(self, *args, **kwargs) -> EffectContext:
        return _current_effect()(*args, **kwargs)

    def __repr__(self) -> str:
        session = CompileSession.active()
        if session is None or session.effects is None:
            return "<effect proxy (unbound)>"
        return repr(session.effects)


# Public module-level effect API; forwards to the active session's Effect.
effect = _EffectProxy()


def get_current_effect_suffix() -> Optional[str]:
    """
    Get the current effect suffix from the context stack.

    Used by the compiler to determine symbol naming.
    Returns None when suppress_effect_suffix() is active.
    """
    if is_effect_suffix_suppressed():
        return None
    return _current_effect()._get_current_suffix()


# Suppression is local to definition/import execution, before a compile frame
# exists. Explicit nested EffectContexts restore their own definition scope.
_suppress_compile_suffix = ContextVar('pythoc_suppress_effect_suffix', default=False)


@contextmanager
def suppress_effect_suffix():
    """Temporarily hide the effect suffix from @compile.

    Functions decorated with @compile inside this context will not pick up
    the current effect suffix. This is used by linearize() and the import
    hook to ensure that module-level @compile functions always produce the
    default (non-suffixed) version, even when they execute inside an active
    ``with effect(...)`` block.

    The suffix stack itself is NOT modified so that ``effect.default()``
    still correctly detects active caller overrides and preserves the
    caller's effect bindings.
    """
    token = _suppress_compile_suffix.set(True)
    try:
        yield
    finally:
        _suppress_compile_suffix.reset(token)


def is_effect_suffix_suppressed() -> bool:
    """Check if the effect suffix is currently suppressed for @compile."""
    return _suppress_compile_suffix.get()


def capture_effect_context() -> Dict[str, Any]:
    """
    Capture the current effect context for later restoration.

    Values are the exact bound implementations. Specialization rewrites a
    variant's ordinary compilation globals, never these provider references.

    Returns:
        Dict mapping effect name to implementation
    """
    return _current_effect()._capture_context()


def capture_effect_override_names() -> Set[str]:
    """Return the active caller override names from nested effect contexts."""
    return set(_current_effect()._scope.get().override_names)


@contextmanager
def restore_effect_context(captured: Dict[str, Any]):
    """
    Context manager to temporarily restore a captured effect context.

    Args:
        captured: Dict from capture_effect_context()
    """
    manager = _current_effect()
    scope = replace(
        manager._scope.get(), bindings=MappingProxyType(dict(captured)), replay=True,
    )
    token = manager._scope.set(scope)
    try:
        yield
    finally:
        manager._scope.reset(token)


# ---------------------------------------------------------------------------
# Always-on import edge recording for the EffectGraph
#
# Group-level dependencies must be obtained structurally — from import
# statements — not by scanning each function's globals.  The recorder below
# wraps builtins.__import__ for the whole process and, for every import,
# records (caller_file, None, None) -> NodeID edges for compiled wrappers
# visible at the imported module's top level. The same stable adapter handles
# effect imports; contexts only change its context-local binding request.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Group-owner tracking
#
# Import edges are only ever queried with caller nodes derived from real
# group keys, so recording imports issued by files that own no compiled
# group is wasted work.  Group creation registers the owning file here; the
# recorder below fast-path skips every other caller (this keeps pythoc's
# own function-local lazy imports, executed millions of times during
# compilation, out of the recorder).  Imports a file issued *before* its
# first group existed are backfilled at registration time by scanning the
# module namespace.
# ---------------------------------------------------------------------------

_group_owner_files: Set[str] = set()
# Negative cache for the recorder fast path: files probed and found to own no
# group.  register_group_owner discards entries here when a file gains its
# first group.
_non_owner_files: Set[str] = set()
_group_owner_lock = threading.Lock()


def register_group_owner(source_file):
    """Mark *source_file* as owning at least one compiled group.

    Called when a compilation group is created.  The first registration for
    a file backfills import edges for wrappers already visible in the file's
    module namespace (imports that ran before the first @compile).
    """
    if not source_file:
        return
    files = {source_file}
    try:
        files.add(os.path.abspath(source_file))
    except Exception:
        pass
    with _group_owner_lock:
        if not files - _group_owner_files:
            return
        _group_owner_files.update(files)
        _non_owner_files.difference_update(files)
    _backfill_import_edges(source_file)


def _resolve_active_output_manager():
    """Best-effort handle to the active output manager (mirrors the recorder)."""
    session = CompileSession.active()
    om = session.output_manager if session is not None else None
    if om is None:
        om_module = sys.modules.get(__name__.rsplit('.', 1)[0] + '.build.output_manager')
        if om_module is None:
            return None
        om = om_module.get_output_manager()
    return om


def _backfill_import_edges(source_file):
    """Record edges for compiled wrappers already visible to *source_file*.

    Mirrors what the live recorder records for the module's import
    statements: names bound directly in the module namespace (from-imports),
    plus the top level of modules bound there (plain imports), recursing
    into packages (`import a.b.c` binds the top package but the edges point
    at wrappers visible in the leaf).
    """
    try:
        om = _resolve_active_output_manager()
        if om is None:
            return
        graph = om._effect_graph
        module = None
        try:
            abs_source = os.path.abspath(source_file)
        except Exception:
            abs_source = source_file
        for mod in list(sys.modules.values()):
            if not isinstance(mod, ModuleType):
                continue
            mod_file = ModuleType.__getattribute__(mod, '__dict__').get('__file__')
            if not mod_file:
                continue
            if mod_file == source_file or mod_file == abs_source:
                module = mod
                break
            try:
                if os.path.abspath(mod_file) == abs_source:
                    module = mod
                    break
            except Exception:
                continue
        if module is None:
            return
        caller_node = (source_file, None, None)
        visited = set()

        def scan_module(mod, depth):
            if id(mod) in visited or depth > 8:
                return
            visited.add(id(mod))
            namespace = ModuleType.__getattribute__(mod, '__dict__')
            for value in list(namespace.values()):
                if isinstance(value, ModuleType):
                    nested = ModuleType.__getattribute__(value, '__dict__')
                    if nested.get('__path__') is not None:
                        scan_module(value, depth + 1)
                else:
                    _record_wrapper_import_edge(graph, caller_node, source_file, value)

        namespace = ModuleType.__getattribute__(module, '__dict__')
        for value in list(namespace.values()):
            if isinstance(value, ModuleType):
                mod_file = ModuleType.__getattribute__(value, '__dict__').get('__file__')
                if not mod_file or mod_file == source_file or mod_file == abs_source:
                    continue
                scan_module(value, 0)
            else:
                _record_wrapper_import_edge(graph, caller_node, source_file, value)
    except Exception:
        pass  # Best-effort, same contract as the live recorder.


def _record_wrapper_import_edge(graph, caller_node, caller_file, value):
    """Record caller_node -> value's group node if value is a compiled wrapper."""
    metadata = _callable_metadata(value)
    if not metadata.get('_is_compiled', False):
        return
    binding = metadata.get('_binding', metadata.get('_state'))
    group_key = getattr(binding, 'group_key', None) if binding else None
    if not group_key or group_key[0] == caller_file:
        # Same-file dependencies are not cross-group planning edges.
        return
    target_node = node_id_from_group_key(group_key)
    if target_node != caller_node:
        graph.add_edge(caller_node, target_node)


def _record_import_edges(name, caller_globals, fromlist, level, module):
    """Record caller -> imported dependency edges for one __import__ call."""
    if not isinstance(caller_globals, dict):
        return
    caller_file = caller_globals.get('__file__')
    if not caller_file:
        return
    if caller_file not in _group_owner_files:
        # Fast path: files that own no compiled group can never be the
        # source of a planning edge (edges are only queried from real group
        # nodes).  Imports such a file issued before its first group existed
        # are backfilled by register_group_owner().
        if caller_file in _non_owner_files:
            return
        if os.path.isabs(caller_file):
            _non_owner_files.add(caller_file)
            return
        abs_file = os.path.abspath(caller_file)
        if abs_file not in _group_owner_files:
            _non_owner_files.add(caller_file)
            return
        caller_file = abs_file
    # Best-effort: imports may happen before any session is installed
    # (e.g. pythoc's own import chain); skip recording in that case.
    session = CompileSession.active()
    if session is None:
        return
    # Resolve the output manager through the active session first.  While
    # the session does not own one yet, fall back to the sys.modules lookup:
    # importing output_manager here would recurse into this recorder, and
    # during pythoc's own init it may not be importable yet.  Edges of
    # modules loaded that early are recorded when user code imports them
    # afterwards.
    om = session.output_manager
    if om is None:
        om_module = sys.modules.get(__name__.rsplit('.', 1)[0] + '.build.output_manager')
        if om_module is None:
            return
        om = om_module.get_output_manager()
    graph = om._effect_graph
    caller_node = (caller_file, None, None)

    modules = []
    if not isinstance(module, ModuleType):
        return
    namespace = ModuleType.__getattribute__(module, '__dict__')
    modules.append(module)
    if fromlist:
        for attr_name in fromlist:
            if attr_name == '*':
                continue
            if attr_name not in namespace:
                continue
            attr = namespace[attr_name]
            if isinstance(attr, ModuleType):
                modules.append(attr)
            else:
                _record_wrapper_import_edge(graph, caller_node, caller_file, attr)
    elif level == 0 and name:
        # `import a.b.c` binds the top package; the leaf module holds the attrs.
        leaf = sys.modules.get(name)
        if leaf is not None and leaf not in modules:
            modules.append(leaf)

    for mod in modules:
        mod_namespace = ModuleType.__getattribute__(mod, '__dict__')
        mod_file = mod_namespace.get('__file__')
        if not mod_file or mod_file == caller_file:
            continue
        for value in list(mod_namespace.values()):
            _record_wrapper_import_edge(graph, caller_node, caller_file, value)


@contextmanager
def _import_request_scope(active, caller_globals):
    manager, suffix, owner_session = active
    try:
        session = CompileSession.current()
    except RuntimeError:
        session = None
    if owner_session is not None and owner_session is not session:
        yield None
        return
    scope = manager._scope.get()
    caller_file = (
        caller_globals.get('__file__') if isinstance(caller_globals, dict) else None
    )
    request = (
        _EffectImportRequest(manager, scope, suffix, caller_file) if suffix else None
    )
    scope_token = manager._scope.set(_EMPTY_SCOPE)
    import_token = _active_import_context.set(None)
    try:
        with suppress_effect_suffix():
            yield request
    finally:
        _active_import_context.reset(import_token)
        manager._scope.reset(scope_token)


def _make_import_hook(delegate):
    def _recording_import(name, globals=None, locals=None, fromlist=(), level=0):
        """Delegate ordinary imports and bind only this from-import's exports."""
        active = _active_import_context.get()
        request = None
        if active is None:
            module = delegate(name, globals, locals, fromlist, level)
        else:
            with _import_request_scope(active, globals) as request:
                module = delegate(name, globals, locals, fromlist, level)
        if request is not None:
            # Module initialization can register defaults and pinned effects.
            # Resolve them against the restored caller scope before freezing
            # this import's bindings; no selection is deferred to a call.
            request = request.capture()
        try:
            _record_import_edges(name, globals, fromlist, level, module)
        except Exception:
            pass  # Dependency observation must not change Python import behavior.
        if request is not None:
            return _bind_import_exports(module, fromlist, request)
        return module

    return _recording_import


def _install_import_hook():
    """Install once, preserving the preceding importer and later wrappers."""
    installed = globals().get('_installed_import_hook')
    if installed is None:
        installed = _make_import_hook(builtins.__import__)
        globals()['_installed_import_hook'] = installed
        builtins.__import__ = installed
    return installed


_recording_import = _install_import_hook()
