# -*- coding: utf-8 -*-
"""
@compile decorator: Compile Python functions to native code.

Key design decisions (from dependency_cache.md):
- Group Key: 4-tuple (source_file, scope, compile_suffix, effect_suffix)
- compile_suffix: from @compile(suffix=T), NOT contagious
- effect_suffix: from with effect(suffix=X), IS contagious to transitive calls
- Mangled names: name_{compile_suffix}_{effect_suffix}
- Same effect_suffix = same implementation = reused across all callers
"""
from functools import wraps
import inspect
import os
import ast
import sys
from typing import Any, List, Optional

from ..call_normalization import pack_native_call_args

# Sentinel value to distinguish "not provided" from "provided as None"
_SCOPE_NOT_PROVIDED = object()

from ..compiler import LLVMCompiler
from ..registry import register_struct_from_class, _unified_registry
from ..context import FunctionBindingState

from .structs import (
    add_struct_handle_call as _add_struct_handle_call,
    compile_dynamic_class as _compile_dynamic_class,
)
from .mangling import mangle_function_name as _mangle_function_name

# Import new utility modules
from ..utils import (
    find_caller_frame,
    get_definition_scope,
    sanitize_filename,
    get_build_paths,
    normalize_suffix,
    get_function_file_and_source,
    get_function_start_line,
)
from ..build import (
    get_output_manager,
    flush_all_pending_outputs,
)
from ..logger import logger, set_source_context


DEFAULT_EFFECT_KEY = "__default__"


def _is_parametric_type(pc_type):
    """Check whether a resolved PC type is the ``param`` compile-time parameter type."""
    return getattr(pc_type, '_is_param', False)


def _make_parametric_suffix(param_values):
    """Build a stable compile suffix from a tuple of parametric argument values."""
    return normalize_suffix(tuple(param_values))


def _extract_raw_annotations(func_ast):
    """Extract raw annotation AST nodes from a function definition.

    Returns a dict mapping parameter names to their annotation AST nodes and
    the key ``'return'`` to the return annotation node.  These raw nodes are
    kept for parametric functions so that annotations referencing parameter
    names can be re-resolved at specialization time.
    """
    raw = {}
    if func_ast.returns is not None:
        raw['return'] = func_ast.returns
    for arg in func_ast.args.args:
        if arg.annotation is not None:
            raw[arg.arg] = arg.annotation
    return raw


def _remove_parametric_params_from_ast(func_ast, parametric_param_names):
    """Return a new AST with parametric parameters removed.

    The returned AST is a deep copy.  Parametric parameters are removed from
    ``args.args`` together with their corresponding defaults.  Phase 1 does
    not support defaults on parametric parameters; an error is raised if one
    is found.
    """
    import copy

    func_ast = copy.deepcopy(func_ast)
    n_args = len(func_ast.args.args)
    n_defaults = len(func_ast.args.defaults)
    default_start = n_args - n_defaults

    new_args = []
    new_defaults = []
    for i, arg in enumerate(func_ast.args.args):
        is_parametric = arg.arg in parametric_param_names
        if is_parametric:
            if i >= default_start:
                logger.error(
                    f"parametric parameter '{arg.arg}' cannot have a default value",
                    node=arg, exc_type=TypeError,
                )
            continue
        new_args.append(arg)
        if i >= default_start:
            new_defaults.append(func_ast.args.defaults[i - default_start])

    func_ast.args.args = new_args
    func_ast.args.defaults = new_defaults
    return func_ast


def _param_value_to_ast(value):
    """Convert a compile-time parametric value into an AST expression node.

    Supports:
      - PythoC builtin types (e.g. ``i32``) -> ``Name`` node using the type name
      - Python constants (int, float, str, bool, None) -> ``Constant`` node

    Returns ``None`` for values that cannot be expressed as a standalone AST
    expression in Phase 1.
    """
    from ..builtin_entities import BuiltinEntity

    if isinstance(value, type) and issubclass(value, BuiltinEntity):
        return ast.Name(id=value.get_name(), ctx=ast.Load())
    if isinstance(value, BuiltinEntity):
        return ast.Name(id=value.get_name(), ctx=ast.Load())
    if isinstance(value, (int, float, str, bool)) or value is None:
        return ast.Constant(value=value)
    return None


class _ParametricNameSubstitutor(ast.NodeTransformer):
    """Substitute parametric parameter names with their compile-time values.

    This is needed for yield-based parametric functions because their bodies
    are inlined into the caller's AST; the caller does not have the parametric
    names in scope.  Replacing the names with concrete AST expressions (type
    names for type params, constants for value params) makes the inlined body
    self-contained.
    """

    def __init__(self, param_values_by_name):
        self.param_values_by_name = param_values_by_name
        self.substitutions = {
            name: _param_value_to_ast(value)
            for name, value in param_values_by_name.items()
        }

    def visit_Name(self, node):
        replacement = self.substitutions.get(node.id)
        if replacement is not None:
            import copy
            return copy.deepcopy(replacement)
        return node


def _substitute_parametric_names_in_ast(func_ast, param_values_by_name):
    """Return a new AST with parametric parameter names replaced by values."""
    import copy

    func_ast = copy.deepcopy(func_ast)
    substitutor = _ParametricNameSubstitutor(param_values_by_name)
    return substitutor.visit(func_ast)


def _resolve_annotations_with_params(raw_annotations, user_globals, param_values_by_name, parametric_names):
    """Resolve annotations using a namespace that includes parametric values.

    Annotations for the parametric parameters themselves (e.g. ``T: param``)
    are skipped because ``param`` is not a runtime type and those parameters
    are removed from the specialized AST.
    """
    from ..type_resolver import TypeResolver

    namespace = dict(user_globals)
    namespace.update(param_values_by_name)
    resolver = TypeResolver(module_context=None, user_globals=namespace)

    resolved = {}
    for name, ann_node in raw_annotations.items():
        if name in parametric_names:
            continue
        resolved[name] = resolver.parse_annotation(ann_node)
    return resolved


def _compile_parametric_specialization(
    factory_wrapper,
    param_values,
):
    """Compile one specialization of a parametric function.

    This is invoked by the factory with the concrete parametric argument values.
    It transforms the original AST (removing parametric parameters), resolves
    annotations in a namespace where the parametric names are bound, and uses
    ``meta.compile_api.compile_ast`` to produce a normal compiled wrapper.
    """
    from ..meta.compile_api import compile_ast

    param_names = factory_wrapper._parametric_param_names
    if len(param_names) != len(param_values):
        raise TypeError(
            f"expected {len(param_names)} parametric arguments, got {len(param_values)}"
        )

    param_values_by_name = dict(zip(param_names, param_values))

    # Build suffix: combine user-provided compile suffix with parametric suffix.
    param_suffix = _make_parametric_suffix(param_values)
    original_suffix = factory_wrapper._original_compile_suffix
    if original_suffix and param_suffix:
        suffix = f"{original_suffix}_{param_suffix}"
    elif original_suffix:
        suffix = original_suffix
    else:
        suffix = param_suffix

    # Transform AST: drop parametric parameters.
    spec_ast = _remove_parametric_params_from_ast(
        factory_wrapper._original_func_ast,
        set(param_names),
    )

    # Resolve annotations with parametric names in scope.
    raw_annotations = factory_wrapper._raw_annotations
    resolved = _resolve_annotations_with_params(
        raw_annotations,
        factory_wrapper._original_user_globals,
        param_values_by_name,
        set(param_names),
    )
    return_type = resolved.get('return')
    param_types = {
        name: resolved[name]
        for name in factory_wrapper._concrete_param_names
    }

    # Augment globals so the body sees parametric names as compile-time values.
    spec_globals = dict(factory_wrapper._original_user_globals)
    spec_globals.update(param_values_by_name)

    source_code = factory_wrapper._original_source_code
    try:
        source_code = ast.unparse(spec_ast)
    except Exception:
        pass

    # Use a synthetic source file for each specialization.  PythoC's build
    # manager locks a source file once its shared library has been loaded, so
    # defining new compiled functions for the original file after native
    # execution has started is not allowed.  Parametric specializations are
    # created lazily at call sites, so we place them in their own file.
    # The synthetic file must be unique per (function, suffix) pair.
    synthetic_source_file = "{}.__parametric__.{}.{}".format(
        factory_wrapper._source_file,
        factory_wrapper._original_name,
        suffix if suffix else "default",
    )

    # Yield-based generators must be returned as placeholders that trigger
    # inline expansion at call sites, just like non-parametric yield functions.
    # Because the inlined body is spliced into the caller, any reference to a
    # parametric parameter name must be replaced by its compile-time value so
    # the caller can resolve it.
    from ..ast_visitor.yield_transform import analyze_yield_function
    yield_analyzer = analyze_yield_function(spec_ast)
    if yield_analyzer:
        from ..ast_visitor.yield_transform import create_yield_iterator_wrapper

        inlined_ast = _substitute_parametric_names_in_ast(spec_ast, param_values_by_name)
        try:
            source_code = ast.unparse(inlined_ast)
        except Exception:
            pass

        class _YieldFuncStub:
            __name__ = factory_wrapper._original_name
            __globals__ = spec_globals

        return create_yield_iterator_wrapper(
            _YieldFuncStub(), inlined_ast, yield_analyzer, spec_globals,
            synthetic_source_file, _get_registry(),
        )

    return compile_ast(
        spec_ast,
        param_types=param_types,
        return_type=return_type,
        name=factory_wrapper._original_name,
        suffix=suffix,
        attrs=factory_wrapper._original_fn_attrs,
        source_file=synthetic_source_file,
        source_code=source_code,
        start_line=factory_wrapper._start_line,
        user_globals=spec_globals,
        effect_suffix=factory_wrapper._original_effect_suffix,
        copy_ast=False,
    )


def _create_parametric_factory_wrapper(
    func,
    func_ast,
    source_file,
    source_code,
    start_line,
    user_globals,
    param_names,
    parametric_param_names,
    fn_attrs,
    compile_suffix,
    effect_suffix,
):
    """Create a factory wrapper for a function with parametric parameters.

    The returned object is callable both from Python and from compiled code.
    Calling it with all arguments splits out the parametric values, produces a
    specialization, and forwards the remaining arguments.
    """
    parametric_set = set(parametric_param_names)
    parametric_indices = [i for i, name in enumerate(param_names) if name in parametric_set]
    concrete_indices = [i for i, name in enumerate(param_names) if name not in parametric_set]
    concrete_param_names = [param_names[i] for i in concrete_indices]

    raw_annotations = _extract_raw_annotations(func_ast)

    n_parametric = len(parametric_param_names)

    @wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) == n_parametric and n_parametric < len(param_names):
            # Partial application: bind compile-time params and return the
            # specialized wrapper; concrete arguments will be supplied later.
            return wrapper._factory_func(*args)

        if len(args) != len(param_names):
            raise TypeError(
                f"{func.__name__}() takes {len(param_names)} positional arguments "
                f"but {len(args)} were given"
            )
        param_values = [args[i] for i in parametric_indices]
        concrete_args = [args[i] for i in concrete_indices]
        specialized = wrapper._factory_func(*param_values)
        return specialized(*concrete_args)

    wrapper._is_compiled = True
    wrapper._is_parametric = True
    wrapper._parametric_indices = parametric_indices
    wrapper._concrete_indices = concrete_indices
    wrapper._parametric_param_names = parametric_param_names
    wrapper._concrete_param_names = concrete_param_names
    wrapper._param_names = param_names
    wrapper._original_name = func.__name__
    wrapper._original_func_ast = func_ast
    wrapper._original_source_code = source_code
    wrapper._original_user_globals = user_globals
    wrapper._raw_annotations = raw_annotations
    wrapper._original_fn_attrs = set(fn_attrs) if fn_attrs else set()
    wrapper._original_compile_suffix = compile_suffix
    wrapper._original_effect_suffix = effect_suffix
    wrapper._source_file = source_file
    wrapper._start_line = start_line

    _specialization_cache = {}

    def factory(*param_values):
        key = tuple(param_values)
        if key in _specialization_cache:
            return _specialization_cache[key]

        # Validate that parametric arguments are hashable (needed for suffix/cache).
        for v in param_values:
            try:
                hash(v)
            except TypeError:
                raise TypeError(
                    f"parametric argument {v!r} is not hashable and cannot be used "
                    f"as a compile-time parameter"
                )

        specialized = _compile_parametric_specialization(wrapper, param_values)
        _specialization_cache[key] = specialized
        return specialized

    wrapper._factory_func = factory

    def handle_call(visitor, func_ref, args, node):
        if len(args) != len(param_names):
            logger.error(
                f"{func.__name__}() takes {len(param_names)} positional arguments "
                f"but {len(args)} were given",
                node=node, exc_type=TypeError,
            )

        param_args = [args[i] for i in parametric_indices]
        concrete_args = [args[i] for i in concrete_indices]

        param_values = []
        for arg in param_args:
            if not arg.is_python_value():
                logger.error(
                    "parametric argument must be a compile-time Python value",
                    node=node, exc_type=TypeError,
                )
            param_values.append(arg.get_python_value())

        specialized = wrapper._factory_func(*param_values)

        # Yield placeholders cannot be lowered like normal compiled wrappers;
        # they must return inline-expansion metadata to the for-loop visitor.
        # The specialized AST has parametric parameters removed, so the call
        # node forwarded to the placeholder must only contain concrete args.
        if getattr(specialized, '_is_yield_generated', False):
            import copy
            specialized_call_node = copy.deepcopy(node)
            specialized_call_node.args = [node.args[i] for i in concrete_indices]
            return specialized.handle_call(
                visitor, func_ref, concrete_args, specialized_call_node
            )

        from ..call_normalization import lower_compile_handle_call
        return lower_compile_handle_call(specialized, visitor, func_ref, concrete_args, node)

    wrapper.handle_call = handle_call
    return wrapper


def _materialize_single_specialization(template_wrapper, effect_key, effect_bindings):
    """Materialize one wrapper specialization and queue its compilation."""
    state = getattr(template_wrapper, '_binding', template_wrapper._state)
    func_info = getattr(template_wrapper, '_func_info', None)
    if effect_key in state.effect_specialized_cache:
        return state.effect_specialized_cache[effect_key]

    if effect_key == DEFAULT_EFFECT_KEY:
        # Default: queue the template's own compile_callback in-place
        om = get_output_manager()
        om.queue_compilation(
            state.group_key,
            state.template_compile_callback,
            func_info,
        )
        state.is_template = False
        state.effect_specialized_cache[DEFAULT_EFFECT_KEY] = template_wrapper
        logger.debug(f"Materialized default specialization: {state.original_name}")
        return template_wrapper
    # Non-default: call _compile_impl with effect_suffix
    from ..effect import capture_effect_override_names, restore_effect_context
    from ..effect import get_current_compilation_context
    original_scope = state.group_key[1] if state.group_key else None
    compilation_ctx = get_current_compilation_context()
    if compilation_ctx:
        effect_override_names = compilation_ctx.get('effect_override_names', set())
    else:
        effect_override_names = capture_effect_override_names()

    logger.debug(f"Materializing specialization: {state.original_name}_{effect_key}")

    with restore_effect_context(effect_bindings):
        specialized_wrapper = _compile_impl(
            template_wrapper.__wrapped__,
            compile_suffix=state.compile_suffix,
            effect_suffix=effect_key,
            captured_symbols=state.captured_symbols,
            effect_scope=original_scope,
            effect_override_names=effect_override_names,
        )

    state.effect_specialized_cache[effect_key] = specialized_wrapper
    return specialized_wrapper


def _iter_global_compiled_wrappers(group_wrappers):
    """Yield compiled wrappers referenced by group wrapper global namespaces."""
    seen = set()
    for wrapper in group_wrappers:
        binding = getattr(wrapper, '_binding', getattr(wrapper, '_state', None))
        globals_dict = getattr(binding, 'compilation_globals', None) if binding else None
        if not isinstance(globals_dict, dict):
            continue
        for value in globals_dict.values():
            candidates = [value]
            value_dict = getattr(value, '__dict__', None)
            if isinstance(value_dict, dict):
                candidates.extend(value_dict.values())
            for candidate in candidates:
                if not callable(candidate) or not getattr(candidate, '_is_compiled', False):
                    continue
                if id(candidate) in seen:
                    continue
                seen.add(id(candidate))
                yield candidate


def _effect_specialized_equivalent(value, effect_key):
    """Return value's already-materialized specialization for effect_key."""
    if not callable(value) or not getattr(value, '_is_compiled', False):
        return None
    binding = getattr(value, '_binding', getattr(value, '_state', None))
    if binding is None:
        return None
    if binding.effect_suffix == effect_key:
        return value
    if binding.effect_suffix is not None:
        return None
    return binding.effect_specialized_cache.get(effect_key)


def _copy_with_effect_specialized_attrs(value, effect_key):
    """Return a shallow namespace/copy with compiled attrs specialized."""
    from types import ModuleType
    if isinstance(value, ModuleType):
        return None

    value_dict = getattr(value, '__dict__', None)
    if value_dict is None or not hasattr(value_dict, 'items'):
        return None

    # Do not turn PythoC types/enums into namespaces; those are semantic types,
    # not import namespace carriers.
    if isinstance(value, type) and (
        hasattr(value, 'get_llvm_type')
        or hasattr(value, '_field_types')
        or hasattr(value, '_enum_info')
    ):
        return None

    replacements = {}
    for attr_name in value_dict:
        if attr_name.startswith('__'):
            continue
        try:
            attr_value = getattr(value, attr_name)
        except Exception:
            continue
        replacement = _effect_specialized_equivalent(attr_value, effect_key)
        if replacement is not None and replacement is not attr_value:
            replacements[attr_name] = replacement

    if not replacements:
        return None

    from types import SimpleNamespace

    if isinstance(value, type):
        attrs = {}
        for attr_name in value_dict:
            if attr_name.startswith('__'):
                continue
            try:
                attrs[attr_name] = getattr(value, attr_name)
            except Exception:
                pass
        attrs.update(replacements)
        return SimpleNamespace(**attrs)

    import copy
    try:
        cloned = copy.copy(value)
        for attr_name, replacement in replacements.items():
            setattr(cloned, attr_name, replacement)
        return cloned
    except Exception:
        attrs = {}
        for attr_name in value_dict:
            if attr_name.startswith('__'):
                continue
            try:
                attrs[attr_name] = getattr(value, attr_name)
            except Exception:
                pass
        attrs.update(replacements)
        return SimpleNamespace(**attrs)


def _rewrite_globals_to_effect_specializations(wrappers, effect_key, by_name=None):
    """Point specialized wrappers' captured globals at specialized imports."""
    replacement_cache = {}

    def get_replacement(value):
        cache_key = id(value)
        if cache_key in replacement_cache:
            return replacement_cache[cache_key]

        replacement = _effect_specialized_equivalent(value, effect_key)
        if replacement is None:
            replacement = _copy_with_effect_specialized_attrs(value, effect_key)
        replacement_cache[cache_key] = replacement
        return replacement

    for wrapper in wrappers:
        binding = getattr(wrapper, '_binding', getattr(wrapper, '_state', None))
        globals_dict = getattr(binding, 'compilation_globals', None) if binding else None
        if not isinstance(globals_dict, dict):
            continue

        if by_name:
            globals_dict.update(by_name)

        for name, value in list(globals_dict.items()):
            replacement = get_replacement(value)
            if replacement is not None and replacement is not value:
                globals_dict[name] = replacement


def materialize_group_specialization(template_wrapper, effect_key, effect_bindings, _visited=None):
    """Materialize a whole compilation group for one effect suffix.

    Effect specialization is group-level for build planning: once any symbol in
    a group is required under an effect suffix, all non-effect-implementation
    wrappers from the corresponding base group are materialized before object
    build starts.  Imported compiled wrappers referenced from the group's global
    namespace are conservatively expanded too; this is group/import based and
    does not require call-site visitation.
    """
    if effect_key == DEFAULT_EFFECT_KEY:
        return _materialize_single_specialization(
            template_wrapper, effect_key, effect_bindings,
        )

    state = getattr(template_wrapper, '_binding', template_wrapper._state)
    group_key = state.group_key
    if not group_key or len(group_key) != 4:
        return _materialize_single_specialization(
            template_wrapper, effect_key, effect_bindings,
        )

    base_group_key = (group_key[0], group_key[1], group_key[2], None)
    if _visited is None:
        _visited = set()
    if base_group_key in _visited:
        return state.effect_specialized_cache.get(effect_key) or template_wrapper
    _visited.add(base_group_key)

    om = get_output_manager()
    group_wrappers = om.get_group_wrappers(base_group_key)
    if not group_wrappers:
        return _materialize_single_specialization(
            template_wrapper, effect_key, effect_bindings,
        )

    eligible_wrappers = []
    for wrapper in group_wrappers:
        binding = getattr(wrapper, '_binding', getattr(wrapper, '_state', None))
        if binding is None or binding.effect_suffix is not None:
            continue
        if getattr(wrapper, '_pc_effect_impl', False):
            continue
        eligible_wrappers.append(wrapper)

    wrapper_ids = tuple(id(wrapper) for wrapper in eligible_wrappers)
    if om.get_group_effect_specialization(base_group_key, effect_key, wrapper_ids):
        return state.effect_specialized_cache.get(effect_key) or template_wrapper

    selected = None
    specialized_by_name = {}
    specialized_wrappers = []
    for wrapper in eligible_wrappers:
        binding = getattr(wrapper, '_binding', getattr(wrapper, '_state', None))
        specialized = _materialize_single_specialization(
            wrapper, effect_key, effect_bindings,
        )
        spec_binding = getattr(specialized, '_binding', getattr(specialized, '_state', None))
        if spec_binding is not None:
            spec_binding.is_group_specialization = True
            specialized_wrappers.append(specialized)
            specialized_by_name[binding.original_name] = specialized
        if wrapper is template_wrapper:
            selected = specialized

    # Keep same-group lexical calls inside the specialized group.  Without this,
    # a specialized function whose body calls another function from the same
    # source file would still see the default wrapper in its captured globals.
    _rewrite_globals_to_effect_specializations(
        specialized_wrappers, effect_key, specialized_by_name,
    )

    for imported_wrapper in _iter_global_compiled_wrappers(group_wrappers):
        binding = getattr(imported_wrapper, '_binding', getattr(imported_wrapper, '_state', None))
        if binding is None:
            continue
        if getattr(imported_wrapper, '_pc_effect_impl', False):
            continue
        if binding.effect_suffix is not None:
            if binding.effect_suffix != effect_key or not binding.group_key or len(binding.group_key) != 4:
                continue
            imported_base_key = (
                binding.group_key[0], binding.group_key[1], binding.group_key[2], None
            )
            imported_base_wrappers = om.get_group_wrappers(imported_base_key)
            if not imported_base_wrappers:
                continue
            imported_wrapper = imported_base_wrappers[0]
        materialize_group_specialization(
            imported_wrapper, effect_key, effect_bindings, _visited,
        )

    # Imported wrappers and factory-captured wrappers have now been materialized
    # recursively.  Rewrite closure/global references such as linearize's
    # ``acquire_func`` and cross-module imports (e.g. c_parser -> c_ast) to the
    # matching effect-specialized bindings before object compilation starts.
    _rewrite_globals_to_effect_specializations(
        specialized_wrappers, effect_key, specialized_by_name,
    )

    om.record_group_effect_specialization(
        base_group_key, effect_key, wrapper_ids, specialized_by_name,
    )

    return selected or state.effect_specialized_cache.get(effect_key) or template_wrapper


def materialize_specialization(template_wrapper, effect_key, effect_bindings):
    """Materialize a specialization from a wrapper.

    Non-default effect specialization is intentionally group-level so object
    build sees a frozen symbol set instead of discovering extra functions during
    codegen.
    """
    if effect_key == DEFAULT_EFFECT_KEY:
        return _materialize_single_specialization(
            template_wrapper, effect_key, effect_bindings,
        )
    return materialize_group_specialization(
        template_wrapper, effect_key, effect_bindings,
    )


def _get_registry():
    return _unified_registry


def get_compiler(source_file, user_globals, has_suffix=False):
    """Get or create compiler instance for source file.
    
    Args:
        source_file: Source file path
        user_globals: User globals for compilation
        has_suffix: If True, always create new compiler (suffix group)
    """
    registry = _get_registry()
    if has_suffix:
        # Suffix group: new compiler instance
        compiler = LLVMCompiler(user_globals=user_globals)
    else:
        # No suffix: reuse existing compiler for source file if available
        existing_compiler = registry.get_compiler(source_file)
        if existing_compiler:
            compiler = existing_compiler
            compiler.update_globals(user_globals)
        else:
            compiler = LLVMCompiler(user_globals=user_globals)
            registry.register_compiler(source_file, compiler)
    return compiler


def compile(func_or_class=None, suffix=None, attrs=None,
            _effect_suffix=None, _effect_scope=_SCOPE_NOT_PROVIDED):
    """
    Compile a Python function or class to native code.
    
    Args:
        func_or_class: Function or class to compile
        suffix: Explicit compile_suffix for function naming (from @compile(suffix=T))
                This is NOT contagious - each function chooses its own compile_suffix.
        attrs: Set of LLVM function-level attributes (e.g. {'readnone', 'nounwind'}).
               Applied to cross-module `declare` so the optimizer can treat calls
               as pure/no-side-effect, enabling CSE and store forwarding.
        _effect_suffix: Internal parameter for effect override (from with effect(suffix=X)).
                This IS contagious - propagates to transitive calls that use effects.
        _effect_scope: Internal parameter for transitive effect compilation.
                When generating effect-specialized versions from within pythoc internals,
                this preserves the original function's scope. Use _SCOPE_NOT_PROVIDED
                sentinel to distinguish "not provided" from "provided as None" (module-level).
                
    Group Key Design (4-tuple):
        (source_file, scope, compile_suffix, effect_suffix)
        
        - source_file: Always the file where the function is defined (NOT caller's file)
        - scope: The enclosing factory function name with .<locals> suffix
        - compile_suffix: From @compile(suffix=T), for generic instantiation
        - effect_suffix: From with effect(suffix=X), for effect override versions
        
    Symbol Naming:
        - name_{compile_suffix}_{effect_suffix} (compile_suffix first, then effect_suffix)
        - e.g., add_i32_mock for @compile(suffix=i32) with effect(suffix="mock")
    """
    # Capture all visible symbols (globals + locals) at decoration time
    from .visible import capture_caller_symbols
    captured_symbols = capture_caller_symbols(depth=1)
    
    # Normalize compile_suffix early
    compile_suffix = normalize_suffix(suffix)
    fn_attrs = set(attrs) if attrs else set()
    
    # Get effect_suffix from context if not explicitly provided
    if _effect_suffix is None:
        from ..effect import get_current_effect_suffix
        effect_suffix = get_current_effect_suffix()
    else:
        effect_suffix = _effect_suffix
    
    if func_or_class is None:
        def decorator(f):
            return _compile_impl(f, 
                                compile_suffix=compile_suffix,
                                effect_suffix=effect_suffix,
                                captured_symbols=captured_symbols,
                                effect_scope=_effect_scope,
                                fn_attrs=fn_attrs)
        return decorator

    return _compile_impl(func_or_class, 
                        compile_suffix=compile_suffix,
                        effect_suffix=effect_suffix,
                        captured_symbols=captured_symbols,
                        effect_scope=_effect_scope,
                        fn_attrs=fn_attrs)


def _compile_impl(func_or_class, 
                  compile_suffix: Optional[str] = None, 
                  effect_suffix: Optional[str] = None,
                  captured_symbols=None,
                  effect_scope=_SCOPE_NOT_PROVIDED,
                  effect_override_names=None,
                  fn_attrs=None):
    """Internal implementation of compile decorator.
    
    Uses 4-tuple group_key: (source_file, scope, compile_suffix, effect_suffix)
    
    Args:
        func_or_class: Function or class to compile
        compile_suffix: From @compile(suffix=T), NOT contagious
        effect_suffix: From effect context, IS contagious to transitive calls
        captured_symbols: Symbols captured at decoration time
        effect_scope: Override scope for transitive effect compilation.
                     _SCOPE_NOT_PROVIDED means use get_definition_scope().
                     None means module-level scope.
        fn_attrs: Set of LLVM function-level attributes for cross-module declares.
    """
    if inspect.isclass(func_or_class):
        return _compile_dynamic_class(
            func_or_class,
            suffix=compile_suffix,
            captured_symbols=captured_symbols,
        )

    func = func_or_class

    from ..native_executor import get_multi_so_executor
    executor = get_multi_so_executor()    

    @wraps(func)
    def wrapper(*args, **kwargs):
        binding = getattr(wrapper, '_binding', getattr(wrapper, '_state', None))
        if binding and binding.is_template:
            materialize_specialization(wrapper, DEFAULT_EFFECT_KEY, {})

        if not hasattr(wrapper, '_native_func'):
            wrapper._native_func = executor.execute_function(wrapper)

        args = pack_native_call_args(wrapper, args, kwargs)
        return wrapper._native_func(*args)
    
    source_file, source_code = get_function_file_and_source(func)
    
    # Get function start line for accurate error messages
    start_line = get_function_start_line(func)
    set_source_context(source_file, start_line - 1)

    registry = _get_registry()
    from .visible import get_all_accessible_symbols
    user_globals = get_all_accessible_symbols(
        func, 
        include_closure=True, 
        include_builtins=True,
        captured_symbols=captured_symbols
    )

    # Determine if we have any suffix (for compiler instance decision)
    has_suffix = compile_suffix is not None or effect_suffix is not None
    compiler = get_compiler(source_file=source_file, user_globals=user_globals, has_suffix=has_suffix)

    func_source = source_code
    registry.register_function_source(source_file, func.__name__, func_source)

    try:
        func_ast = ast.parse(func_source).body[0]
        if not isinstance(func_ast, ast.FunctionDef):
            raise RuntimeError(f"Expected FunctionDef, got {type(func_ast)}")
    except Exception as e:
        raise RuntimeError(f"Failed to parse function {func.__name__}: {e}")

    actual_func_name = func.__name__

    # Determine parameter names early; we need them to inject parametric names
    # into the annotation namespace so that annotations like ``x: T`` can be
    # resolved when ``T`` itself is a compile-time parameter.
    param_names = [arg.arg for arg in func_ast.args.args]

    # Pre-identify parameters annotated directly with ``param`` so that their
    # names can be made available during annotation resolution without shadowing
    # non-parametric parameter names.
    from ..builtin_entities import param as param_type
    raw_parametric_names = set()

    def _annotation_is_param(ann):
        if isinstance(ann, str):
            return ann.strip() == 'param'
        return ann is param_type

    if hasattr(func, '__annotations__') and func.__annotations__:
        for name, ann in func.__annotations__.items():
            if name == 'return':
                continue
            if _annotation_is_param(ann):
                raw_parametric_names.add(name)
    else:
        for arg in func_ast.args.args:
            if arg.annotation is not None and _annotation_is_param(arg.annotation):
                raw_parametric_names.add(arg.arg)
        if func_ast.args.vararg is not None and func_ast.args.vararg.annotation is not None:
            if _annotation_is_param(func_ast.args.vararg.annotation):
                raw_parametric_names.add(func_ast.args.vararg.arg)
        if func_ast.args.kwarg is not None and func_ast.args.kwarg.annotation is not None:
            if _annotation_is_param(func_ast.args.kwarg.annotation):
                raw_parametric_names.add(func_ast.args.kwarg.arg)

    # --- Parametric polymorphism: functions with ``param`` parameters are
    # desugared into a factory.  Resolve their annotations lazily at
    # specialization time, because parameter names like ``T`` are not in scope
    # at decoration time.  This must happen before the yield check so that
    # parametric generator functions are also handled by the factory path.
    if raw_parametric_names:
        # Phase 1 restrictions.
        if func_ast.args.vararg is not None and func_ast.args.vararg.arg in raw_parametric_names:
            logger.error(
                f"parametric *args is not supported in Phase 1: {func_ast.args.vararg.arg}",
                node=func_ast, exc_type=NotImplementedError,
            )
        if func_ast.args.kwarg is not None and func_ast.args.kwarg.arg in raw_parametric_names:
            logger.error(
                f"parametric **kwargs is not supported in Phase 1: {func_ast.args.kwarg.arg}",
                node=func_ast, exc_type=NotImplementedError,
            )

        n_args = len(func_ast.args.args)
        n_defaults = len(func_ast.args.defaults)
        default_start = n_args - n_defaults
        for i, arg in enumerate(func_ast.args.args):
            if arg.arg in raw_parametric_names and i >= default_start:
                logger.error(
                    f"parametric parameter '{arg.arg}' cannot have a default value",
                    node=arg, exc_type=TypeError,
                )

        # Preserve the original parameter order; ``list(raw_parametric_names)``
        # would be unordered because it is a set.
        parametric_param_names = [name for name in param_names if name in raw_parametric_names]

        return _create_parametric_factory_wrapper(
            func=func,
            func_ast=func_ast,
            source_file=source_file,
            source_code=func_source,
            start_line=start_line,
            user_globals=user_globals,
            param_names=param_names,
            parametric_param_names=parametric_param_names,
            fn_attrs=fn_attrs,
            compile_suffix=compile_suffix,
            effect_suffix=effect_suffix,
        )

    # Check if this is a yield-based generator function
    from ..ast_visitor.yield_transform import analyze_yield_function
    yield_analyzer = analyze_yield_function(func_ast)

    if yield_analyzer:
        import copy
        original_func_ast = copy.deepcopy(func_ast)

        from .visible import get_all_accessible_symbols
        transform_globals = get_all_accessible_symbols(
            func,
            include_closure=True,
            include_builtins=True,
            captured_symbols=captured_symbols
        )

        from ..ast_visitor.yield_transform import create_yield_iterator_wrapper
        wrapper = create_yield_iterator_wrapper(
            func, func_ast, yield_analyzer, transform_globals, source_file, registry
        )
        wrapper._original_ast = original_func_ast
        return wrapper

    from ..type_resolver import TypeResolver
    from ..registry import FunctionInfo

    type_resolver = TypeResolver(compiler.module.context, user_globals=user_globals)
    return_type_hint = None
    param_type_hints = {}

    is_dynamic = '.<locals>.' in func.__qualname__

    if hasattr(func, '__annotations__') and func.__annotations__:
        from ..builtin_entities import BuiltinEntity
        from .annotation_resolver import build_annotation_namespace, resolve_annotations_dict
        
        eval_namespace = build_annotation_namespace(user_globals, is_dynamic=is_dynamic)
        resolved_annotations = resolve_annotations_dict(
            func.__annotations__, 
            eval_namespace, 
            type_resolver
        )
        
        for param_name, resolved_type in resolved_annotations.items():
            if param_name == 'return':
                if isinstance(resolved_type, type) and issubclass(resolved_type, BuiltinEntity):
                    if resolved_type.can_be_type():
                        return_type_hint = resolved_type
                elif isinstance(resolved_type, str):
                    pass
                else:
                    return_type_hint = resolved_type
            else:
                if isinstance(resolved_type, type) and issubclass(resolved_type, BuiltinEntity):
                    if resolved_type.can_be_type():
                        param_type_hints[param_name] = resolved_type
                elif not isinstance(resolved_type, str):
                    param_type_hints[param_name] = resolved_type
        if return_type_hint is None:
            from ..builtin_entities.types import void
            return_type_hint = void
    else:
        if func_ast.returns:
            return_type_hint = type_resolver.parse_annotation(func_ast.returns)
        else:
            from ..builtin_entities.types import void
            return_type_hint = void
        for arg in func_ast.args.args:
            if arg.annotation:
                param_type = type_resolver.parse_annotation(arg.annotation)
                if param_type:
                    param_type_hints[arg.arg] = param_type
    from ..ast_visitor.varargs import resolve_varargs, resolve_kwargs
    resolved_varargs = resolve_varargs(func_ast, type_resolver)
    varargs_name = resolved_varargs.param_name
    if resolved_varargs.is_typed and varargs_name in param_type_hints:
        del param_type_hints[varargs_name]

    if resolved_varargs.is_typed:
        # *args: T -> single parameter of type T (symmetric with **kwargs: T)
        param_names.append(varargs_name)
        param_type_hints[varargs_name] = resolved_varargs.parsed_type

    # **kwargs: T -> add a single parameter of type T
    resolved_kwargs = resolve_kwargs(func_ast, type_resolver)
    if resolved_kwargs.is_typed:
        param_names.append(resolved_kwargs.param_name)
        param_type_hints[resolved_kwargs.param_name] = resolved_kwargs.parsed_type

    if _is_parametric_type(return_type_hint):
        logger.error(
            "'param' cannot be used as a function return type",
            node=func_ast.returns, exc_type=TypeError,
        )

    # Build mangled name: name_{compile_suffix}_{effect_suffix}
    # compile_suffix comes first, then effect_suffix
    mangled_name = None
    suffix_parts = []
    if compile_suffix:
        suffix_parts.append(compile_suffix)
    if effect_suffix:
        suffix_parts.append(effect_suffix)
    
    if suffix_parts:
        combined_suffix = '_'.join(suffix_parts)
        mangled_name = func.__name__ + '_' + combined_suffix

    if mangled_name:
        import copy
        func_ast = copy.deepcopy(func_ast)
        func_ast.name = mangled_name
        actual_func_name = mangled_name

    from ..logger import set_source_file
    set_source_file(source_file)
    
    # Determine grouping key and output paths
    # Design: group_key = (source_file, scope, compile_suffix, effect_suffix)
    output_manager = get_output_manager()
    
    # Get scope name (e.g., "GenericType.<locals>" or None for module-level)
    # If effect_scope is provided (not the sentinel), use it directly
    # to avoid detecting pythoc internals as the scope during transitive compilation
    if effect_scope is not _SCOPE_NOT_PROVIDED:
        # effect_scope was explicitly provided (possibly as None for module-level)
        scope_name = effect_scope
    elif compile_suffix or effect_suffix:
        scope_name = get_definition_scope()
        if scope_name == 'module':
            scope_name = None
    else:
        scope_name = None
    
    # Sanitize components for filename
    safe_scope_name = sanitize_filename(scope_name) if scope_name else None
    safe_compile_suffix = sanitize_filename(compile_suffix) if compile_suffix else None
    safe_effect_suffix = sanitize_filename(effect_suffix) if effect_suffix else None
    
    # Build 4-tuple group key
    group_key = (source_file, safe_scope_name, safe_compile_suffix, safe_effect_suffix)
    
    # Build output file paths
    if compile_suffix or effect_suffix:
        cwd = os.getcwd()
        if source_file.startswith(cwd + os.sep) or source_file.startswith(cwd + '/'):
            rel_path = os.path.relpath(source_file, cwd)
        else:
            base_name = os.path.splitext(os.path.basename(source_file))[0]
            rel_path = f"external/{base_name}"
        build_dir = os.path.join('build', os.path.dirname(rel_path))
        base_name = os.path.splitext(os.path.basename(source_file))[0]
        
        # Build filename: base.scope.compile_suffix.effect_suffix.{ll,o,so}
        file_parts = [base_name]
        if safe_scope_name:
            file_parts.append(safe_scope_name)
        if safe_compile_suffix:
            file_parts.append(safe_compile_suffix)
        if safe_effect_suffix:
            file_parts.append(safe_effect_suffix)
        file_base = '.'.join(file_parts)
        
        os.makedirs(build_dir, exist_ok=True)
        ir_file = os.path.join(build_dir, f"{file_base}.ll")
        obj_file = os.path.join(build_dir, f"{file_base}.o")
        from ..utils.link_utils import get_shared_lib_extension
        so_file = os.path.join(build_dir, f"{file_base}{get_shared_lib_extension()}")
    else:
        # No suffix: use standard paths
        build_dir, ir_file, obj_file, so_file = get_build_paths(source_file)
    
    # Note: We do NOT check cache here. Cache is checked at flush time.
    # This allows all @compile decorators to register before any flush happens.
    
    # --- Registration-phase cache: if the same group was already compiled
    # and its object file is still up-to-date, reuse the existing wrapper
    # instead of rebuilding AST / binding_state / compile_callback.
    existing_wrapper = _try_reuse_cached_wrapper(
        output_manager, group_key, func, wrapper,
        compile_suffix, effect_suffix,
    )
    if existing_wrapper is not None:
        return existing_wrapper

    func_info = FunctionInfo(
        name=func.__name__,
        source_file=source_file,
        source_code=func_source,
        return_type_hint=return_type_hint,
        param_type_hints=param_type_hints,
        param_names=param_names,
        overload_enabled=False,
        fn_attrs=fn_attrs or set(),
        has_llvm_varargs=resolved_varargs.has_llvm_varargs,
        has_varargs=resolved_varargs.is_typed,
        has_kwargs=resolved_kwargs.is_typed,
    )

    group_compiler = compiler
    
    group = output_manager.get_or_create_group(
        group_key, group_compiler, ir_file, obj_file, so_file, 
        source_file
    )
    compiler = group['compiler']
    logger.debug(f"@compile {func.__name__}: group_key={group_key}")
    
    from ..effect import capture_effect_context, capture_effect_override_names
    from ..effect import restore_effect_context
    from ..effect import start_effect_tracking, stop_effect_tracking
    from ..effect import push_compilation_context, pop_compilation_context
    _captured_effect_context = capture_effect_context()
    _effect_override_names = (
        set(effect_override_names)
        if effect_override_names is not None
        else capture_effect_override_names()
    )

    binding_state = FunctionBindingState(
        compiler=compiler,
        so_file=so_file,
        source_file=source_file,
        mangled_name=mangled_name,
        original_name=func.__name__,
        actual_func_name=actual_func_name,
        group_key=group_key,
        compile_suffix=compile_suffix,
        effect_suffix=effect_suffix,
        captured_effect_context=_captured_effect_context,
        effect_override_names=_effect_override_names,
        captured_symbols=captured_symbols,
        compilation_globals=dict(user_globals),
        wrapper=wrapper,
    )
    func_info.binding_state = binding_state
    wrapper._func_info = func_info
    wrapper._signature = func_info
    wrapper._binding = binding_state
    wrapper._state = binding_state  # Compatibility alias; `_binding` is canonical.
    output_manager.record_group_planning_deps_from_globals(
        group_key, binding_state.compilation_globals,
    )
    from ..build.deps import get_dependency_tracker
    get_dependency_tracker().record_type_layout_deps_from_globals(
        group_key, binding_state.compilation_globals,
    )

    # Always queue compilation callback - cache check is done at flush time
    # These closure locals are needed because they are AST/source artifacts
    # that don't belong on FunctionBindingState (they are compilation inputs,
    # not per-function state).
    _func_ast = func_ast
    _func_source = func_source
    _param_type_hints = param_type_hints
    _return_type_hint = return_type_hint
    _start_line = start_line

    def compile_callback(comp):
        """Deferred compilation callback.

        Reads phase-1 state from wrapper._binding (the FunctionBindingState),
        not from closure locals that duplicate the same data.
        """
        st = wrapper._binding
        start_effect_tracking()

        if st.effect_suffix:
            push_compilation_context(st.compile_suffix, st.effect_suffix,
                                     st.captured_effect_context, st.group_key,
                                     st.effect_override_names)

        try:
            with restore_effect_context(st.captured_effect_context):
                set_source_context(st.source_file, _start_line - 1)
                comp.compile_function_from_ast(
                    _func_ast,
                    _func_source,
                    reset_module=False,
                    param_type_hints=_param_type_hints,
                    return_type_hint=_return_type_hint,
                    user_globals=st.compilation_globals,
                    group_key=st.group_key,
                    func_state=st,
                    source_start_line=_start_line,
                    original_name=st.original_name,
                )
        finally:
            if st.effect_suffix:
                pop_compilation_context()

        effect_deps = stop_effect_tracking()
        if effect_deps:
            func_info.effect_dependencies = effect_deps
            logger.debug(f"Function {_func_ast.name} uses effects: {effect_deps}")

            from ..build.deps import get_dependency_tracker
            dep_tracker = get_dependency_tracker()
            for effect in effect_deps:
                dep_tracker.record_effect_usage(st.group_key, effect)
    
    from ..effect import is_effect_suffix_suppressed, effect as _effect_singleton
    # Template condition: suppress_effect_suffix() is hiding a real suffix AND
    # this wrapper has no compile_suffix AND no effect_suffix.
    # This means: module-level @compile running inside the import hook's suppress context.
    # Linearize is safe: it always sets compile_suffix via @compile(suffix=X).
    _has_suppressed_suffix = (is_effect_suffix_suppressed()
                              and _effect_singleton._get_current_suffix() is not None)
    _should_be_template = (_has_suppressed_suffix
                           and effect_suffix is None
                           and compile_suffix is None)

    if _should_be_template:
        binding_state.is_template = True
        binding_state.template_compile_callback = compile_callback
        logger.debug(f"@compile {func.__name__}: created as template (suppress active)")
    else:
        output_manager.queue_compilation(group_key, compile_callback, func_info)
        binding_state.is_template = False

    # Backward-compat aliases for external consumers (tests, etc.)
    wrapper._so_file = wrapper._binding.so_file
    wrapper._source_file = wrapper._binding.source_file
    wrapper._compiler = wrapper._binding.compiler
    wrapper._original_name = wrapper._binding.original_name
    wrapper._actual_func_name = wrapper._binding.actual_func_name
    wrapper._mangled_name = wrapper._binding.mangled_name
    wrapper._group_key = wrapper._binding.group_key
    wrapper._compile_suffix = wrapper._binding.compile_suffix
    wrapper._effect_suffix = wrapper._binding.effect_suffix

    output_manager.add_wrapper_to_group(group_key, wrapper)
    
    def get_effect_specialized(target_effect_suffix, effect_overrides):
        """Get or create an effect-specialized version of this wrapper."""
        if wrapper._binding.effect_suffix == target_effect_suffix:
            return wrapper

        return materialize_specialization(wrapper, target_effect_suffix, effect_overrides)

    wrapper.get_effect_specialized = get_effect_specialized

    def handle_call(visitor, func_ref, args, node):
        """Handle calling a @compile function."""
        from ..call_normalization import lower_compile_handle_call
        return lower_compile_handle_call(wrapper, visitor, func_ref, args, node)

    wrapper.handle_call = handle_call
    wrapper._is_compiled = True
    return wrapper


def _clone_binding_to_wrapper(binding, existing, wrapper):
    """Clone binding state from *existing* wrapper onto *new* wrapper."""
    import copy
    new_binding = copy.copy(binding)
    new_binding.wrapper = wrapper

    wrapper._binding = new_binding
    wrapper._state = new_binding
    wrapper._func_info = existing._func_info
    wrapper._signature = existing._signature

    # Backward-compat aliases
    wrapper._so_file = new_binding.so_file
    wrapper._source_file = new_binding.source_file
    wrapper._compiler = new_binding.compiler
    wrapper._original_name = new_binding.original_name
    wrapper._actual_func_name = new_binding.actual_func_name
    wrapper._mangled_name = new_binding.mangled_name
    wrapper._group_key = new_binding.group_key
    wrapper._compile_suffix = new_binding.compile_suffix
    wrapper._effect_suffix = new_binding.effect_suffix
    wrapper._is_compiled = True

    # The compiler lowers calls to this wrapper via handle_call. It must be a
    # fresh closure capturing the new wrapper, not a copy of the existing one.
    def handle_call(visitor, func_ref, args, node):
        from ..call_normalization import lower_compile_handle_call
        return lower_compile_handle_call(wrapper, visitor, func_ref, args, node)

    wrapper.handle_call = handle_call

    def get_effect_specialized(target_effect_suffix, effect_overrides):
        if wrapper._binding.effect_suffix == target_effect_suffix:
            return wrapper
        return materialize_specialization(
            wrapper, target_effect_suffix, effect_overrides)

    wrapper.get_effect_specialized = get_effect_specialized


def _try_reuse_cached_wrapper(
    output_manager, group_key, func, wrapper,
    compile_suffix, effect_suffix,
):
    """Try to reuse an existing wrapper from a previously-registered group.

    Two tiers of caching are checked in order:

    1. On-disk cache: the group's ``.o`` file already exists and is
       up-to-date. The flush-time cache will later reuse the object file;
       here we only avoid rebuilding the registration metadata (AST parse,
       FunctionInfo, FunctionBindingState) when an identical wrapper already
       exists in this process.

    2. In-process cache: no ``.o`` exists yet, but the same group already
       holds wrappers in this process (e.g. a second instantiation of the
       same generic type with the same ``type_suffix``). We clone the
       existing ``FunctionBindingState`` onto the new wrapper, skipping the
       expensive ``getsource -> ast.parse -> binding_state`` work entirely.
       Compilation correctness is not affected because the real lowering
       still happens at ``flush_all`` time when the ``.o`` is eventually
       produced.
    """
    group = output_manager._all_groups.get(group_key)
    if group is None:
        return None

    # Helper – locate an existing wrapper whose binding matches the
    # requested signature.
    def _find_match():
        for existing in group.get('all_wrappers', []):
            binding = getattr(existing, '_binding', getattr(existing, '_state', None))
            if binding is None:
                continue
            if (
                binding.original_name == func.__name__
                and binding.compile_suffix == compile_suffix
                and binding.effect_suffix == effect_suffix
            ):
                return existing, binding
        return None, None

    # Tier 1 – .o cache hit (object file on disk is fresh).
    # We still record type-layout dependencies from the current globals so
    # source_embed edges are not lost across registration-phase reuse.
    if output_manager._group_object_cache_hit(group_key, group):
        existing, binding = _find_match()
        if existing is not None:
            _clone_binding_to_wrapper(binding, existing, wrapper)
            from ..build.deps import get_dependency_tracker
            get_dependency_tracker().record_type_layout_deps_from_globals(
                tuple(group_key), wrapper._binding.compilation_globals,
            )
            output_manager.add_wrapper_to_group(group_key, wrapper)
            return wrapper
        return None

    # Tier 2 – in-process cache: the same group was already registered
    # in this process (no on-disk .o yet, and possibly not flushed).
    existing, binding = _find_match()
    if existing is not None:
        _clone_binding_to_wrapper(binding, existing, wrapper)
        output_manager.add_wrapper_to_group(group_key, wrapper)
        return wrapper

    return None
