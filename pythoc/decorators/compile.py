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

    actual_func_name = func.__name__

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

    param_names = [arg.arg for arg in func_ast.args.args]
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
