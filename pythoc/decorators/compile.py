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


def materialize_specialization(template_wrapper, effect_key, effect_bindings):
    """Materialize a specialization from a template wrapper.

    Single entry point for producing executable code from a template.
    Both default (DEFAULT_EFFECT_KEY) and non-default go through this path.

    Args:
        template_wrapper: A wrapper with _is_template=True
        effect_key: DEFAULT_EFFECT_KEY for default, or an effect suffix string
        effect_bindings: Dict of effect name -> implementation (empty for default)

    Returns:
        Materialized wrapper with compilation queued
    """
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
    else:
        # Non-default: call _compile_impl with effect_suffix
        from ..effect import restore_effect_context
        original_scope = state.group_key[1] if state.group_key else None

        logger.debug(f"Materializing specialization: {state.original_name}_{effect_key}")

        with restore_effect_context(effect_bindings):
            specialized_wrapper = _compile_impl(
                template_wrapper.__wrapped__,
                compile_suffix=state.compile_suffix,
                effect_suffix=effect_key,
                captured_symbols=state.captured_symbols,
                effect_scope=original_scope,
            )

        state.effect_specialized_cache[effect_key] = specialized_wrapper
        return specialized_wrapper


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
        return _compile_dynamic_class(func_or_class, suffix=compile_suffix)

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
    from ..ast_visitor.varargs import resolve_varargs
    resolved_varargs = resolve_varargs(func_ast, type_resolver)
    varargs_kind = resolved_varargs.kind
    varargs_name = resolved_varargs.param_name
    if varargs_kind == 'struct' and varargs_name in param_type_hints:
        del param_type_hints[varargs_name]

    for i, elem_pc_type in enumerate(resolved_varargs.element_types):
        if varargs_kind != 'struct':
            break
        expanded_param_name = f'{varargs_name}_elem{i}'
        param_names.append(expanded_param_name)
        param_type_hints[expanded_param_name] = elem_pc_type

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
    )

    group_compiler = compiler
    
    group = output_manager.get_or_create_group(
        group_key, group_compiler, ir_file, obj_file, so_file, 
        source_file
    )
    compiler = group['compiler']
    logger.debug(f"@compile {func.__name__}: group_key={group_key}")
    
    from ..effect import capture_effect_context, restore_effect_context
    from ..effect import start_effect_tracking, stop_effect_tracking
    from ..effect import push_compilation_context, pop_compilation_context
    _captured_effect_context = capture_effect_context()

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
        captured_symbols=captured_symbols,
        compilation_globals=dict(user_globals),
        wrapper=wrapper,
    )
    func_info.binding_state = binding_state
    wrapper._func_info = func_info
    wrapper._signature = func_info
    wrapper._binding = binding_state
    wrapper._state = binding_state  # Compatibility alias; `_binding` is canonical.

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
                                     st.captured_effect_context, st.group_key)

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
        from ..valueref import wrap_value
        from ..builtin_entities import func as func_type_cls
        from ..builtin_entities.python_type import PythonType
        from ..build.deps import get_dependency_tracker
        
        # Record dependency: caller -> callee (at call time, not from LLVM IR)
        caller_group_key = getattr(visitor, 'current_group_key', None)
        callee_group_key = wrapper._binding.group_key
        if caller_group_key and callee_group_key and caller_group_key != callee_group_key:
            dep_tracker = get_dependency_tracker()
            dep_tracker.record_group_dependency(caller_group_key, callee_group_key, "function_call")
        
        wrapper_ref = wrap_value(wrapper, kind="python", type_hint=PythonType.wrap(wrapper))
        converted_func_ref = visitor.type_converter.convert(wrapper_ref, func_type_cls, node)
        func_type = converted_func_ref.type_hint
        return func_type.handle_call(visitor, converted_func_ref, args, node)

    wrapper.handle_call = handle_call
    wrapper._is_compiled = True
    return wrapper
