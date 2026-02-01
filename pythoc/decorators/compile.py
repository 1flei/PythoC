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


def compile(func_or_class=None, suffix=None, _effect_suffix=None, _effect_scope=_SCOPE_NOT_PROVIDED):
    """
    Compile a Python function or class to native code.
    
    Args:
        func_or_class: Function or class to compile
        suffix: Explicit compile_suffix for function naming (from @compile(suffix=T))
                This is NOT contagious - each function chooses its own compile_suffix.
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
                                effect_scope=_effect_scope)
        return decorator

    return _compile_impl(func_or_class, 
                        compile_suffix=compile_suffix,
                        effect_suffix=effect_suffix,
                        captured_symbols=captured_symbols,
                        effect_scope=_effect_scope)


def _compile_impl(func_or_class, 
                  compile_suffix: Optional[str] = None, 
                  effect_suffix: Optional[str] = None,
                  captured_symbols=None,
                  effect_scope=_SCOPE_NOT_PROVIDED):
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
    """
    if inspect.isclass(func_or_class):
        return _compile_dynamic_class(func_or_class, suffix=compile_suffix)

    func = func_or_class

    from ..native_executor import get_multi_so_executor
    executor = get_multi_so_executor()    

    @wraps(func)
    def wrapper(*args, **kwargs):
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

    param_names = [arg.arg for arg in func_ast.args.args]
    
    # Detect varargs expansion
    from ..ast_visitor.varargs import detect_varargs
    varargs_kind, element_types, varargs_name = detect_varargs(func_ast, type_resolver)
    if varargs_kind == 'struct':
        if varargs_name in param_type_hints:
            del param_type_hints[varargs_name]
        
        element_pc_types = []
        if element_types:
            for elem_type in element_types:
                if hasattr(elem_type, 'get_llvm_type'):
                    element_pc_types.append(elem_type)
                else:
                    elem_pc_type = type_resolver.parse_annotation(elem_type)
                    element_pc_types.append(elem_pc_type)
        else:
            if func_ast.args.vararg and func_ast.args.vararg.annotation:
                annotation = func_ast.args.vararg.annotation
                parsed_type = type_resolver.parse_annotation(annotation)
                if hasattr(parsed_type, '_struct_fields'):
                    for field_name, field_type in parsed_type._struct_fields:
                        element_pc_types.append(field_type)
                elif hasattr(parsed_type, '_field_types'):
                    element_pc_types = parsed_type._field_types
        
        for i in range(len(element_pc_types)):
            expanded_param_name = f'{varargs_name}_elem{i}'
            param_names.append(expanded_param_name)
            param_type_hints[expanded_param_name] = element_pc_types[i]

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
    
    # Sanitize suffixes for filename
    safe_compile_suffix = sanitize_filename(compile_suffix) if compile_suffix else None
    safe_effect_suffix = sanitize_filename(effect_suffix) if effect_suffix else None
    
    # Build 4-tuple group key
    group_key = (source_file, scope_name, safe_compile_suffix, safe_effect_suffix)
    
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
        if scope_name:
            file_parts.append(scope_name)
        if safe_compile_suffix:
            file_parts.append(safe_compile_suffix)
        if safe_effect_suffix:
            file_parts.append(safe_effect_suffix)
        file_base = '.'.join(file_parts)
        
        os.makedirs(build_dir, exist_ok=True)
        ir_file = os.path.join(build_dir, f"{file_base}.ll")
        obj_file = os.path.join(build_dir, f"{file_base}.o")
        so_file = os.path.join(build_dir, f"{file_base}.so")
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
        mangled_name=mangled_name,
        overload_enabled=False,
        so_file=so_file,
    )
    
    # Associate wrapper with func_info early (before registration)
    # This ensures mutual recursion can find the wrapper via func_info
    func_info.wrapper = wrapper
    wrapper._func_info = func_info

    # Store user_globals as a mutable dict on func_info
    # This will be augmented with group scope at flush time for recursion support
    func_info.compilation_globals = dict(user_globals)

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
    _compile_suffix = compile_suffix
    _effect_suffix = effect_suffix
    _group_key = group_key
    
    # Always queue compilation callback - cache check is done at flush time
    _func_ast = func_ast
    _func_source = func_source
    _param_type_hints = param_type_hints
    _return_type_hint = return_type_hint
    _source_file = source_file
    _registry = registry
    _start_line = start_line
    _func_info = func_info
    _mangled_name = mangled_name
    _obj_file = obj_file
    
    def compile_callback(comp):
        """Deferred compilation callback"""
        start_effect_tracking()
        
        if _effect_suffix:
            push_compilation_context(_compile_suffix, _effect_suffix, _captured_effect_context, _group_key)
        
        try:
            with restore_effect_context(_captured_effect_context):
                set_source_context(_source_file, _start_line - 1)
                # Use func_info.compilation_globals which has been augmented
                # with group scope at flush time for recursion support
                comp.compile_function_from_ast(
                    _func_ast,
                    _func_source,
                    reset_module=False,
                    param_type_hints=_param_type_hints,
                    return_type_hint=_return_type_hint,
                    user_globals=_func_info.compilation_globals,
                    group_key=_group_key,
                )
        finally:
            if _effect_suffix:
                pop_compilation_context()
        
        effect_deps = stop_effect_tracking()
        if effect_deps:
            _func_info.effect_dependencies = effect_deps
            logger.debug(f"Function {_func_ast.name} uses effects: {effect_deps}")
            
            # Record effect dependencies at group level immediately
            from ..build.deps import get_dependency_tracker
            dep_tracker = get_dependency_tracker()
            for effect in effect_deps:
                dep_tracker.record_effect_usage(_group_key, effect)
    
    output_manager.queue_compilation(group_key, compile_callback, func_info)
    
    wrapper._compiler = compiler
    wrapper._so_file = so_file
    wrapper._source_file = source_file
    wrapper._mangled_name = mangled_name
    wrapper._original_name = func.__name__
    wrapper._actual_func_name = actual_func_name
    wrapper._group_key = group_key
    wrapper._captured_effect_context = _captured_effect_context
    wrapper._captured_symbols = captured_symbols
    wrapper._compile_suffix = compile_suffix
    wrapper._effect_suffix = effect_suffix
    wrapper._effect_specialized_cache = {}  # Cache for effect-specialized versions
    # Note: wrapper._func_info and func_info.wrapper are set earlier (before registration)
    
    output_manager.add_wrapper_to_group(group_key, wrapper)
    
    def get_effect_specialized(target_effect_suffix, effect_overrides):
        """
        Get or create an effect-specialized version of this wrapper.
        
        Args:
            target_effect_suffix: Effect suffix string (e.g., "MyEffect")
            effect_overrides: Dict of effect overrides from context
        
        Returns:
            Specialized wrapper with the given effect_suffix
        """
        # If this wrapper already has the target effect_suffix, return self
        if wrapper._effect_suffix == target_effect_suffix:
            return wrapper
        
        # Check cache
        if target_effect_suffix in wrapper._effect_specialized_cache:
            return wrapper._effect_specialized_cache[target_effect_suffix]
        
        # Create new specialized version
        from ..effect import restore_effect_context
        
        original_scope = wrapper._group_key[1] if wrapper._group_key else None
        
        logger.debug(f"Creating effect-specialized version: {wrapper._original_name}_{target_effect_suffix}")
        
        with restore_effect_context(effect_overrides):
            # IMPORTANT: reuse captured symbols from the original wrapper.
            # This is required for metaprogrammed functions (e.g., linear_wrap)
            # whose annotations reference locals (not in closure).
            specialized_wrapper = _compile_impl(
                wrapper.__wrapped__,
                compile_suffix=wrapper._compile_suffix,
                effect_suffix=target_effect_suffix,
                captured_symbols=getattr(wrapper, '_captured_symbols', None),
                effect_scope=original_scope
            )
        
        # Cache the specialized wrapper
        wrapper._effect_specialized_cache[target_effect_suffix] = specialized_wrapper
        return specialized_wrapper
    
    wrapper.get_effect_specialized = get_effect_specialized

    def handle_call(visitor, func_ref, args, node):
        """Handle calling a @compile function."""
        from ..valueref import wrap_value
        from ..builtin_entities import func as func_type_cls
        from ..builtin_entities.python_type import PythonType
        from ..build.deps import get_dependency_tracker
        
        # Record dependency: caller -> callee (at call time, not from LLVM IR)
        caller_group_key = getattr(visitor, 'current_group_key', None)
        callee_group_key = wrapper._group_key
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
