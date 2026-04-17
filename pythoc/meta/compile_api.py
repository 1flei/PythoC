"""
Compilation APIs for meta-generated code.

This module provides compile_ast(), compile_generated(), and compile_artifact()
which reuse the existing @compile lifecycle (wrapper creation, group key,
output manager, native loading) but start from AST/GeneratedFunction instead
of a decorated Python function.
"""

import ast
import copy
import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Set, Union

from ..compiler import LLVMCompiler
from ..context import FunctionBindingState
from ..registry import FunctionInfo, _unified_registry
from ..build import get_output_manager
from ..decorators.compile import get_compiler
from ..utils import (
    normalize_suffix,
    sanitize_filename,
    get_build_paths,
)
from ..logger import logger, set_source_context, set_source_file

from .generated import GeneratedFunction, MetaArtifact


def compile_ast(
    fn_ast,
    *,
    param_types=None,
    return_type=None,
    name=None,
    suffix=None,
    attrs=None,
    source_file=None,
    source_code=None,
    start_line=1,
    user_globals=None,
    debug_source=None,
    effect_suffix=None,
    effect_scope=None,
    copy_ast=True,
):
    """Compile an ast.FunctionDef through the standard @compile lifecycle.

    This is a "headless @compile" -- it constructs the same wrapper, group key,
    FunctionInfo, compile_callback, and FunctionBindingState as the @compile
    decorator, but starts from a pre-built AST node instead of a decorated
    Python function.

    Args:
        fn_ast: An ast.FunctionDef node to compile.
        param_types: Dict mapping parameter names to pythoc type objects.
        return_type: Return type hint (a pythoc type object).
        name: Override the function name (applied to AST before compilation).
        suffix: Explicit compile_suffix for specialization.
        attrs: Set of LLVM function-level attributes.
        source_file: Source file for grouping/diagnostics. If None, uses
            the caller's file.
        source_code: Source code string for diagnostics.
        start_line: Start line number for diagnostics.
        user_globals: Dict of symbols needed during compilation.
        debug_source: Optional debug source text.
        effect_suffix: Effect suffix for effect specialization.
        effect_scope: Effect scope override.

    Returns:
        A compiled wrapper function (same shape as @compile produces).
    """
    if not isinstance(fn_ast, ast.FunctionDef):
        raise TypeError(
            "compile_ast expects ast.FunctionDef, got {}".format(type(fn_ast).__name__)
        )

    # Deep-copy the AST unless the caller guarantees ownership of a fresh node.
    if copy_ast:
        fn_ast = copy.deepcopy(fn_ast)

    # Apply name override
    if name is not None:
        fn_ast.name = name

    func_name = fn_ast.name
    param_type_hints = dict(param_types) if param_types else {}
    return_type_hint = return_type
    if return_type_hint is None:
        from ..builtin_entities.types import void
        return_type_hint = void

    user_globals = dict(user_globals) if user_globals else {}
    fn_attrs = set(attrs) if attrs else set()

    # Synthesize source_file from caller's frame if not provided
    if source_file is None:
        frame = inspect.currentframe()
        try:
            caller = frame.f_back
            source_file = caller.f_code.co_filename if caller else "<meta>"
        finally:
            del frame

    # Synthesize source_code from AST if not provided
    if source_code is None:
        try:
            source_code = ast.unparse(fn_ast)
        except Exception:
            source_code = "# meta-generated: {}".format(func_name)

    # Normalize suffixes
    compile_suffix = normalize_suffix(suffix)
    effect_suffix = normalize_suffix(effect_suffix) if effect_suffix else None

    # Build mangled name
    mangled_name = None
    actual_func_name = func_name
    suffix_parts = []
    if compile_suffix:
        suffix_parts.append(compile_suffix)
    if effect_suffix:
        suffix_parts.append(effect_suffix)
    if suffix_parts:
        combined = '_'.join(suffix_parts)
        mangled_name = func_name + '_' + combined
        fn_ast.name = mangled_name
        actual_func_name = mangled_name

    # Get compiler instance
    has_suffix = compile_suffix is not None or effect_suffix is not None
    compiler = get_compiler(
        source_file=source_file,
        user_globals=user_globals,
        has_suffix=has_suffix,
    )

    # Register source
    registry = _unified_registry
    registry.register_function_source(source_file, func_name, source_code)

    # Param names from AST
    param_names = [arg.arg for arg in fn_ast.args.args]

    set_source_file(source_file)
    set_source_context(source_file, start_line - 1)

    # Build group key
    output_manager = get_output_manager()

    scope_name = None
    if effect_scope is not None:
        scope_name = effect_scope
    elif compile_suffix or effect_suffix:
        scope_name = "meta"

    safe_scope = sanitize_filename(scope_name) if scope_name else None
    safe_csuffix = sanitize_filename(compile_suffix) if compile_suffix else None
    safe_esuffix = sanitize_filename(effect_suffix) if effect_suffix else None

    group_key = (source_file, safe_scope, safe_csuffix, safe_esuffix)

    # Build output file paths
    if compile_suffix or effect_suffix:
        cwd = os.getcwd()
        if source_file.startswith(cwd + os.sep) or source_file.startswith(cwd + '/'):
            rel_path = os.path.relpath(source_file, cwd)
        else:
            base = os.path.splitext(os.path.basename(source_file))[0]
            rel_path = "external/{}".format(base)
        build_dir = os.path.join('build', os.path.dirname(rel_path))
        base = os.path.splitext(os.path.basename(source_file))[0]

        file_parts = [base]
        if safe_scope:
            file_parts.append(safe_scope)
        if safe_csuffix:
            file_parts.append(safe_csuffix)
        if safe_esuffix:
            file_parts.append(safe_esuffix)
        file_base = '.'.join(file_parts)

        os.makedirs(build_dir, exist_ok=True)
        ir_file = os.path.join(build_dir, "{}.ll".format(file_base))
        obj_file = os.path.join(build_dir, "{}.o".format(file_base))
        from ..utils.link_utils import get_shared_lib_extension
        so_file = os.path.join(
            build_dir, "{}{}".format(file_base, get_shared_lib_extension())
        )
    else:
        _, ir_file, obj_file, so_file = get_build_paths(source_file)

    # Create FunctionInfo
    func_info = FunctionInfo(
        name=func_name,
        source_file=source_file,
        source_code=source_code,
        return_type_hint=return_type_hint,
        param_type_hints=param_type_hints,
        param_names=param_names,
        overload_enabled=False,
        fn_attrs=fn_attrs,
    )

    # Create wrapper
    from ..native_executor import get_multi_so_executor
    executor = get_multi_so_executor()

    def wrapper(*args, **kwargs):
        if not hasattr(wrapper, '_native_func'):
            wrapper._native_func = executor.execute_function(wrapper)
        return wrapper._native_func(*args)

    # Get or create group
    group = output_manager.get_or_create_group(
        group_key, compiler, ir_file, obj_file, so_file, source_file
    )
    compiler = group['compiler']
    logger.debug("meta.compile_ast {}: group_key={}".format(func_name, group_key))

    # Capture effect context
    from ..effect import capture_effect_context, restore_effect_context
    from ..effect import start_effect_tracking, stop_effect_tracking
    from ..effect import push_compilation_context, pop_compilation_context
    captured_effect_ctx = capture_effect_context()

    binding_state = FunctionBindingState(
        compiler=compiler,
        so_file=so_file,
        source_file=source_file,
        mangled_name=mangled_name,
        original_name=func_name,
        actual_func_name=actual_func_name,
        group_key=group_key,
        compile_suffix=compile_suffix,
        effect_suffix=effect_suffix,
        captured_effect_context=captured_effect_ctx,
        compilation_globals=dict(user_globals),
        wrapper=wrapper,
    )
    func_info.binding_state = binding_state
    wrapper._func_info = func_info
    wrapper._signature = func_info
    wrapper._binding = binding_state
    wrapper._state = binding_state  # Compatibility alias; `_binding` is canonical.

    # Closure locals for compile_callback
    _fn_ast = fn_ast
    _source_code = source_code
    _param_type_hints = param_type_hints
    _return_type_hint = return_type_hint
    _start_line = start_line

    def compile_callback(comp):
        st = wrapper._binding
        start_effect_tracking()

        if st.effect_suffix:
            push_compilation_context(
                st.compile_suffix, st.effect_suffix,
                st.captured_effect_context, st.group_key,
            )

        try:
            with restore_effect_context(st.captured_effect_context):
                set_source_context(st.source_file, _start_line - 1)
                comp.compile_function_from_ast(
                    _fn_ast,
                    _source_code,
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
            logger.debug(
                "Function {} uses effects: {}".format(_fn_ast.name, effect_deps)
            )
            from ..build.deps import get_dependency_tracker
            dep_tracker = get_dependency_tracker()
            for eff in effect_deps:
                dep_tracker.record_effect_usage(st.group_key, eff)

    # Queue compilation and set state
    output_manager.queue_compilation(group_key, compile_callback, func_info)
    binding_state.is_template = False

    # Backward-compat aliases
    wrapper._so_file = binding_state.so_file
    wrapper._source_file = binding_state.source_file
    wrapper._compiler = binding_state.compiler
    wrapper._original_name = binding_state.original_name
    wrapper._actual_func_name = binding_state.actual_func_name
    wrapper._mangled_name = binding_state.mangled_name
    wrapper._group_key = binding_state.group_key
    wrapper._compile_suffix = binding_state.compile_suffix
    wrapper._effect_suffix = binding_state.effect_suffix

    output_manager.add_wrapper_to_group(group_key, wrapper)

    # Set up handle_call for cross-module calls
    def handle_call(visitor, func_ref, args, node):
        from ..call_normalization import lower_compile_handle_call
        return lower_compile_handle_call(wrapper, visitor, func_ref, args, node)

    wrapper.handle_call = handle_call
    wrapper._is_compiled = True

    return wrapper


def compile_generated(
    fn,
    *,
    user_globals=None,
    suffix=None,
    source_file=None,
    group_key=None,
):
    """Compile a GeneratedFunction through the standard lifecycle.

    This is a thin wrapper over compile_ast that extracts the AST and
    type hints from a GeneratedFunction.

    Args:
        fn: A GeneratedFunction instance.
        user_globals: Symbols needed during compilation.
        suffix: Compile suffix for specialization.
        source_file: Source file for grouping. If None, uses caller's file.
        group_key: Explicit group key (for artifact compilation).

    Returns:
        A compiled wrapper function.
    """
    if not isinstance(fn, GeneratedFunction):
        raise TypeError(
            "compile_generated expects GeneratedFunction, got {}".format(
                type(fn).__name__
            )
        )

    globals_to_use = dict(user_globals) if user_globals else {}
    if fn.required_globals:
        globals_to_use.update(fn.required_globals)

    # Synthesize source_file from caller if not provided
    if source_file is None:
        if fn.source_file:
            source_file = fn.source_file
        else:
            frame = inspect.currentframe()
            try:
                caller = frame.f_back
                source_file = caller.f_code.co_filename if caller else "<meta>"
            finally:
                del frame

    func_def = fn.to_func_def()
    return compile_ast(
        func_def,
        param_types=fn.get_param_type_hints(),
        return_type=fn.get_return_type_hint(),
        user_globals=globals_to_use,
        suffix=suffix,
        attrs=fn.attrs,
        source_file=source_file,
        source_code=fn.debug_source,
        start_line=fn.start_line,
        copy_ast=False,
    )


def compile_artifact(
    art,
    *,
    user_globals=None,
    suffix=None,
    source_file=None,
):
    """Compile all functions in a MetaArtifact.

    The primary function is compiled first, then helpers. All share the
    same compilation group. Each compiled wrapper is progressively added
    to user_globals so later functions can reference earlier ones.

    Args:
        art: A MetaArtifact instance.
        user_globals: Symbols needed during compilation.
        suffix: Compile suffix for specialization.
        source_file: Source file for grouping.

    Returns:
        The compiled wrapper for the primary function.
    """
    if not isinstance(art, MetaArtifact):
        raise TypeError(
            "compile_artifact expects MetaArtifact, got {}".format(
                type(art).__name__
            )
        )

    globals_to_use = dict(user_globals) if user_globals else {}
    if art.required_globals:
        globals_to_use.update(art.required_globals)

    # Synthesize source_file from caller if not provided
    if source_file is None:
        frame = inspect.currentframe()
        try:
            caller = frame.f_back
            source_file = caller.f_code.co_filename if caller else "<meta>"
        finally:
            del frame

    results = []

    # Collect all functions to compile: primary + helpers
    all_funcs = []
    primary = art.primary
    if isinstance(primary, GeneratedFunction):
        all_funcs.append(primary)
    elif isinstance(primary, ast.FunctionDef):
        all_funcs.append(primary)

    for helper in art.helpers:
        all_funcs.append(helper)

    for fn in all_funcs:
        if isinstance(fn, GeneratedFunction):
            w = compile_generated(
                fn,
                user_globals=globals_to_use,
                suffix=suffix,
                source_file=source_file,
            )
            globals_to_use[fn.name] = w
        elif isinstance(fn, ast.FunctionDef):
            w = compile_ast(
                fn,
                user_globals=globals_to_use,
                suffix=suffix,
                source_file=source_file,
            )
            globals_to_use[fn.name] = w
        else:
            raise TypeError(
                "Unsupported artifact member type: {}".format(type(fn).__name__)
            )
        results.append(w)

    # Return primary wrapper (first in results)
    return results[0] if results else None
