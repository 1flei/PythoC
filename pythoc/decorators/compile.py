# -*- coding: utf-8 -*-
from functools import wraps
import inspect
import os
import ast
import sys
from typing import Any, List

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
    get_anonymous_suffix,
    get_function_file_and_source,
    get_function_start_line,
)
from ..build import (
    BuildCache,
    get_output_manager,
    flush_all_pending_outputs,
)
from ..logger import logger, set_source_context


def _get_registry():
    return _unified_registry


def get_compiler(source_file, user_globals, suffix=None):
    registry = _get_registry()
    if suffix:
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



def compile(func_or_class=None, anonymous=False, suffix=None, _effect_caller_module=None, _effect_group_key=None):
    """
    Compile a Python function or class to native code.
    
    Args:
        func_or_class: Function or class to compile
        anonymous: If True, generate unique suffix for this function
        suffix: Explicit suffix for function naming
        _effect_caller_module: Internal parameter for effect override imports.
            When set, the compiled function is grouped with the caller's module
            instead of the original source file. This avoids symbol conflicts
            when the same function is compiled with different effect contexts.
        _effect_group_key: Internal parameter for transitive effect propagation.
            When set, the compiled function is added to this specific group
            (same .so file as the caller). Takes precedence over _effect_caller_module.
    """
    # Normalize suffix early
    suffix = normalize_suffix(suffix)
    
    # If no explicit suffix, check effect context for suffix
    if suffix is None:
        from ..effect import get_current_effect_suffix
        effect_suffix = get_current_effect_suffix()
        if effect_suffix is not None:
            suffix = effect_suffix
    
    if func_or_class is None:
        def decorator(f):
            return compile(f, anonymous=anonymous, suffix=suffix, 
                          _effect_caller_module=_effect_caller_module)
        return decorator

    if inspect.isclass(func_or_class):
        return _compile_dynamic_class(func_or_class, anonymous=anonymous, suffix=suffix)

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
    # Set logger context: line_offset = start_line - 1 because AST lineno starts from 1
    set_source_context(source_file, start_line - 1)

    registry = _get_registry()
    # Use get_all_accessible_symbols to extract ALL accessible symbols
    # This includes: closure variables, enclosing scopes, caller frame locals, and globals
    from .visible import get_all_accessible_symbols
    user_globals = get_all_accessible_symbols(
        func, 
        include_closure=True, 
        include_builtins=True
    )

    compiler = get_compiler(source_file=source_file, user_globals=user_globals, suffix=suffix)

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
        # Save original AST before transformation (for inlining)
        import copy
        original_func_ast = copy.deepcopy(func_ast)
        
        # Get accessible symbols for yield transform
        from .visible import get_all_accessible_symbols
        transform_globals = get_all_accessible_symbols(
            func, 
            include_closure=True, 
            include_builtins=True
        )
        
        # Transform yield function into inline continuation placeholder
        # Yield functions MUST be inlined at call sites - no vtable generation
        from ..ast_visitor.yield_transform import create_yield_iterator_wrapper
        wrapper = create_yield_iterator_wrapper(
            func, func_ast, yield_analyzer, transform_globals, source_file, registry
        )
        # Save original AST for inlining optimization
        wrapper._original_ast = original_func_ast
        return wrapper

    # Track the actual function name (may be specialized for meta-programming or mangled)
    actual_func_name = func.__name__

    compiled_funcs = registry.list_compiled_functions(source_file).get(source_file, [])
    compiled_funcs.append(func.__name__)

    from ..type_resolver import TypeResolver
    from ..registry import FunctionInfo
    
    type_resolver = TypeResolver(compiler.module.context, user_globals=user_globals)
    return_type_hint = None
    param_type_hints = {}

    is_dynamic = '.<locals>.' in func.__qualname__

    if hasattr(func, '__annotations__') and func.__annotations__:
        from ..builtin_entities import BuiltinEntity
        from .annotation_resolver import build_annotation_namespace, resolve_annotations_dict
        
        # Build namespace for resolving string annotations
        eval_namespace = build_annotation_namespace(
            user_globals, 
            is_dynamic=is_dynamic
        )
        
        # Resolve all annotations
        resolved_annotations = resolve_annotations_dict(
            func.__annotations__, 
            eval_namespace, 
            type_resolver
        )
        
        # Extract return type and parameter types
        for param_name, resolved_type in resolved_annotations.items():
            if param_name == 'return':
                if isinstance(resolved_type, type) and issubclass(resolved_type, BuiltinEntity):
                    if resolved_type.can_be_type():
                        return_type_hint = resolved_type
                elif isinstance(resolved_type, str):
                    # Still a string - will be handled later (forward reference)
                    pass
                else:
                    return_type_hint = resolved_type
            else:
                if isinstance(resolved_type, type) and issubclass(resolved_type, BuiltinEntity):
                    if resolved_type.can_be_type():
                        param_type_hints[param_name] = resolved_type
                elif not isinstance(resolved_type, str):
                    # Accept non-BuiltinEntity types (e.g., struct classes)
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

    mangled_name = None
    # Apply suffix or anonymous naming
    # suffix: deterministic naming for deduplication (replaces anonymous in the future)
    # anonymous: auto-generated unique naming (legacy, will be deprecated)
    if suffix:
        # suffix takes priority - use it for deterministic deduplication
        name_suffix = f'_{suffix}'
        if mangled_name:
            mangled_name = mangled_name + name_suffix
        else:
            mangled_name = func.__name__ + name_suffix
    elif anonymous:
        # Auto-generate unique suffix (legacy behavior)
        anonymous_suffix = get_anonymous_suffix()
        if mangled_name:
            mangled_name = mangled_name + anonymous_suffix
        else:
            mangled_name = func.__name__ + anonymous_suffix

    param_names = [arg.arg for arg in func_ast.args.args]
    
    # Detect varargs expansion to include expanded params in FunctionInfo
    from ..ast_visitor.varargs import detect_varargs
    varargs_kind, element_types, varargs_name = detect_varargs(func_ast, type_resolver)
    if varargs_kind == 'struct':
        # For struct varargs, we need to expand the parameter names and types
        # First, remove the varargs parameter itself from param_type_hints (if it was added from __annotations__)
        if varargs_name in param_type_hints:
            del param_type_hints[varargs_name]
        
        # Parse element types if not already parsed
        element_pc_types = []
        if element_types:
            for elem_type in element_types:
                if hasattr(elem_type, 'get_llvm_type'):
                    element_pc_types.append(elem_type)
                else:
                    elem_pc_type = type_resolver.parse_annotation(elem_type)
                    element_pc_types.append(elem_pc_type)
        else:
            # Empty element_types means the varargs annotation is a @compile decorated struct class
            if func_ast.args.vararg and func_ast.args.vararg.annotation:
                annotation = func_ast.args.vararg.annotation
                parsed_type = type_resolver.parse_annotation(annotation)
                if hasattr(parsed_type, '_struct_fields'):
                    for field_name, field_type in parsed_type._struct_fields:
                        element_pc_types.append(field_type)
                elif hasattr(parsed_type, '_field_types'):
                    element_pc_types = parsed_type._field_types
        
        # Add expanded parameter names and types
        for i in range(len(element_pc_types)):
            expanded_param_name = f'{varargs_name}_elem{i}'
            param_names.append(expanded_param_name)
            param_type_hints[expanded_param_name] = element_pc_types[i]
    
    reset_module = len(compiled_funcs) == 1 and not is_dynamic

    if mangled_name:
        import copy
        func_ast = copy.deepcopy(func_ast)
        func_ast.name = mangled_name
        # Update actual_func_name to use the mangled name
        actual_func_name = mangled_name

    # Varargs detection: all varargs types are now handled during IR generation
    # No AST transformation needed - keeps original AST for debugging

    # Set source file for better error messages
    from ..logger import set_source_file
    set_source_file(source_file)
    
    # Determine grouping key and output paths
    output_manager = get_output_manager()
    
    # Transitive effect propagation: use the caller's group key if provided
    # This ensures transitive suffix versions are compiled into the same .so file
    if _effect_group_key is not None:
        group_key = _effect_group_key
        logger.debug(f"Using _effect_group_key: {_effect_group_key}")
        # Get existing group info to reuse paths
        existing_group = output_manager.get_group(_effect_group_key)
        logger.debug(f"existing_group: {existing_group}")
        if existing_group:
            ir_file = existing_group.get('ir_file')
            obj_file = existing_group.get('obj_file')
            so_file = existing_group.get('so_file')
            skip_codegen = existing_group.get('skip_codegen', False)
        else:
            # Fallback: shouldn't happen, but handle gracefully
            build_dir, ir_file, obj_file, so_file = get_build_paths(source_file)
            skip_codegen = BuildCache.check_timestamp_skip(ir_file, obj_file, source_file)
    elif suffix:
        safe_suffix = sanitize_filename(suffix)
        
        # Effect override imports: group by caller module to avoid symbol conflicts
        if _effect_caller_module is not None:
            # Use effect_override as scope to distinguish from regular suffix functions
            scope_name = 'effect_override'
            # Use caller module for grouping - all effect overrides from same caller
            # with same suffix go to the same .so file
            group_key = (_effect_caller_module, scope_name, safe_suffix)
            
            # Build file paths based on caller module
            # Convert module name to path (e.g., 'test.integration.test_foo' -> 'test/integration/test_foo')
            caller_path = _effect_caller_module.replace('.', os.sep)
            build_dir = os.path.join('build', os.path.dirname(caller_path))
            base_name = os.path.basename(caller_path)
            file_base = f"{base_name}.{scope_name}.{safe_suffix}"
        else:
            # Regular suffix functions: group by (definition_file, scope, suffix)
            definition_file = source_file
            scope_name = get_definition_scope()
            group_key = (definition_file, scope_name, safe_suffix)
            
            # Build file paths with scope and suffix
            cwd = os.getcwd()
            if definition_file.startswith(cwd):
                rel_path = os.path.relpath(definition_file, cwd)
            else:
                rel_path = definition_file
            build_dir = os.path.join('build', os.path.dirname(rel_path))
            base_name = os.path.splitext(os.path.basename(definition_file))[0]
            file_base = f"{base_name}.{scope_name}.{safe_suffix}"
        
        # Define output file paths
        os.makedirs(build_dir, exist_ok=True)
        ir_file = os.path.join(build_dir, f"{file_base}.ll")
        obj_file = os.path.join(build_dir, f"{file_base}.o")
        so_file = os.path.join(build_dir, f"{file_base}.so")
    else:
        # No suffix: group by (source_file, None, None)
        group_key = (source_file, None, None)
        
        # Build file paths without suffix
        build_dir, ir_file, obj_file, so_file = get_build_paths(source_file)
    
    # Check timestamp before creating group
    skip_codegen = BuildCache.check_timestamp_skip(ir_file, obj_file, source_file)
    
    # Register function info (now that we have so_file)
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
    registry.register_function(func_info)
    
    group_compiler = compiler
    
    # Get or create group
    group = output_manager.get_or_create_group(
        group_key, group_compiler, ir_file, obj_file, so_file, 
        source_file, skip_codegen
    )
    compiler = group['compiler']
    skip_codegen = group['skip_codegen']
    
    # Capture effect context at decoration time
    # This ensures JIT compilation uses the effect bindings that were active
    # when @compile was applied, not when the function is first called
    from ..effect import capture_effect_context, restore_effect_context
    from ..effect import start_effect_tracking, stop_effect_tracking
    from ..effect import push_compilation_context, pop_compilation_context
    _captured_effect_context = capture_effect_context()
    _current_suffix = suffix  # Capture the suffix for this function
    _caller_module = _effect_caller_module  # Capture caller module for grouping
    _group_key = group_key  # Capture group key for transitive propagation
    
    # Queue compilation callback instead of compiling immediately
    # This enables two-pass compilation for mutual recursion support
    if not skip_codegen:
        # Capture variables for the callback closure
        _func_ast = func_ast
        _func_source = func_source
        _param_type_hints = param_type_hints
        _return_type_hint = return_type_hint
        _user_globals = user_globals
        _is_dynamic = is_dynamic
        _source_file = source_file
        _registry = registry
        _start_line = start_line
        _func_info = func_info
        
        def compile_callback(comp):
            """Deferred compilation callback"""
            # Start tracking effect usage for this function
            start_effect_tracking()
            
            # Push compilation context so handle_call knows the current suffix
            # and effect overrides for transitive propagation
            logger.debug(f"compile_callback: _current_suffix={_current_suffix}, _captured_effect_context={_captured_effect_context}")
            if _current_suffix:
                push_compilation_context(_current_suffix, _captured_effect_context, _caller_module, _group_key)
                logger.debug(f"Pushed compilation context: suffix={_current_suffix}")
            
            try:
                # Restore effect context that was captured at decoration time
                # This ensures effect.xxx resolves to the correct implementation
                with restore_effect_context(_captured_effect_context):
                    # Set source context for accurate error messages during compilation
                    set_source_context(_source_file, _start_line - 1)
                    # Compile the function into group's compiler
                    logger.debug(f"Deferred compile {_func_ast.name}")
                    comp.compile_function_from_ast(
                        _func_ast,
                        _func_source,
                        reset_module=False,  # Never reset since forward declarations exist
                        param_type_hints=_param_type_hints,
                        return_type_hint=_return_type_hint,
                        user_globals=_user_globals,
                    )
            finally:
                # Pop compilation context
                if _current_suffix:
                    pop_compilation_context()
            
            # Stop tracking and record effect dependencies
            effect_deps = stop_effect_tracking()
            if effect_deps:
                _func_info.effect_dependencies = effect_deps
                logger.debug(f"Function {_func_ast.name} uses effects: {effect_deps}")
            
            # After compilation, scan for declared functions and record dependencies
            if not hasattr(comp, 'imported_user_functions'):
                comp.imported_user_functions = {}
            for name, value in comp.module.globals.items():
                if hasattr(value, 'is_declaration') and value.is_declaration:
                    dep_func_info = _registry.get_function_info(name)
                    if not dep_func_info:
                        dep_func_info = _registry.get_function_info_by_mangled(name)
                    if dep_func_info and dep_func_info.source_file and dep_func_info.source_file != _source_file:
                        comp.imported_user_functions[name] = dep_func_info.source_file
        
        # Queue the compilation callback for deferred two-pass compilation
        output_manager.queue_compilation(group_key, compile_callback, func_info)
    
    # Setup wrapper attributes
    wrapper._compiler = compiler
    wrapper._so_file = so_file
    wrapper._source_file = source_file
    wrapper._mangled_name = mangled_name
    wrapper._original_name = func.__name__
    wrapper._actual_func_name = actual_func_name
    wrapper._group_key = group_key
    wrapper._captured_effect_context = _captured_effect_context
    
    # Store wrapper reference in func_info for on-demand suffix generation
    func_info.wrapper = wrapper
    
    # Add wrapper to group
    output_manager.add_wrapper_to_group(group_key, wrapper)
    


    def handle_call(visitor, args, node):
        from llvmlite import ir
        from ..registry import get_unified_registry
        from ..effect import get_current_compilation_context
        
        logger.debug(f"@compile handle_call: func={func.__name__}, wrapper._mangled_name={getattr(wrapper, '_mangled_name', None)}")
        lookup_mangled = getattr(wrapper, '_mangled_name', None)
        func_name = func.__name__
        registry = get_unified_registry()
        func_info_lookup = None
        
        if lookup_mangled:
            # If wrapper carries a specific mangled version, use it directly
            resolved_info = registry.get_function_info_by_mangled(lookup_mangled)
            if not resolved_info:
                raise NameError(f"Function '{func_name}' with mangled '{lookup_mangled}' not found in registry")
            actual_func_name = resolved_info.mangled_name
        else:
            func_info_lookup = registry.get_function_info(func_name)
            if not func_info_lookup:
                raise NameError(f"Function '{func_name}' not found in registry")
            
            # If overloading is enabled and we don't have a specific mangled target, resolve by args
            resolved_info = func_info_lookup
            actual_func_name = func_info_lookup.mangled_name if func_info_lookup.mangled_name else func_name
            if func_info_lookup.overload_enabled:
                # Use pre-evaluated arguments to determine their PC types
                arg_types = []
                for arg in args:
                    if hasattr(arg, 'type_hint') and arg.type_hint:
                        arg_types.append(arg.type_hint)
                    else:
                        raise TypeError("Overloaded call requires PC type hints for arguments; missing type_hint")
                
                mangled = _mangle_function_name(func_name, arg_types)
                resolved_info = registry.get_function_info_by_mangled(mangled)
                if not resolved_info:
                    raise NameError(f"Overloaded function '{func_name}' with signature {[getattr(t,'get_name',lambda:str(t))() for t in arg_types]} not registered")
                actual_func_name = resolved_info.mangled_name if resolved_info.mangled_name else func_name

        # Get or declare the function in the module
        logger.debug(f"Function call: actual_func_name={actual_func_name}, resolved_info.param_names={resolved_info.param_names}")
        try:
            ir_func = visitor.module.get_global(actual_func_name)
        except KeyError:
            param_llvm_types = []
            for param_name in resolved_info.param_names:
                pc_param_type = resolved_info.param_type_hints.get(param_name)
                if pc_param_type and hasattr(pc_param_type, 'get_llvm_type'):
                    # All PC types now accept module_context parameter uniformly
                    param_llvm_types.append(pc_param_type.get_llvm_type(compiler.module.context))
                else:
                    raise TypeError(f"Invalid parameter type hint for '{param_name}' in function '{actual_func_name}'")
            if resolved_info.return_type_hint and hasattr(resolved_info.return_type_hint, 'get_llvm_type'):
                # All PC types now accept module_context parameter uniformly
                return_llvm_type = resolved_info.return_type_hint.get_llvm_type(compiler.module.context)
            else:
                return_llvm_type = ir.VoidType()
            func_type = ir.FunctionType(return_llvm_type, param_llvm_types)
            ir_func = ir.Function(visitor.module, func_type, actual_func_name)
            
            # Automatically track function dependency when declaring it
            # Add to visitor.compiler if available, otherwise to visitor
            target_deps = getattr(visitor, 'compiler', visitor)
            if resolved_info.source_file:
                if not hasattr(target_deps, 'imported_user_functions'):
                    target_deps.imported_user_functions = {}
                if actual_func_name not in target_deps.imported_user_functions:
                    target_deps.imported_user_functions[actual_func_name] = resolved_info.source_file
        
        # Propagate transitive dependencies: if the called function has its own dependencies,
        # add them to the current visitor's compiler dependency list
        if hasattr(compiler, 'imported_user_functions'):
            target_deps = getattr(visitor, 'compiler', visitor)
            if not hasattr(target_deps, 'imported_user_functions'):
                target_deps.imported_user_functions = {}
            for dep_func_name, dep_source_file in compiler.imported_user_functions.items():
                if dep_func_name not in target_deps.imported_user_functions:
                    target_deps.imported_user_functions[dep_func_name] = dep_source_file
        
        # Perform the call using resolved function info
        # Get module context for struct types
        module_context = visitor.module.context if hasattr(visitor, 'module') else None
        
        # Build parameter LLVM types with proper module_context handling
        param_llvm_types = []
        for p in resolved_info.param_names:
            param_type = resolved_info.param_type_hints[p]
            # All PC types now accept module_context parameter uniformly
            param_llvm_types.append(param_type.get_llvm_type(module_context))
        
        logger.debug(f"@compile handle_call: func={actual_func_name}, return_type={resolved_info.return_type_hint}, is_linear={hasattr(resolved_info.return_type_hint, 'is_linear') and resolved_info.return_type_hint.is_linear()}")
        
        result = visitor._perform_call(
            node,
            ir_func,
            param_llvm_types,
            resolved_info.return_type_hint,
            evaluated_args=args,  # Pass pre-evaluated arguments
        )
        
        # If this is a yield-generated function, mark the result for potential inlining
        if hasattr(wrapper, '_is_yield_generated') and wrapper._is_yield_generated:
            result._yield_inline_info = {
                'func_obj': wrapper,
                'original_ast': getattr(wrapper, '_original_ast', None),
                'call_node': node,
                'call_args': args
            }
        
        return result

    def handle_cast(visitor, node):
        """Handle casting @compile function to function pointer type
        
        This is called when a @compile function is used as a value (e.g., returned or assigned).
        It returns a ValueRef containing the function pointer.
        
        Unified protocol:
        - Signature: handle_cast(visitor, node) -> ValueRef
        - visitor: AST visitor instance (provides module, builder, etc.)
        - node: ast.Name node (the reference to the function)
        
        Args:
            visitor: AST visitor instance (needed to access module and declare function)
            node: ast.Name node (the function reference)
        """
        from ..valueref import ValueRef
        from ..registry import get_unified_registry
        from ..effect import get_current_compilation_context
        from llvmlite import ir
        
        # Get function info from registry
        registry = get_unified_registry()
        func_name = func.__name__
        suffix_info = None  # Initialize for later use
        
        # Check if this wrapper has a specific mangled name (for overloaded functions)
        lookup_mangled = getattr(wrapper, '_mangled_name', None)
        if lookup_mangled:
            func_info = registry.get_function_info_by_mangled(lookup_mangled)
            if not func_info:
                raise NameError(f"Function '{func_name}' with mangled '{lookup_mangled}' not found in registry")
            actual_func_name = func_info.mangled_name
        else:
            func_info = registry.get_function_info(func_name)
            if not func_info:
                raise NameError(f"Function '{func_name}' not found in registry")
            actual_func_name = func_info.mangled_name if func_info.mangled_name else func_name
        
        # ========== Transitive Effect Propagation for Function Pointers ==========
        # When a function is used as a value (function pointer), we also need to
        # check if we're in a compilation context with effect overrides.
        # If the function uses an overridden effect, we should return the suffix version.
        compilation_ctx = get_current_compilation_context()
        
        if compilation_ctx and not lookup_mangled:
            ctx_suffix = compilation_ctx.get('suffix')
            ctx_effects = compilation_ctx.get('effect_overrides', {})
            ctx_group_key = compilation_ctx.get('group_key')
            
            if ctx_suffix and ctx_effects:
                # Check if the function uses any overridden effect
                func_effect_deps = func_info.effect_dependencies if func_info else set()
                overridden_effects = set(ctx_effects.keys())
                
                if func_effect_deps & overridden_effects:
                    # The function uses an overridden effect
                    # We need to use/generate a suffix version
                    suffix_mangled_name = f"{func_name}_{ctx_suffix}"
                    
                    # Check if suffix version already exists
                    suffix_info = registry.get_function_info_by_mangled(suffix_mangled_name)
                    
                    if not suffix_info:
                        # Generate suffix version on-demand
                        logger.debug(f"handle_cast: Generating transitive suffix version: {suffix_mangled_name}")
                        original_wrapper = func_info.wrapper if func_info else None
                        
                        if original_wrapper and hasattr(original_wrapper, '__wrapped__'):
                            # Re-compile with the current effect context
                            # Use the caller's group_key to ensure the transitive function
                            # is compiled into the same .so file
                            from ..effect import restore_effect_context
                            with restore_effect_context(ctx_effects):
                                new_wrapper = compile(
                                    original_wrapper.__wrapped__,
                                    suffix=ctx_suffix,
                                    _effect_group_key=ctx_group_key
                                )
                            
                            # Get the newly registered function info
                            suffix_info = registry.get_function_info_by_mangled(suffix_mangled_name)
                    
                    if suffix_info:
                        # Use the suffix version
                        func_info = suffix_info
                        actual_func_name = suffix_mangled_name
                        logger.debug(f"handle_cast: Using transitive suffix version: {actual_func_name}")
        
        # Try to get function from module, or declare it
        try:
            ir_func = visitor.module.get_global(actual_func_name)
        except KeyError:
            # Declare the function
            param_llvm_types = []
            for param_name in func_info.param_names:
                param_type = func_info.param_type_hints.get(param_name)
                if param_type and hasattr(param_type, 'get_llvm_type'):
                    # All PC types now accept module_context parameter uniformly
                    param_llvm_types.append(param_type.get_llvm_type(wrapper._compiler.module.context))
                else:
                    param_llvm_types.append(ir.IntType(32))
            
            if func_info.return_type_hint and hasattr(func_info.return_type_hint, 'get_llvm_type'):
                # All PC types now accept module_context parameter uniformly
                return_llvm_type = func_info.return_type_hint.get_llvm_type(wrapper._compiler.module.context)
            else:
                return_llvm_type = ir.IntType(32)
            
            func_type = ir.FunctionType(return_llvm_type, param_llvm_types)
            ir_func = ir.Function(visitor.module, func_type, actual_func_name)
        
        # Build func type hint
        from ..builtin_entities import func as func_type
        from ..valueref import wrap_value
        param_types = [func_info.param_type_hints[p] for p in func_info.param_names]
        if param_types:
            func_type_hint = func_type[param_types, func_info.return_type_hint]
        else:
            func_type_hint = func_type[[], func_info.return_type_hint]
        
        # Return ValueRef with actual IR function pointer
        # Return ValueRef with actual IR function pointer
        # Note: Transitive effect propagation is already handled above by selecting
        # the correct suffix version. The returned ir_func is the correct function.
        result = wrap_value(
            ir_func,
            kind='pointer',
            type_hint=func_type_hint,
        )
        return result
    
    wrapper.handle_call = handle_call
    wrapper.handle_cast = handle_cast
    wrapper._is_compiled = True
    return wrapper
