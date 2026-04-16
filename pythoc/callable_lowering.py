"""
Callable Lowering - resolve @compile wrappers to LLVM IR function references.

This module owns the "Python wrapper object -> IR symbol" resolution path:
1. Let the effect system decide if the wrapper should be swapped
2. Read wrapper metadata (_binding, _func_info)
3. Template materialization
4. Group dependency recording
5. LLVM function declaration in the target module

This is NOT type conversion (value -> value). It is symbol introduction:
a high-level Python wrapper is lowered to a concrete IR declaration.
This module has no knowledge of effect internals.
"""

from typing import Optional, TYPE_CHECKING
from llvmlite import ir

from .valueref import ValueRef, wrap_value
from .logger import logger

if TYPE_CHECKING:
    from .registry import FunctionInfo
    from .context import FunctionBindingState


def _get_binding(wrapper):
    """Extract FunctionBindingState from a wrapper."""
    return getattr(wrapper, '_binding', getattr(wrapper, '_state', None))


def lower_compile_wrapper(
    wrapper,
    module: ir.Module,
    caller_group_key: Optional[tuple] = None,
    node=None,
) -> ValueRef:
    """Lower a @compile wrapper to an LLVM function pointer ValueRef.

    Performs the full resolution chain:
    1. Effect resolution (may swap wrapper for a specialized version)
    2. Read wrapper metadata
    3. Template materialization if needed
    4. Record group dependency (caller -> callee)
    5. Declare or find the function in the target LLVM module

    Args:
        wrapper: The @compile-decorated wrapper object
        module: Target LLVM module for the declaration
        caller_group_key: Caller's group key for dependency recording
        node: AST node for error reporting

    Returns:
        ValueRef(kind='pointer', type_hint=func[param_types..., return_type])
    """
    # --- Step 1: Let effect system resolve transitive propagation ---
    from .effect import resolve_effect_wrapper
    wrapper = resolve_effect_wrapper(wrapper, caller_group_key)

    # --- Step 2: Read wrapper metadata ---
    binding_state = _get_binding(wrapper)
    if binding_state is None:
        logger.error(
            "Function '{}' missing binding metadata".format(
                getattr(wrapper, '__name__', wrapper)
            ),
            node=node, exc_type=NameError,
        )

    func_info = getattr(wrapper, '_func_info', None)
    if not func_info:
        logger.error(
            f"Function '{binding_state.original_name}' missing _func_info attribute",
            node=node, exc_type=NameError,
        )

    actual_func_name = (
        func_info.mangled_name if func_info.mangled_name
        else binding_state.original_name
    )

    # --- Step 3: Template materialization ---
    _ensure_materialized(func_info)

    # --- Step 4: Record group dependency ---
    _record_dependency(func_info, binding_state, caller_group_key)

    # --- Step 5: Declare or find function in module ---
    ir_func = _declare_or_get(func_info, actual_func_name, module, node)

    return wrap_value(ir_func, kind='pointer', type_hint=func_info.callable_pc_type)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_materialized(func_info):
    """If the callee is still a template, materialize the default version."""
    callee_w = func_info.wrapper if func_info else None
    callee_binding = _get_binding(callee_w) if callee_w else None
    if callee_binding and callee_binding.is_template:
        from .decorators.compile import materialize_specialization, DEFAULT_EFFECT_KEY
        materialize_specialization(callee_w, DEFAULT_EFFECT_KEY, {})


def _record_dependency(func_info, binding_state, caller_group_key):
    """Record group dependency (caller -> callee).

    Always recorded, even for effect-implementation callables.  This is
    important on Windows where DLL link-time needs the dependency graph.
    """
    callee_wrapper = getattr(func_info, 'wrapper', None) if func_info else None
    callee_binding = _get_binding(callee_wrapper) if callee_wrapper else None
    callee_group_key = (
        callee_binding.group_key if callee_binding else None
    ) or (binding_state.group_key if binding_state else None)

    if (caller_group_key and callee_group_key
            and caller_group_key != callee_group_key):
        from .build.deps import get_dependency_tracker
        get_dependency_tracker().record_group_dependency(
            caller_group_key, callee_group_key, "function_ref"
        )


def _declare_or_get(func_info, actual_func_name, module, node=None):
    """Get existing or declare new ir.Function in the module."""
    module_context = module.context

    try:
        return module.get_global(actual_func_name)
    except KeyError:
        pass

    # Build LLVM parameter types from PC type hints
    param_llvm_types = []
    for param_name in func_info.param_names:
        param_type = func_info.param_type_hints.get(param_name)
        if param_type is None or not hasattr(param_type, 'get_llvm_type'):
            logger.error(
                f"Parameter '{param_name}' of function '{actual_func_name}' "
                f"has no valid PC type (got {param_type!r})",
                node=node, exc_type=TypeError,
            )
        param_llvm_types.append(param_type.get_llvm_type(module_context))

    return_type = func_info.return_type_hint
    if return_type is None or not hasattr(return_type, 'get_llvm_type'):
        logger.error(
            f"Return type of function '{actual_func_name}' "
            f"has no valid PC type (got {return_type!r})",
            node=node, exc_type=TypeError,
        )
    return_llvm_type = return_type.get_llvm_type(module_context)

    # Declare with C ABI via LLVMBuilder
    from .builder import LLVMBuilder
    temp_builder = LLVMBuilder()
    func_wrapper = temp_builder.declare_function(
        module, actual_func_name,
        param_llvm_types, return_llvm_type
    )
    ir_func = func_wrapper.ir_function

    # Apply function-level attributes for cross-module optimization
    if func_info.fn_attrs:
        for attr in func_info.fn_attrs:
            ir_func.attributes.add(attr)

    return ir_func


# ---------------------------------------------------------------------------
# Extern wrapper lowering
# ---------------------------------------------------------------------------

def lower_extern_wrapper(
    extern_wrapper,
    module: ir.Module,
    caller_group_key: Optional[tuple] = None,
    node=None,
) -> ValueRef:
    """Lower an @extern wrapper to an LLVM function pointer ValueRef.

    This is the extern counterpart of lower_compile_wrapper.  It declares the
    extern symbol in the current LLVM module (with var_arg flag if applicable)
    and returns a ValueRef typed as func[fixed_params..., return_type].

    After this, the value can be called through func.handle_call just like
    any @compile function pointer.

    Args:
        extern_wrapper: The ExternFunctionWrapper object.
        module: Target LLVM module.
        caller_group_key: Caller's group key for dependency recording.
        node: AST node for error reporting.

    Returns:
        ValueRef(kind='pointer', type_hint=func[param_types..., return_type])
    """
    func_name = extern_wrapper.func_name

    # --- Try to reuse existing declaration ---
    try:
        existing = module.get_global(func_name)
        if existing.is_declaration:
            ir_func = existing
        else:
            logger.error(
                f"Cannot call extern function '{func_name}': "
                f"a local function with the same name is already defined.",
                node=node, exc_type=NameError,
            )
            return None  # unreachable
    except KeyError:
        # --- Declare the extern function ---
        module_context = module.context

        param_llvm_types = []
        for param_name, param_type in extern_wrapper.param_types:
            if param_name == 'args':
                continue
            if param_type is None:
                logger.error(
                    f"Extern function '{func_name}': parameter '{param_name}' "
                    f"has no type annotation",
                    node=node, exc_type=TypeError,
                )
            param_llvm_types.append(param_type.get_llvm_type(module_context))

        if extern_wrapper.return_type is None:
            return_llvm_type = ir.VoidType()
        else:
            return_llvm_type = extern_wrapper.return_type.get_llvm_type(module_context)

        has_varargs = any(
            name == 'args' for name, _ in extern_wrapper.param_types
        )

        from .builder import LLVMBuilder
        func_wrapper = LLVMBuilder().declare_function(
            module, func_name,
            param_llvm_types, return_llvm_type,
            var_arg=has_varargs,
        )
        ir_func = func_wrapper.ir_function

        if extern_wrapper.calling_convention == 'stdcall':
            ir_func.calling_convention = 'x86_stdcallcc'
        else:
            ir_func.calling_convention = 'ccc'

    # --- Record link library dependency ---
    lib = extern_wrapper.lib
    if lib:
        # Global registry — used by the linker to determine -l flags
        from .registry import get_unified_registry
        get_unified_registry().add_link_library(lib)
        # Group-level deps — for precise dependency tracking
        if caller_group_key:
            from .build.deps import get_dependency_tracker
            get_dependency_tracker().record_extern_dependency(
                caller_group_key, [lib],
            )

    # --- Build PC func type ---
    from .builtin_entities.func import func as func_type_cls
    from .builtin_entities.types import void

    fixed_pc_types = tuple(
        pt for name, pt in extern_wrapper.param_types if name != 'args'
    )
    ret_type = extern_wrapper.return_type if extern_wrapper.return_type is not None else void
    callable_pc_type = func_type_cls[fixed_pc_types + (ret_type,)]

    return wrap_value(ir_func, kind='pointer', type_hint=callable_pc_type)
