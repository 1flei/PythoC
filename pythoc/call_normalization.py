"""
Call argument normalization for typed *args / **kwargs.

Public API:
- ``normalize_ast_call_args``  -- AST visit_Call path (keyword binding + carrier packing)
- ``normalize_typed_collectors`` -- func.handle_call convergence point (carrier packing only)
- ``pack_native_call_args``    -- Python wrapper -> native call (ctypes struct packing)
- ``build_varargs_carrier``    -- pc_tuple carrier builder (shared by the above)
- ``build_kwargs_carrier``     -- pc_dict carrier builder (shared by the above)
- ``lower_compile_handle_call``-- shared handle_call for @compile / meta compile_api wrappers

All typed *args / **kwargs carrier packing converges through
``normalize_typed_collectors``, which is called from ``func.handle_call``
(the single LLVM call emission point).
"""

from __future__ import annotations

import copy
import inspect
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

from .valueref import ValueRef, wrap_value
from .logger import logger

if TYPE_CHECKING:
    import ast


class NormalizedCallArgs(list):
    """Arguments bound to formal parameters, including collector carriers.

    Plain sequences contain source-level positional arguments. Only binders
    and producers of native ABI arguments construct this representation.
    """


# ---------------------------------------------------------------------------
# Carrier builders (shared by AST path, func.handle_call, and tests)
# ---------------------------------------------------------------------------

def build_varargs_carrier(extra_positional: List[ValueRef]) -> ValueRef:
    """Build a pc_tuple carrier from excess positional arguments."""
    from .builtin_entities.pc_tuple import create_pc_tuple_type
    from .builtin_entities.python_type import PythonType

    tuple_type = create_pc_tuple_type(list(extra_positional))
    tuple_hint = PythonType.wrap(tuple_type, is_constant=True)
    return wrap_value(tuple_type, kind='python', type_hint=tuple_hint)


def build_kwargs_carrier(kwargs_payload: Dict[str, ValueRef]) -> ValueRef:
    """Build a pc_dict carrier from keyword payload entries."""
    from .builtin_entities.pc_dict import create_pc_dict_type
    from .builtin_entities.python_type import PythonType

    entries = []
    for name, val in kwargs_payload.items():
        key_hint = PythonType.wrap(name, is_constant=True)
        key_ref = wrap_value(name, kind='python', type_hint=key_hint)
        entries.append((key_ref, val))

    dict_type = create_pc_dict_type(entries)
    dict_hint = PythonType.wrap(dict_type, is_constant=True)
    return wrap_value(dict_type, kind='python', type_hint=dict_hint)


# ---------------------------------------------------------------------------
# func.handle_call convergence point
# ---------------------------------------------------------------------------

def normalize_typed_collectors(
    args: Sequence[ValueRef],
    param_types: tuple,
    *,
    has_varargs: bool,
    has_kwargs: bool,
) -> List[ValueRef]:
    """Pack raw positional args into typed *args / **kwargs carriers.

    Called from ``func.handle_call`` for every typed-collector call.
    NormalizedCallArgs explicitly identifies already bound arguments. Raw
    tuple/struct values remain payloads, irrespective of their shape.

    Args:
        args: Raw positional arguments from the caller.
        param_types: The func type's ``param_types`` tuple (lowered layout).
        has_varargs: True if the func was declared with ``*args: T``.
        has_kwargs: True if the func was declared with ``**kwargs: T``.

    Returns:
        Normalized arg list matching ``param_types`` length.
    """
    kwargs_count = 1 if has_kwargs else 0
    varargs_count = 1 if has_varargs else 0
    fixed_count = len(param_types) - varargs_count - kwargs_count

    if isinstance(args, NormalizedCallArgs):
        return args

    normalized = list(args[:fixed_count])
    extra = list(args[fixed_count:])

    if has_varargs:
        normalized.append(build_varargs_carrier(extra))
    else:
        normalized.extend(extra)

    if has_kwargs:
        normalized.append(build_kwargs_carrier({}))

    return NormalizedCallArgs(normalized)


# ---------------------------------------------------------------------------
# AST-level normalization (keyword binding + carrier packing)
# ---------------------------------------------------------------------------

def _get_callee_signature_for_ast(func_ref: ValueRef) -> Optional[Dict[str, Any]]:
    """Extract signature metadata from a callee for AST keyword binding.

    This is only used by ``normalize_ast_call_args`` to resolve keyword
    argument names and detect typed collectors.  It does NOT need to handle
    the ``func.handle_call`` convergence case (that uses ``cls`` directly).
    """
    from .builtin_entities.func import func as func_cls
    from .builtin_entities.python_type import PythonType

    hint = getattr(func_ref, 'type_hint', None)

    value = getattr(func_ref, 'value', None)
    signature = getattr(value, '_pc_call_signature', None)
    if signature is not None:
        return signature

    # 1. func[...] type with param_names
    if hint is not None and isinstance(hint, type) and issubclass(hint, func_cls):
        names = list(getattr(hint, 'param_names', None) or [])
        if names:
            return _build_sig(
                names,
                has_varargs=getattr(hint, 'has_varargs', False),
                has_llvm_varargs=getattr(hint, 'has_llvm_varargs', False),
                has_kwargs=getattr(hint, 'has_kwargs', False),
            )

    # 2. FunctionInfo (with AST node or param_names)
    func_info = _get_func_info(func_ref)
    if func_info is not None:
        ast_node = getattr(func_info, 'ast_node', None)
        if ast_node is not None:
            return _build_sig_from_ast(
                ast_node.args,
                has_llvm_varargs=getattr(func_info, 'has_llvm_varargs', False),
            )

        if func_info.param_names:
            return _build_sig(
                list(func_info.param_names),
                has_varargs=getattr(func_info, 'has_varargs', False),
                has_llvm_varargs=getattr(func_info, 'has_llvm_varargs', False),
                has_kwargs=getattr(func_info, 'has_kwargs', False),
            )

    # 3. ExternFunctionWrapper / raw callable with Python signature
    val = getattr(func_ref, 'value', None) if func_ref else None
    if val is not None:
        extern_config = getattr(val, '_extern_config', None)
        signature = extern_config.get('signature') if extern_config else None
        if signature is None:
            target = getattr(val, 'func', None)
            if target is not None:
                try:
                    signature = inspect.signature(target)
                except (TypeError, ValueError):
                    signature = None
        if signature is not None:
            return _build_sig_from_python(signature)

        pt = getattr(val, 'param_types', None)
        if pt and isinstance(pt, list) and pt and isinstance(pt[0], tuple):
            return _build_sig([name for name, _ in pt])

    return None


def normalize_ast_call_args(
    func_ref: ValueRef,
    positional_args: List[ValueRef],
    keywords: List,  # List[ast.keyword]
    visit_rvalue_expression,  # callable: ast.expr -> ValueRef
    node: 'ast.Call',
) -> List[ValueRef]:
    """Normalize call arguments from an AST Call node.

    Handles keyword binding, positional/keyword conflict detection,
    typed *args packing, typed **kwargs packing, and bare LLVM varargs
    pass-through.
    """
    signature = _get_callee_signature_for_ast(func_ref)
    if signature is None:
        # BuiltinFunction (e.g. llvm_asm) handles keyword args itself via
        # handle_type_call, so bypass normalization and pass through.
        type_hint = getattr(func_ref, 'type_hint', None)
        py_obj = getattr(type_hint, '_python_object', None)
        from .builtin_entities import BuiltinFunction
        if isinstance(py_obj, type) and issubclass(py_obj, BuiltinFunction):
            return positional_args
        # Python-constant callables (PythonType) also bypass: keyword
        # arguments are evaluated as constants by PythonType._eval_call.
        if py_obj is not None:
            return positional_args
        if keywords:
            logger.error(
                "keyword arguments require a callable with named parameters",
                node=node, exc_type=TypeError,
            )
        return positional_args

    return bind_call_args(signature, positional_args, keywords,
                          visit_rvalue_expression, node)


def bind_call_args(signature, positional_args, keywords,
                   visit_rvalue_expression, node):
    """Bind evaluated arguments using the shared parameter-kind metadata."""
    fixed_param_names = signature['fixed_param_names']
    has_varargs = signature['has_varargs']
    has_kwargs = signature['has_kwargs']
    positional_only = signature['positional_only']
    keyword_only = signature['keyword_only']
    positional_names = [name for name in fixed_param_names
                        if name not in keyword_only]
    positional_count = len(positional_names)
    extra_positional = list(positional_args[positional_count:])
    if extra_positional and not (has_varargs or signature['has_llvm_varargs']):
        logger.error('too many positional arguments', node=node,
                     exc_type=TypeError)
    bound = dict(zip(positional_names, positional_args))
    kwargs_payload: Dict[str, ValueRef] = {}

    for kw in keywords:
        if kw.arg is None:
            logger.error(
                "**kwargs splat in calls is not supported",
                node=node, exc_type=TypeError,
            )

        value_ref = visit_rvalue_expression(kw.value)
        if kw.arg in positional_only and not has_kwargs:
            logger.error(f"argument '{kw.arg}' is positional-only", node=node,
                         exc_type=TypeError)
        if kw.arg in fixed_param_names and kw.arg not in positional_only:
            if kw.arg in bound:
                from_positional = (
                    kw.arg in positional_names
                    and positional_names.index(kw.arg) < len(positional_args))
                origin = ('both as positional and keyword' if from_positional
                          else 'more than once as keyword')
                logger.error(
                    f"argument '{kw.arg}' given {origin}",
                    node=node, exc_type=TypeError,
                )
            bound[kw.arg] = value_ref
            continue

        if kw.arg in kwargs_payload:
            logger.error(
                f"duplicate keyword argument: '{kw.arg}'",
                node=node, exc_type=TypeError,
            )

        if not has_kwargs:
            logger.error(
                f"unexpected keyword argument: '{kw.arg}' "
                f"(valid: {', '.join(fixed_param_names)})",
                node=node, exc_type=TypeError,
            )

        kwargs_payload[kw.arg] = value_ref

    for name in fixed_param_names:
        if name not in bound:
            defaults = signature.get('defaults', {})
            if name in defaults:
                bound[name] = defaults[name]
                continue
            logger.error(
                f"missing required argument: '{name}'",
                node=node, exc_type=TypeError,
            )

    normalized = [bound[name] for name in fixed_param_names]

    if has_varargs:
        normalized.append(build_varargs_carrier(extra_positional))
    else:
        normalized.extend(extra_positional)

    if has_kwargs:
        normalized.append(build_kwargs_carrier(kwargs_payload))

    return NormalizedCallArgs(normalized)


# ---------------------------------------------------------------------------
# Shared handle_call for @compile / meta compile_api wrappers
# ---------------------------------------------------------------------------

def lower_compile_handle_call(wrapper, visitor, func_ref, args, node):
    """Shared handle_call implementation for @compile and meta compile_api.

    Converts the Python wrapper to a func type pointer and delegates
    to func.handle_call (which handles normalization if needed).
    """
    from .valueref import wrap_value
    from .builtin_entities import func as func_type_cls
    from .builtin_entities.python_type import PythonType

    wrapper_ref = wrap_value(wrapper, kind="python", type_hint=PythonType.wrap(wrapper))
    converted_func_ref = visitor.type_converter.convert(wrapper_ref, func_type_cls, node)
    func_type = converted_func_ref.type_hint
    return func_type.handle_call(visitor, converted_func_ref, args, node)


# ---------------------------------------------------------------------------
# Python-side native wrapper binding
# ---------------------------------------------------------------------------

def pack_native_call_args(wrapper, args: tuple, kwargs: dict) -> tuple:
    """Bind Python call arguments and lower typed collectors for native calls.

    This handles the Python -> ctypes ABI boundary.  It uses
    ``inspect.signature().bind()`` for parameter binding and constructs
    ctypes structs for typed collectors.
    """
    func_info = getattr(wrapper, '_func_info', None)
    if func_info is None:
        if kwargs:
            target = getattr(wrapper, '__wrapped__', wrapper)
            sig = inspect.signature(target)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            return tuple(bound.arguments.values())
        return args

    has_varargs = getattr(func_info, 'has_varargs', False)
    has_llvm_varargs = getattr(func_info, 'has_llvm_varargs', False)
    has_kwargs = getattr(func_info, 'has_kwargs', False)

    # Bare LLVM varargs: ctypes handles varargs natively, just pass through
    if has_llvm_varargs and not has_varargs and not has_kwargs:
        return args

    if not kwargs and not has_varargs and not has_llvm_varargs and not has_kwargs:
        return args

    target = getattr(wrapper, '__wrapped__', wrapper)
    bound = inspect.signature(target).bind(*args, **kwargs)
    bound.apply_defaults()
    ast_node = func_info.ast_node

    if ast_node is not None:
        normal_param_names = [arg.arg for arg in ast_node.args.args]
        varargs_name = ast_node.args.vararg.arg if ast_node.args.vararg else None
        kwargs_name = ast_node.args.kwarg.arg if ast_node.args.kwarg else None
    else:
        total = len(func_info.param_names)
        collector_count = (1 if has_varargs or has_llvm_varargs else 0) + (1 if has_kwargs else 0)
        normal_count = total - collector_count
        normal_param_names = list(func_info.param_names[:normal_count])
        varargs_name = func_info.param_names[normal_count] if (has_varargs or has_llvm_varargs) else None
        kwargs_name = func_info.param_names[-1] if has_kwargs else None

    result = []
    for name in normal_param_names:
        if name not in bound.arguments:
            raise TypeError(f"missing required argument: '{name}'")
        result.append(bound.arguments[name])

    if has_llvm_varargs:
        result.extend(tuple(bound.arguments.get(varargs_name, ())))
    elif has_varargs:
        varargs_type = func_info.param_type_hints.get(varargs_name)
        if varargs_type is None or not hasattr(varargs_type, 'get_ctypes_type'):
            raise TypeError(f"Cannot build ctypes value for *{varargs_name}")
        collected = tuple(bound.arguments.get(varargs_name, ()))
        result.append(varargs_type.get_ctypes_type()(*collected))

    if has_kwargs:
        kwargs_type = func_info.param_type_hints.get(kwargs_name)
        if kwargs_type is None or not hasattr(kwargs_type, 'get_ctypes_type'):
            raise TypeError(f"Cannot build ctypes value for **{kwargs_name}")

        from .schema_protocol import get_schema_field_names, is_schema_type

        if not is_schema_type(kwargs_type):
            raise TypeError(f"**{kwargs_name}: T requires a schema type, got {kwargs_type}")

        field_names = get_schema_field_names(kwargs_type)
        if any(name is None for name in field_names):
            raise TypeError(f"**{kwargs_name}: T requires named fields")

        payload = dict(bound.arguments.get(kwargs_name, {}))
        unexpected = [name for name in payload if name not in field_names]
        if unexpected:
            raise TypeError(f"unexpected keyword argument: '{unexpected[0]}'")

        missing = [name for name in field_names if name not in payload]
        if missing:
            raise TypeError(f"missing required argument: '{missing[0]}'")

        ordered_vals = [payload[name] for name in field_names]
        result.append(kwargs_type.get_ctypes_type()(*ordered_vals))

    return tuple(result)


# ---------------------------------------------------------------------------
# Internal helpers (not part of public API)
# ---------------------------------------------------------------------------

def _get_func_info(func_ref: ValueRef):
    """Extract FunctionInfo from a func_ref (any source)."""
    from .builtin_entities.python_type import PythonType

    hint = getattr(func_ref, 'type_hint', None)
    func_info = None
    if isinstance(hint, type) and issubclass(hint, PythonType):
        obj = PythonType.unwrap(hint)
        func_info = getattr(obj, '_func_info', None)
    elif isinstance(hint, PythonType):
        obj = hint._python_object
        func_info = getattr(obj, '_func_info', None)
    if func_info is None:
        val = getattr(func_ref, 'value', None) if func_ref else None
        if val is not None:
            func_info = getattr(val, '_func_info', None)
    return func_info


def _build_sig(
    param_names: List[str],
    *,
    has_varargs: bool = False,
    has_llvm_varargs: bool = False,
    has_kwargs: bool = False,
    positional_only=(),
    keyword_only=(),
    defaults=None,
) -> Dict[str, Any]:
    """Build signature metadata dict for AST keyword binding."""
    fixed_count = len(param_names)
    if has_varargs or has_llvm_varargs:
        fixed_count -= 1
    if has_kwargs:
        fixed_count -= 1
    if fixed_count < 0:
        fixed_count = 0

    return {
        'param_names': list(param_names),
        'fixed_param_names': list(param_names[:fixed_count]),
        'has_varargs': has_varargs,
        'has_llvm_varargs': has_llvm_varargs,
        'has_kwargs': has_kwargs,
        'positional_only': tuple(positional_only),
        'keyword_only': tuple(keyword_only),
        'defaults': dict(defaults or {}),
    }


def _build_sig_from_ast(arguments, defaults=None, *, has_llvm_varargs=False):
    fixed = arguments.posonlyargs + arguments.args + arguments.kwonlyargs
    collectors = [arg for arg in (arguments.vararg, arguments.kwarg)
                  if arg is not None]
    return _build_sig(
        [arg.arg for arg in fixed + collectors],
        has_varargs=arguments.vararg is not None and not has_llvm_varargs,
        has_llvm_varargs=has_llvm_varargs,
        has_kwargs=arguments.kwarg is not None,
        positional_only=[arg.arg for arg in arguments.posonlyargs],
        keyword_only=[arg.arg for arg in arguments.kwonlyargs],
        defaults=defaults,
    )


def lower_inline_signature(func_ast, defaults):
    """Retain source binding rules while lowering inline ABI parameters."""
    signature = _build_sig_from_ast(func_ast.args, defaults)
    lowered = copy.deepcopy(func_ast)
    arguments = lowered.args
    collectors = [arg for arg in (arguments.vararg, arguments.kwarg)
                  if arg is not None]
    for arg in collectors:
        if arg.annotation is None:
            raise TypeError('Inline collectors require a type annotation')
    arguments.args = (arguments.posonlyargs + arguments.args
                      + arguments.kwonlyargs + collectors)
    arguments.posonlyargs = []
    arguments.kwonlyargs = []
    arguments.defaults = []
    arguments.kw_defaults = []
    arguments.vararg = None
    arguments.kwarg = None
    return lowered, signature


def _build_sig_from_python(signature: inspect.Signature) -> Dict[str, Any]:
    """Build signature info from a Python ``inspect.Signature``."""
    parameters = list(signature.parameters.values())
    fixed = [param for param in parameters if param.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )]
    varargs = [param for param in parameters
               if param.kind == inspect.Parameter.VAR_POSITIONAL]
    kwargs = [param for param in parameters
              if param.kind == inspect.Parameter.VAR_KEYWORD]

    return _build_sig(
        [param.name for param in fixed + varargs + kwargs],
        has_llvm_varargs=bool(varargs),
        has_kwargs=bool(kwargs),
        positional_only=[param.name for param in fixed
                         if param.kind == inspect.Parameter.POSITIONAL_ONLY],
        keyword_only=[param.name for param in fixed
                      if param.kind == inspect.Parameter.KEYWORD_ONLY],
    )
