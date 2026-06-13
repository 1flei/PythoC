"""
instantiate: Hoist compile-time generators to runtime iterators.

instantiate([1, 2, 3])      -> constant iterable
instantiate(yield_fn())     -> yield fn call (zero or more args)
instantiate((f(x) for x))   -> generator expression

Returns api with .Iter / .next(s) / .value(s) / .init(s).
"""

from __future__ import annotations
import ast
from enum import Enum, auto
from typing import Any


class _SourceKind(Enum):
    YIELD_FUNCTION = auto()
    YIELD_CALL = auto()
    GENERATOR_EXPRESSION = auto()
    CLOSURE = auto()
    CONSTANT_ITERABLE = auto()
    UNSUPPORTED = auto()


def instantiate(source: Any) -> Any:
    kind = _classify_source(source)

    if kind in (_SourceKind.YIELD_FUNCTION, _SourceKind.YIELD_CALL):
        from ..meta.instantiate import _instantiate_yield
        if kind is _SourceKind.YIELD_FUNCTION:
            bare_func = source
            fa = getattr(bare_func, '_original_ast', None)
            if fa is None:
                raise TypeError(
                    "instantiate: function has no _original_ast – "
                    "did you forget @compile?")
            class _SyntheticInfo:
                pass
            source = _SyntheticInfo()
            source._yield_inline_info = {
                'original_ast': fa,
                'call_args': [],
                'placeholder': bare_func,
                'callee_globals': getattr(bare_func, '_yield_callee_globals', None) or {},
            }
        return _instantiate_yield(source)

    if kind is _SourceKind.GENERATOR_EXPRESSION:
        from ..meta.instantiate import _instantiate_genexpr
        return _instantiate_genexpr(source)

    if kind is _SourceKind.CLOSURE:
        from ..meta.instantiate import _instantiate_closure
        return _instantiate_closure(source)

    if kind is _SourceKind.CONSTANT_ITERABLE:
        from ..meta.instantiate import _instantiate_const_iterable
        return _instantiate_const_iterable(list(source))

    raise TypeError(
        f"instantiate: unsupported source {type(source).__name__}"
    )


def _classify_source(source: Any) -> _SourceKind:
    if callable(source) and hasattr(source, '_original_ast'):
        return _SourceKind.YIELD_FUNCTION
    if hasattr(source, '_yield_inline_info') and source._yield_inline_info:
        return _SourceKind.YIELD_CALL
    if _is_genexpr(source):
        return _SourceKind.GENERATOR_EXPRESSION
    if _is_closure_source(source):
        return _SourceKind.CLOSURE
    if hasattr(source, '__iter__'):
        return _SourceKind.CONSTANT_ITERABLE
    return _SourceKind.UNSUPPORTED


def _is_genexpr(obj: Any) -> bool:
    info = getattr(obj, '_pc_generator_expr_info', None)
    if info:
        return info.get('kind') == 'generator_expression'
    return getattr(obj, '_pc_is_generator_expr', False)


def _is_closure_source(obj: Any) -> bool:
    return (
        isinstance(getattr(obj, 'func_ast', None), ast.FunctionDef)
        and isinstance(getattr(obj, 'func_globals', None), dict)
    )


# ------------------------------------------------------------------
# Compile-time call protocol
# ------------------------------------------------------------------
# When ``instantiate`` is used inside a @compile function, the AST
# visitor normally routes Python calls through ``PythonType._eval_call``,
# which unwraps ValueRef arguments (passing ``arg.value``).  That
# strips the ``_yield_inline_info`` / ``_pc_generator_expr_info``
# metadata attached to the ValueRef wrapper.  By exposing
# ``handle_call`` on the function object itself we bypass
# ``_eval_call`` and receive raw ValueRefs.
# ------------------------------------------------------------------

def _instantiate_handle_call(visitor, func_ref, args, node):
    """AST-level compile-time handler for ``instantiate(...)`` calls."""
    from ..valueref import ValueRef, wrap_value
    from ..builtin_entities.python_type import PythonType

    if not args:
        raise TypeError("instantiate() missing 1 required positional argument")

    source = args[0]

    kind = _classify_source(source)
    if kind is not _SourceKind.UNSUPPORTED:
        result = instantiate(source)
        return wrap_value(
            result,
            kind="python",
            type_hint=PythonType(result, is_constant=True),
        )

    if isinstance(source, ValueRef) and source.is_python_value():
        result = instantiate(source.value)
        return wrap_value(
            result,
            kind="python",
            type_hint=PythonType(result, is_constant=True),
        )

    result = instantiate(source)
    return wrap_value(
        result,
        kind="python",
        type_hint=PythonType(result, is_constant=True),
    )


instantiate.handle_call = _instantiate_handle_call
