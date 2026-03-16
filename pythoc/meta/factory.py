"""
Factory decorators for compile-time code generation.

This module provides @meta.factory and @meta.compile_factory which
turn Python functions into compile-time factories. Factory inputs are
structurally normalized into compile suffixes for caching and
specialization

Example::

    @meta.compile_factory
    def make_adder(offset):
        return meta.func(
            name="adder",
            params=[("x", i32)],
            return_type=i32,
            body=[ast.Return(value=ast.BinOp(
                left=ast.Name(id='x', ctx=ast.Load()),
                op=ast.Add(),
                right=ast.Constant(value=offset),
            ))],
        )

    add_10 = make_adder(10)
    add_20 = make_adder(20)
"""

import inspect
from functools import wraps
from typing import Any, Dict, Optional

from .normalize import normalize_factory_key
from .generated import GeneratedFunction, MetaArtifact
from .compile_api import compile_ast, compile_generated, compile_artifact


def factory(func=None, *, cache=True):
    """Decorator: wrapped function returns GeneratedFunction or MetaArtifact.

    The decorated function becomes a factory callable. Each call with
    different arguments produces a new GeneratedFunction or MetaArtifact.

    If cache=True (default), results are cached by normalized factory key
    so the same arguments always return the same object.

    Args:
        func: The factory function (when used as bare decorator).
        cache: Whether to cache factory results (default True).

    Returns:
        A factory callable.

    Example::

        @meta.factory
        def make_doubler(ty):
            return meta.func(
                name="doubler",
                params=[("x", ty)],
                return_type=ty,
                body=[...],
            )

        gf = make_doubler(i32)  # GeneratedFunction
    """
    def decorator(f):
        _cache = {} if cache else None

        @wraps(f)
        def wrapper(*args, **kwargs):
            if _cache is not None:
                key = normalize_factory_key(*args, **kwargs)
                if key in _cache:
                    return _cache[key]

            result = f(*args, **kwargs)

            if not isinstance(result, (GeneratedFunction, MetaArtifact)):
                raise TypeError(
                    "@meta.factory function must return GeneratedFunction or "
                    "MetaArtifact, got {}".format(type(result).__name__)
                )

            if _cache is not None:
                _cache[key] = result

            return result

        wrapper._is_meta_factory = True
        wrapper._factory_func = f
        wrapper._factory_cache = _cache
        return wrapper

    if func is None:
        return decorator
    return decorator(func)


def compile_factory(func=None, *, cache=True, suffix=None, attrs=None):
    """Decorator: wrapped function returns GeneratedFunction/MetaArtifact,
    which is immediately compiled. Returns compiled wrapper.

    The decorated function becomes a factory callable that directly
    returns compiled wrappers. Factory inputs are structurally normalized
    into compile suffixes for specialization.

    If cache=True (default), compiled wrappers are cached by factory key
    so the same arguments always return the same compiled function.

    Args:
        func: The factory function (when used as bare decorator).
        cache: Whether to cache compiled results (default True).
        suffix: Additional user-provided suffix for specialization.
        attrs: LLVM function-level attributes to apply.

    Returns:
        A factory callable that returns compiled wrappers.

    Example::

        @meta.compile_factory
        def make_adder(offset):
            return meta.func(
                name="adder",
                params=[("x", i32)],
                return_type=i32,
                body=[...],
            )

        add_10 = make_adder(10)   # compiled wrapper
        add_20 = make_adder(20)   # different compiled wrapper
        add_10b = make_adder(10)  # same as add_10 (cached)
    """
    def decorator(f):
        _cache = {} if cache else None

        # Capture caller's globals for compilation
        frame = inspect.currentframe()
        try:
            caller = frame.f_back
            if caller is not None:
                caller_back = caller.f_back
                caller_globals = dict(caller_back.f_globals) if caller_back else {}
                caller_file = caller_back.f_code.co_filename if caller_back else "<meta_factory>"
            else:
                caller_globals = {}
                caller_file = "<meta_factory>"
        finally:
            del frame

        @wraps(f)
        def wrapper(*args, **kwargs):
            # Normalize factory arguments into a suffix
            factory_suffix = normalize_factory_key(*args, **kwargs)

            # Combine with user-provided suffix
            if suffix:
                combined_suffix = "{}_{}".format(suffix, factory_suffix)
            else:
                combined_suffix = factory_suffix

            # Check cache
            if _cache is not None and combined_suffix in _cache:
                return _cache[combined_suffix]

            # Call the user function to get GeneratedFunction or MetaArtifact
            result = f(*args, **kwargs)

            # Compile based on result type
            if isinstance(result, GeneratedFunction):
                compiled = compile_generated(
                    result,
                    user_globals=caller_globals,
                    suffix=combined_suffix,
                    source_file=caller_file,
                )
            elif isinstance(result, MetaArtifact):
                compiled = compile_artifact(
                    result,
                    user_globals=caller_globals,
                    suffix=combined_suffix,
                    source_file=caller_file,
                )
            else:
                raise TypeError(
                    "@meta.compile_factory function must return "
                    "GeneratedFunction or MetaArtifact, got {}".format(
                        type(result).__name__
                    )
                )

            # Cache the compiled wrapper
            if _cache is not None:
                _cache[combined_suffix] = compiled

            return compiled

        wrapper._is_meta_compile_factory = True
        wrapper._factory_func = f
        wrapper._factory_cache = _cache
        return wrapper

    if func is None:
        return decorator
    return decorator(func)
