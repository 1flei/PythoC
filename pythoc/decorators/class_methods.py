# -*- coding: utf-8 -*-
"""Class-body method support for decorated classes.

This module gives @compile (struct), @union and @enum decorated classes the
ability to declare methods directly inside the class body using a plain
``def f(...):`` form. Each such method is compiled exactly as if it had been
written outside the class with ``@compile`` and then attached as a class
attribute, mirroring the long-standing manual pattern used in
``pythoc/std/vector.py``.

Design summary
==============
* Methods do **not** receive a synthetic ``self`` -- the first parameter is a
  regular pythoc parameter, identical to today's manual pattern.
* Plain methods are compiled with an implicit class-qualified suffix. This
  gives ``A.make`` and ``B.make`` distinct native symbols while preserving the
  user-facing ``Cls.method(...)`` spelling. The enclosing class
  compile_suffix is included in that method suffix so generic factories (e.g.
  ``Vector(int)``) remain deduplicated per instantiation. A method may still
  be explicitly decorated with ``@compile(suffix=...)`` to override; such
  members are detected via ``_is_compiled`` and attached as-is.
* Non-function descriptors (``classmethod``/``staticmethod``/``property`` ...)
  are left alone so users keep their full Python-side flexibility.
* The class object itself is injected into the per-method symbol namespace so
  self-referential annotations such as ``ptr[_Vector]`` resolve correctly.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from ..logger import logger


def extract_class_method_defs(source_cls) -> List[Tuple[str, ast.FunctionDef]]:
    """Return ``(name, FunctionDef)`` pairs for plain ``def`` members.

    The class source is parsed once. Async functions, nested classes and any
    other non-FunctionDef node are ignored. Results are returned in source
    order so that registration ordering is deterministic.
    """
    try:
        raw_source = inspect.getsource(source_cls)
    except (OSError, TypeError):
        return []

    source = textwrap.dedent(raw_source)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        logger.error(
            f"Failed to parse class body for '{source_cls.__name__}'",
            node=None,
            exc_type=SyntaxError,
        )

    if not tree.body or not isinstance(tree.body[0], ast.ClassDef):
        return []

    methods: List[Tuple[str, ast.FunctionDef]] = []
    for stmt in tree.body[0].body:
        if isinstance(stmt, ast.FunctionDef):
            methods.append((stmt.name, stmt))
    return methods


def attach_class_methods(
    target_cls,
    source_cls=None,
    class_compile_suffix: Optional[str] = None,
    captured_symbols: Optional[Dict[str, Any]] = None,
) -> None:
    """Compile each plain-def method of ``source_cls`` and attach to ``target_cls``.

    Parameters
    ----------
    target_cls:
        The class object that will receive the compiled wrappers. For struct
        and union this is the same class that was decorated; for enum the
        decorator returns a freshly built class so ``target_cls`` differs from
        ``source_cls``.
    source_cls:
        The class whose source/dict the methods are read from. Defaults to
        ``target_cls``.
    class_compile_suffix:
        compile_suffix value to inherit on each method that is not already a
        compiled wrapper.
    captured_symbols:
        Symbols visible at class-decoration time (typically the factory's
        locals). Threaded into ``_compile_impl`` so that method annotations
        resolve closure-captured names. The class itself is added under its
        own ``__name__`` so self-referential annotations resolve.
    """
    if source_cls is None:
        source_cls = target_cls

    methods = extract_class_method_defs(source_cls)
    if not methods:
        return

    # Build the per-method symbol namespace once. The decorated class is
    # registered under its own name so methods may reference it (e.g.
    # ``ptr[_Vector]`` inside ``def push_back(...)``). We do not mutate the
    # caller's dict.
    method_symbols: Dict[str, Any] = dict(captured_symbols or {})
    method_symbols.setdefault(target_cls.__name__, target_cls)
    if source_cls is not target_cls:
        method_symbols.setdefault(source_cls.__name__, target_cls)

    # Imported lazily to avoid circular import at module load time.
    from .compile import _compile_impl

    attached: List[str] = []
    class_suffix_parts = [target_cls.__name__]
    if class_compile_suffix:
        class_suffix_parts.append(class_compile_suffix)
    implicit_method_suffix = "_".join(class_suffix_parts)

    for name, _ast_node in methods:
        member = source_cls.__dict__.get(name)
        if member is None:
            # Should not happen for AST-discovered methods, but be defensive.
            continue

        # Already-compiled wrapper from an explicit @compile(...) on the
        # method: keep it verbatim, just make sure it lives on target_cls.
        if getattr(member, "_is_compiled", False):
            setattr(target_cls, name, member)
            attached.append(name)
            continue

        # Skip descriptors (classmethod/staticmethod/property/...). Anything
        # not a plain function falls through to its native Python behaviour.
        if not inspect.isfunction(member):
            continue

        wrapper = _compile_impl(
            member,
            compile_suffix=implicit_method_suffix,
            effect_suffix=None,
            captured_symbols=method_symbols,
        )
        setattr(target_cls, name, wrapper)
        attached.append(name)

    if attached:
        logger.debug(
            f"attach_class_methods: {target_cls.__name__} <- {attached}"
        )


def lookup_class_method(decorated_cls, attr_name: str):
    """Return a class-attached compiled wrapper or ``None``.

    Decorated classes (struct/union/enum) treat field/variant access as their
    primary attribute protocol. When the requested name is not a field/variant,
    the protocol falls back here to surface methods attached by
    :func:`attach_class_methods` (or manually attached compiled functions, as
    in the legacy ``pythoc/std/vector.py`` ``api`` pattern). Returning ``None``
    lets the caller emit its own diagnostic if neither lookup succeeds.

    The struct/union internal ``handle_attribute`` is a classmethod bound to
    the unified type (e.g. ``StructType`` subclass) rather than to the user's
    Python class. ``_python_class`` is checked as a transparent redirect so
    callers do not need to special-case which class object they hold.
    """
    candidates = [decorated_cls]
    py_cls = getattr(decorated_cls, "_python_class", None)
    if py_cls is not None and py_cls is not decorated_cls:
        candidates.append(py_cls)

    for candidate in candidates:
        member = getattr(candidate, attr_name, None)
        if member is not None and getattr(member, "_is_compiled", False):
            return member
    return None


def wrap_class_method_as_python_value(member, node):
    """Wrap a compiled class-method wrapper as a python-value ValueRef.

    Centralised here so struct/union/enum dispatchers share the same fallback
    code. The returned ValueRef behaves like ``Cls.method`` from any other
    decoration path -- subsequent ``handle_call`` will use the wrapper's own
    ``handle_call`` protocol.
    """
    from ..valueref import wrap_value
    from ..builtin_entities.python_type import PythonType

    py_type = PythonType.wrap(member, is_constant=True)
    return wrap_value(member, kind="python", type_hint=py_type)
