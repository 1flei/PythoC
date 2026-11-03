"""
Structural normalization of compile-time factory inputs.

This module provides normalize_factory_key() which converts factory
arguments into deterministic suffix strings for caching and
specialization.

Supported types:
  - None, bool, int, float, str, bytes
  - tuple/list of supported values
  - dict with supported keys and values
  - PythoC type objects (via the lossless type ID projection)
"""

import base64
import json

from ..type_descriptor import specialization_key


def _normalize_value(value, _active=None):
    active = set() if _active is None else _active
    if not isinstance(value, (tuple, list, dict)):
        return _display_value(value, active)
    if id(value) in active:
        raise TypeError('Cannot normalize cyclic factory input')
    active.add(id(value))
    try:
        return _display_value(value, active)
    finally:
        active.remove(id(value))


def _display_value(value, active):
    """Recursively normalize a single value into a canonical string.

    Returns a string fragment that is deterministic for equal inputs.

    Raises:
        TypeError: If the value type is not supported for normalization.
    """
    if value is None:
        return "None"

    if isinstance(value, bool):
        # Must check bool before int since bool is subclass of int
        return "True" if value else "False"

    if isinstance(value, int):
        return "i{}".format(value)

    if isinstance(value, float):
        return "f{}".format(repr(value))

    if isinstance(value, str):
        return "s{}".format(repr(value))

    if isinstance(value, bytes):
        return "b{}".format(repr(value))

    if isinstance(value, tuple):
        inner = ",".join(_normalize_value(v, active) for v in value)
        return "T({})".format(inner)

    if isinstance(value, list):
        inner = ",".join(_normalize_value(v, active) for v in value)
        return "L({})".format(inner)

    if isinstance(value, dict):
        # Sort by normalized key for determinism
        pairs = []
        for k in sorted(value.keys(), key=lambda k: _normalize_value(k, active)):
            pairs.append("{}:{}".format(_normalize_value(k, active), _normalize_value(value[k])))
        return "D({})".format(",".join(pairs))

    # PythoC builtin type objects (have get_name + get_llvm_type)
    from ..builtin_entities.base import _is_type_like
    if _is_type_like(value):
        from ..type_id import get_type_id
        return "PC({})".format(get_type_id(value))

    # PythoC builtin entities (have _name attribute)
    if hasattr(value, '_name') and hasattr(value, '__class__'):
        cls_name = type(value).__name__
        return "BE({}.{})".format(cls_name, value._name)

    raise TypeError(
        "Cannot normalize factory input of type {}. "
        "Supported: None, bool, int, float, str, bytes, tuple, list, dict, "
        "PythoC type objects.".format(type(value).__name__)
    )


def normalize_factory_key(*args, **kwargs):
    """Normalize factory arguments into a deterministic suffix string.

    The resulting string is suitable for use as a compile_suffix to
    distinguish different specializations of the same factory.

    Args:
        *args: Positional factory arguments.
        **kwargs: Keyword factory arguments.

    Returns:
        A deterministic string representing the normalized factory key.

    Raises:
        TypeError: If any argument type is not supported for normalization.

    Examples:
        >>> normalize_factory_key(1, 2, 3)
        'i1_i2_i3'
        >>> normalize_factory_key((1, 2), name="foo")
        'T(i1,i2)_KW_name_s\\'foo\\''
    """
    parts = []

    # Normalize positional args
    for arg in args:
        parts.append(_normalize_value(arg))

    # Normalize keyword args (sorted for determinism)
    if kwargs:
        parts.append("KW")
        for key in sorted(kwargs.keys()):
            parts.append(key)
            parts.append(_normalize_value(kwargs[key]))

    if not parts:
        return "empty"
    display = "_".join(parts)
    identity = (specialization_key(args), specialization_key(kwargs))
    wire = json.dumps(identity, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    encoded = base64.urlsafe_b64encode(wire).decode('ascii').rstrip('=')
    return display + '__id_' + encoded
