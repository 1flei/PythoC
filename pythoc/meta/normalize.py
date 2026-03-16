"""
Structural normalization of compile-time factory inputs.

This module provides normalize_factory_key() which converts factory
arguments into deterministic suffix strings for caching and
specialization.

Supported types:
  - None, bool, int, float, str, bytes
  - tuple/list of supported values
  - dict with supported keys and values
  - PythoC builtin type objects (via canonical identity / get_name())
"""


_SCALAR_TYPES = (type(None), bool, int, float, str, bytes)


def _normalize_value(value):
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
        inner = ",".join(_normalize_value(v) for v in value)
        return "T({})".format(inner)

    if isinstance(value, list):
        inner = ",".join(_normalize_value(v) for v in value)
        return "L({})".format(inner)

    if isinstance(value, dict):
        # Sort by normalized key for determinism
        pairs = []
        for k in sorted(value.keys(), key=lambda k: _normalize_value(k)):
            pairs.append("{}:{}".format(_normalize_value(k), _normalize_value(value[k])))
        return "D({})".format(",".join(pairs))

    # PythoC builtin type objects (have get_name + get_llvm_type)
    if hasattr(value, 'get_name') and hasattr(value, 'get_llvm_type'):
        return "PC({})".format(value.get_name())

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

    return "_".join(parts) if parts else "empty"
