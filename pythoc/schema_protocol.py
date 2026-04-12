"""Schema and field-layout protocol helpers.

This module centralizes access to field-bearing PC types so callers do not
inspect `_field_types` or `_field_names` directly.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from .logger import logger


FieldLayoutEntry = Tuple[int, Optional[str], Any]


def unwrap_python_object(value: Any) -> Any:
    if hasattr(value, '_python_object'):
        return value._python_object

    get_python_object = getattr(value, 'get_python_object', None)
    if callable(get_python_object):
        return get_python_object()

    return value


def resolve_schema_type(type_hint: Any) -> Any:
    from .builtin_entities.refined import RefinedType

    resolved = unwrap_python_object(type_hint)
    if isinstance(resolved, type) and issubclass(resolved, RefinedType):
        base_type = getattr(resolved, '_base_type', None)
        if base_type is not None:
            return resolve_schema_type(base_type)
    return resolved


def _ensure_field_layout_resolved(type_hint: Any) -> Any:
    field_owner = resolve_schema_type(type_hint)
    if not isinstance(field_owner, type):
        return field_owner

    ensure = getattr(field_owner, '_ensure_field_types_resolved', None)
    if callable(ensure):
        ensure()

    return field_owner


def _is_literal_carrier_layout(field_owner: Any) -> bool:
    if not isinstance(field_owner, type):
        return False

    if any(
        getattr(field_owner, flag_name, False)
        for flag_name in ('_is_pc_tuple', '_is_pc_list', '_is_pc_dict')
    ):
        return True

    elements = getattr(field_owner, '_elements', None)
    field_names = getattr(field_owner, '_field_names', None)
    python_class = getattr(field_owner, '_python_class', None)
    if elements is not None and not field_names and python_class is None:
        return True

    return False


def get_field_layout_owner(type_hint: Any) -> Optional[type]:
    field_owner = _ensure_field_layout_resolved(type_hint)
    if not isinstance(field_owner, type):
        return None

    if getattr(field_owner, '_field_types', None) is None:
        return None

    return field_owner


def has_field_layout(type_hint: Any) -> bool:
    return get_field_layout_owner(type_hint) is not None


def get_field_layout_types(type_hint: Any) -> List[Any]:
    field_owner = get_field_layout_owner(type_hint)
    if field_owner is None:
        return []

    field_types = getattr(field_owner, '_field_types', None)
    if not field_types:
        return []

    return list(field_types)


def get_field_layout_names(type_hint: Any) -> List[Optional[str]]:
    field_owner = get_field_layout_owner(type_hint)
    if field_owner is None:
        return []

    field_names = getattr(field_owner, '_field_names', None)
    if not field_names:
        return []

    return list(field_names)


def iter_field_layout(type_hint: Any) -> List[FieldLayoutEntry]:
    field_types = get_field_layout_types(type_hint)
    field_names = get_field_layout_names(type_hint)

    entries: List[FieldLayoutEntry] = []
    for index, field_type in enumerate(field_types):
        field_name = field_names[index] if index < len(field_names) else None
        entries.append((index, field_name, field_type))
    return entries


def get_field_layout_count(type_hint: Any) -> int:
    return len(get_field_layout_types(type_hint))


def get_field_layout_type(type_hint: Any, index: int) -> Any:
    field_types = get_field_layout_types(type_hint)
    if index < 0 or index >= len(field_types):
        logger.error(
            f"Field layout index {index} out of range for {type_hint}",
            node=None,
            exc_type=IndexError,
        )
    return field_types[index]


def get_field_layout_name(type_hint: Any, index: int) -> Optional[str]:
    field_names = get_field_layout_names(type_hint)
    if index < 0:
        logger.error(
            f"Field layout index {index} out of range for {type_hint}",
            node=None,
            exc_type=IndexError,
        )
    return field_names[index] if index < len(field_names) else None


def get_schema_type_owner(type_hint: Any) -> Optional[type]:
    field_owner = get_field_layout_owner(type_hint)
    if field_owner is None:
        return None

    if _is_literal_carrier_layout(field_owner):
        return None

    return field_owner


def is_schema_type(type_hint: Any) -> bool:
    return get_schema_type_owner(type_hint) is not None


def get_schema_field_types(type_hint: Any) -> List[Any]:
    schema_owner = get_schema_type_owner(type_hint)
    if schema_owner is None:
        return []
    return get_field_layout_types(schema_owner)


def get_schema_field_names(type_hint: Any) -> List[Optional[str]]:
    schema_owner = get_schema_type_owner(type_hint)
    if schema_owner is None:
        return []
    return get_field_layout_names(schema_owner)


def iter_schema_fields(type_hint: Any) -> List[FieldLayoutEntry]:
    schema_owner = get_schema_type_owner(type_hint)
    if schema_owner is None:
        return []
    return iter_field_layout(schema_owner)


def get_schema_field_count(type_hint: Any) -> int:
    return len(get_schema_field_types(type_hint))


def get_schema_field_type(type_hint: Any, index: int) -> Any:
    field_types = get_schema_field_types(type_hint)
    if index < 0 or index >= len(field_types):
        logger.error(
            f"Schema field index {index} out of range for {type_hint}",
            node=None,
            exc_type=IndexError,
        )
    return field_types[index]


def get_schema_field_name(type_hint: Any, index: int) -> Optional[str]:
    field_names = get_schema_field_names(type_hint)
    if index < 0:
        logger.error(
            f"Schema field index {index} out of range for {type_hint}",
            node=None,
            exc_type=IndexError,
        )
    return field_names[index] if index < len(field_names) else None


def is_linear_schema_type(type_hint: Any) -> bool:
    from .builtin_entities import linear

    schema_type = resolve_schema_type(type_hint)

    if schema_type is linear:
        return True

    if isinstance(schema_type, type) and getattr(schema_type, '_is_linear', False):
        return True

    for field_type in get_schema_field_types(schema_type):
        if is_linear_schema_type(field_type):
            return True

    return False


def get_linear_schema_paths(
    type_hint: Any,
    prefix: Tuple[int, ...] = (),
) -> List[Tuple[int, ...]]:
    from .builtin_entities import linear

    schema_type = resolve_schema_type(type_hint)

    if schema_type is linear:
        return [prefix]

    if isinstance(schema_type, type) and getattr(schema_type, '_is_linear', False):
        return [prefix]

    paths = []
    for index, _, field_type in iter_schema_fields(schema_type):
        paths.extend(get_linear_schema_paths(field_type, prefix + (index,)))
    return paths
