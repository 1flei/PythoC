"""Literal carrier and composite schema protocols.

This module centralizes carrier shape handling and composite traversal so
callers do not need to inspect tuple/list/dict/struct details directly.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Tuple

from .logger import logger
from .valueref import ValueRef


def _has_flag(value: Any, flag_name: str) -> bool:
    return isinstance(value, type) and getattr(value, flag_name, False)


def is_pc_tuple_type(value: Any) -> bool:
    return _has_flag(value, '_is_pc_tuple')


def is_pc_list_type(value: Any) -> bool:
    return _has_flag(value, '_is_pc_list')


def is_pc_dict_type(value: Any) -> bool:
    return _has_flag(value, '_is_pc_dict')


def is_sequence_carrier(value: Any) -> bool:
    return (
        isinstance(value, (tuple, list))
        or is_pc_tuple_type(value)
        or is_pc_list_type(value)
        or hasattr(value, '_elements') and getattr(value, '_elements') is not None
    )


def is_mapping_carrier(value: Any) -> bool:
    return isinstance(value, dict) or is_pc_dict_type(value)


def get_sequence_elements(value: Any) -> List[Any]:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return list(value)
    if is_pc_tuple_type(value) or is_pc_list_type(value):
        return list(value.get_elements())
    if hasattr(value, '_elements') and getattr(value, '_elements') is not None:
        return list(value._elements)
    logger.error(f"Object is not a sequence carrier: {value}", node=None, exc_type=TypeError)


def get_sequence_length(value: Any) -> int:
    return len(get_sequence_elements(value))


def get_sequence_element(value: Any, index: int) -> Any:
    elements = get_sequence_elements(value)
    if index < 0 or index >= len(elements):
        raise IndexError(index)
    return elements[index]


def get_mapping_entries(value: Any) -> List[Tuple[Any, Any]]:
    if isinstance(value, dict):
        return list(value.items())
    if is_pc_dict_type(value):
        return list(value.get_entries())
    logger.error(f"Object is not a mapping carrier: {value}", node=None, exc_type=TypeError)


def get_mapping_value(value: Any, key: Any) -> Any:
    if isinstance(value, dict):
        return value[key]
    if is_pc_dict_type(value):
        return value.get_value(key)
    logger.error(f"Object is not a mapping carrier: {value}", node=None, exc_type=TypeError)


def unwrap_python_object(value: Any) -> Any:
    if hasattr(value, '_python_object'):
        return value._python_object

    get_python_object = getattr(value, 'get_python_object', None)
    if callable(get_python_object):
        return get_python_object()

    return value


def _resolve_linear_schema_type(type_hint: Any) -> Any:
    from .builtin_entities.refined import RefinedType

    resolved = unwrap_python_object(type_hint)
    if isinstance(resolved, type) and issubclass(resolved, RefinedType):
        base_type = getattr(resolved, '_base_type', None)
        if base_type is not None:
            return _resolve_linear_schema_type(base_type)
    return resolved


def get_composite_field_types(type_hint: Any) -> List[Any]:
    schema_type = _resolve_linear_schema_type(type_hint)
    field_types = getattr(schema_type, '_field_types', None)
    if not field_types:
        return []
    return list(field_types)


def is_linear_schema_type(type_hint: Any) -> bool:
    from .builtin_entities import linear

    schema_type = _resolve_linear_schema_type(type_hint)

    if schema_type is linear:
        return True

    if isinstance(schema_type, type) and hasattr(schema_type, '_is_linear'):
        if schema_type._is_linear:
            return True

    for field_type in get_composite_field_types(schema_type):
        if is_linear_schema_type(field_type):
            return True

    return False


def get_linear_schema_paths(
    type_hint: Any,
    prefix: Tuple[int, ...] = (),
) -> List[Tuple[int, ...]]:
    from .builtin_entities import linear

    schema_type = _resolve_linear_schema_type(type_hint)

    if schema_type is linear:
        return [prefix]

    if isinstance(schema_type, type) and hasattr(schema_type, '_is_linear'):
        if schema_type._is_linear:
            return [prefix]

    paths = []
    for index, field_type in enumerate(get_composite_field_types(schema_type)):
        paths.extend(get_linear_schema_paths(field_type, prefix + (index,)))
    return paths


def project_tracking_metadata(
    base_value_ref: ValueRef,
    child_index: int,
) -> Tuple[Optional[str], Optional[Tuple[int, ...]]]:
    base_var_name = getattr(base_value_ref, 'var_name', None)
    base_linear_path = getattr(base_value_ref, 'linear_path', None)

    if base_var_name and base_linear_path is not None:
        return base_var_name, base_linear_path + (child_index,)

    return None, None


def iter_literal_value_refs(value: Any) -> Iterable[ValueRef]:
    """Yield runtime-tracked ValueRefs stored inside literal carriers.

    Python-backed carrier wrappers are traversed recursively, but are not yielded
    themselves. This lets callers reason about embedded ownership without knowing
    whether the source was tuple/list/dict or pc_tuple/pc_list/pc_dict.
    """
    if isinstance(value, ValueRef):
        if value.is_python_value():
            yield from iter_literal_value_refs(value.get_python_value())
            return
        yield value
        return

    if is_sequence_carrier(value):
        for item in get_sequence_elements(value):
            yield from iter_literal_value_refs(item)
        return

    if is_mapping_carrier(value):
        for key, mapped_value in get_mapping_entries(value):
            yield from iter_literal_value_refs(key)
            yield from iter_literal_value_refs(mapped_value)


def iter_value_ref_leaves(value_ref: ValueRef) -> Iterable[ValueRef]:
    """Yield ValueRefs relevant for ownership and copy checks.

    Non-Python values yield themselves. Python-backed literal carriers yield
    their embedded runtime-tracked leaves instead.
    """
    if value_ref.is_python_value():
        yield from iter_literal_value_refs(value_ref.get_python_value())
        return

    yield value_ref


def unwrap_literal_item(item: Any) -> Any:
    if isinstance(item, ValueRef):
        if item.is_python_value():
            return item.get_python_value()
        if item.type_hint is not None:
            return item.type_hint
    return item


def extract_subscript_items(items: Any) -> Tuple[Any, ...]:
    from .builtin_entities.base import BuiltinType
    from .builtin_entities.refined import RefinedType

    if isinstance(items, type) and issubclass(items, RefinedType):
        tags = getattr(items, '_tags', [])
        if 'tuple' in tags:
            return BuiltinType._extract_tuple_from_refined(items)

    if is_sequence_carrier(items):
        return tuple(unwrap_literal_item(item) for item in get_sequence_elements(items))

    return (items,)


def extract_runtime_tuple_items(struct_type: Any) -> Tuple[Any, ...]:
    from .builtin_entities.base import BuiltinType
    from .builtin_entities.refined import RefinedType

    field_types = getattr(struct_type, '_field_types', [])
    if not field_types:
        logger.error(f"Struct type has no field types: {struct_type}", node=None, exc_type=TypeError)

    items = []
    for field_type in field_types:
        actual_value = unwrap_python_object(field_type)

        if isinstance(actual_value, type) and issubclass(actual_value, RefinedType):
            tags = getattr(actual_value, '_tags', [])
            if 'slice' in tags:
                items.append(BuiltinType._extract_slice_from_refined(actual_value))
                continue
            if 'tuple' in tags:
                items.extend(BuiltinType._extract_tuple_from_refined(actual_value))
                continue

        items.append(actual_value)

    return tuple(items)


def _extract_tagged_tuple_type_items(type_hint: Any) -> Optional[Tuple[Any, ...]]:
    from .builtin_entities.refined import RefinedType

    resolved = unwrap_python_object(type_hint)
    if not (isinstance(resolved, type) and issubclass(resolved, RefinedType)):
        return None

    if 'tuple' not in getattr(resolved, '_tags', []):
        return None

    base_type = getattr(resolved, '_base_type', None)
    if base_type is None:
        return None

    return extract_runtime_tuple_items(base_type)


def get_multidim_subscript_indices(visitor, index_ref: ValueRef, node) -> Optional[List[ValueRef]]:
    """Normalize multi-dimensional index carriers into a list of ValueRefs."""
    if index_ref.is_python_value():
        python_value = index_ref.get_python_value()
        if is_sequence_carrier(python_value):
            return [wrap_literal_result(item) for item in get_sequence_elements(python_value)]
        return None

    tagged_tuple_items = _extract_tagged_tuple_type_items(getattr(index_ref, 'type_hint', None))
    if tagged_tuple_items is not None:
        return [wrap_literal_result(item) for item in tagged_tuple_items]

    if index_ref.is_struct_value():
        pc_type = index_ref.get_pc_type()
        get_all_fields = getattr(pc_type, 'get_all_fields', None)
        if callable(get_all_fields):
            return list(get_all_fields(visitor, index_ref, node))

    return None


def get_unpack_values(visitor, value_ref: ValueRef, node) -> List[ValueRef]:
    """Return unpackable children through the shared carrier protocol."""
    if value_ref.is_python_value():
        python_value = value_ref.get_python_value()
        if not is_sequence_carrier(python_value):
            logger.error(
                f"Cannot unpack Python value of type {type(python_value)}",
                node=node,
                exc_type=TypeError,
            )
        return [wrap_literal_result(item) for item in get_sequence_elements(python_value)]

    type_hint = getattr(value_ref, 'type_hint', None)
    if type_hint is None:
        logger.error("Cannot unpack value without type information", node=node, exc_type=TypeError)

    if hasattr(type_hint, 'get_all_fields'):
        return list(type_hint.get_all_fields(visitor, value_ref, node))

    logger.error(f"Unsupported unpacking type: {type_hint}", node=node, exc_type=TypeError)


def rebuild_sequence_carrier(template: Any, elements: List[Any]) -> Any:
    from .builtin_entities.pc_list import pc_list
    from .builtin_entities.pc_tuple import pc_tuple

    if isinstance(template, list):
        return list(elements)
    if isinstance(template, tuple):
        return tuple(elements)
    if is_pc_list_type(template):
        return pc_list.from_elements(elements)
    if is_pc_tuple_type(template):
        return pc_tuple.from_elements(elements)
    if hasattr(template, '_elements') and getattr(template, '_elements') is not None:
        if getattr(template, '_is_pc_list', False):
            return pc_list.from_elements(elements)
        return pc_tuple.from_elements(elements)

    logger.error(
        f"Object is not a sequence carrier template: {template}",
        node=None,
        exc_type=TypeError,
    )


def rebuild_mapping_carrier(template: Any, entries: List[Tuple[Any, Any]]) -> Any:
    from .builtin_entities.pc_dict import pc_dict

    if isinstance(template, dict):
        return dict(entries)
    if is_pc_dict_type(template):
        return pc_dict.from_entries(entries)

    logger.error(
        f"Object is not a mapping carrier template: {template}",
        node=None,
        exc_type=TypeError,
    )


def wrap_literal_result(result: Any):
    from .builtin_entities.pc_dict import pc_dict
    from .builtin_entities.pc_list import pc_list
    from .builtin_entities.pc_tuple import pc_tuple
    from .builtin_entities.python_type import PythonType
    from .valueref import ValueRef, wrap_value

    if isinstance(result, ValueRef):
        return result

    if isinstance(result, tuple):
        elements = [wrap_literal_result(item) for item in result]
        tuple_type = pc_tuple.from_elements(elements)
        return wrap_value(tuple_type, kind='python', type_hint=PythonType.wrap(tuple_type, is_constant=True))

    if isinstance(result, list):
        elements = [wrap_literal_result(item) for item in result]
        list_type = pc_list.from_elements(elements)
        return wrap_value(list_type, kind='python', type_hint=PythonType.wrap(list_type, is_constant=True))

    if isinstance(result, dict):
        entries = [(wrap_literal_result(key), wrap_literal_result(value)) for key, value in result.items()]
        dict_type = pc_dict.from_entries(entries)
        return wrap_value(dict_type, kind='python', type_hint=PythonType.wrap(dict_type, is_constant=True))

    python_type = PythonType.wrap(result, is_constant=True)
    return wrap_value(result, kind='python', type_hint=python_type)
