"""Literal carrier protocols.

This module centralizes tuple/list/dict carrier handling so callers do not
need to inspect literal wrapper details directly.
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


def is_pc_literal(value: Any) -> bool:
    from .builtin_entities.pc_literal import pc_literal
    return isinstance(value, pc_literal)


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
    if is_sequence_carrier(items):
        return tuple(unwrap_literal_item(item) for item in get_sequence_elements(items))

    return (items,)


def get_multidim_subscript_indices(visitor, index_ref: ValueRef, node) -> Optional[List[ValueRef]]:
    """Normalize multi-dimensional index carriers into a list of ValueRefs.

    Supported inputs:
    - python-backed literal sequence carriers like pc_tuple/pc_list
    - runtime struct values whose fields are used as indices
    """
    if index_ref.is_python_value():
        python_value = index_ref.get_python_value()
        if is_sequence_carrier(python_value):
            return [wrap_literal_result(item) for item in get_sequence_elements(python_value)]
        return None

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


def aggregate_layout(visitor, target_pc_type):
    """Return ``(llvm_aggregate_type, [slot_pc_type, ...])`` for an array/struct
    target, or None for non-aggregates.

    Array slots repeat the (possibly inner-array) element type ``size`` times;
    struct slots are the field types. This is the single description of an
    aggregate's shape shared by the constant and value materializers, so the two
    never diverge on element typing or zero-fill counts.
    """
    from .schema_protocol import is_schema_type, get_schema_field_types
    from .type_converter import strip_qualifiers

    target_pc_type = strip_qualifiers(target_pc_type)
    ctx = visitor.module.context

    if hasattr(target_pc_type, 'is_array') and target_pc_type.is_array():
        dims = target_pc_type.dimensions
        if not isinstance(dims, (list, tuple)):
            dims = [dims]
        size = dims[0]
        if len(dims) == 1:
            inner_type = target_pc_type.element_type
        else:
            from .builtin_entities import array
            inner_type = array[(target_pc_type.element_type,) + tuple(dims[1:])]
        return target_pc_type.get_llvm_type(ctx), [inner_type] * size

    if is_schema_type(target_pc_type):
        field_types = get_schema_field_types(target_pc_type)
        if field_types is None:
            return None
        return target_pc_type.get_llvm_type(ctx), list(field_types)

    return None


def _lower_aggregate_slots(visitor, carrier, target_pc_type, element_fn):
    """Lower a sequence literal onto an aggregate's slots, or return None.

    Walks the shared :func:`aggregate_layout`, lowering each provided element
    through ``element_fn(visitor, elem, slot_pc_type)`` and zero-filling any
    trailing slots. Returns ``(llvm_aggregate_type, [ir_value, ...])``. Returns
    None when the target is not an aggregate, there are too many initializers, or
    ``element_fn`` rejects an element (by returning None).
    """
    if not is_sequence_carrier(carrier):
        return None
    layout = aggregate_layout(visitor, target_pc_type)
    if layout is None:
        return None
    agg_llvm, slot_types = layout
    elements = get_sequence_elements(carrier)
    if len(elements) > len(slot_types):
        return None

    ctx = visitor.module.context
    tc = visitor.type_converter
    values = []
    for i, slot_type in enumerate(slot_types):
        if i < len(elements):
            v = element_fn(visitor, elements[i], slot_type)
            if v is None:
                return None
        else:
            v = tc.create_zero_constant(slot_type.get_llvm_type(ctx))
        values.append(v)
    return agg_llvm, values


def lower_sequence_to_constant(visitor, carrier, target_pc_type):
    """Fold a sequence literal carrier to a single ir.Constant, or return None.

    The constant-only view of aggregate materialization: every leaf must fold to
    an ir.Constant (so the result is usable as a static/global initializer, which
    cannot run instructions). Any non-constant leaf or unsupported shape returns
    None. Builder-free, so it is valid at global scope.
    """
    from llvmlite import ir

    lowered = _lower_aggregate_slots(
        visitor, carrier, target_pc_type, _lower_element_to_constant)
    if lowered is None:
        return None
    agg_llvm, consts = lowered
    return ir.Constant(agg_llvm, consts)


def materialize_sequence_value(visitor, carrier, target_pc_type):
    """Build an aggregate IR *value* from a sequence literal (array or struct).

    Single pass for both kinds: each element is lowered once, then the aggregate
    is assembled as an ir.Constant when every element folded to a constant, or as
    an ``insertvalue`` chain otherwise. There is no separate constant/runtime
    code path -- constness is just a property of the assembled result. Requires a
    builder (use at function scope); for static seeds use
    :func:`lower_sequence_to_constant`.
    """
    from llvmlite import ir

    lowered = _lower_aggregate_slots(
        visitor, carrier, target_pc_type, _lower_element_value)
    if lowered is None:
        logger.error(
            f"Cannot materialize sequence literal into {target_pc_type}",
            node=None, exc_type=TypeError)
    agg_llvm, values = lowered
    if all(isinstance(v, ir.Constant) for v in values):
        return ir.Constant(agg_llvm, values)
    agg = ir.Constant(agg_llvm, ir.Undefined)
    for i, v in enumerate(values):
        agg = visitor.builder.insert_value(agg, v, i)
    return agg


def _carrier_of(elem):
    """Return the nested sequence carrier inside an element, or None."""
    if isinstance(elem, ValueRef) and elem.is_python_value():
        inner = elem.get_python_value()
        return inner if is_sequence_carrier(inner) else None
    return elem if is_sequence_carrier(elem) else None


def _lower_element_to_constant(visitor, elem, target_pc_type):
    """Fold one aggregate element to an ir.Constant, or return None.

    Nested sequence literals recurse through ``lower_sequence_to_constant``;
    every other leaf goes through ``TypeConverter.convert`` and is accepted only
    if it folds to an ir.Constant.
    """
    from llvmlite import ir
    from .valueref import ensure_ir, wrap_value

    nested = _carrier_of(elem)
    if nested is not None:
        return lower_sequence_to_constant(visitor, nested, target_pc_type)

    if not isinstance(elem, ValueRef):
        from .builtin_entities.python_type import PythonType
        elem = wrap_value(elem, kind="python",
                          type_hint=PythonType.wrap(elem, is_constant=True))
    ir_val = ensure_ir(visitor.type_converter.convert(elem, target_pc_type))
    return ir_val if isinstance(ir_val, ir.Constant) else None


def _lower_element_value(visitor, elem, target_pc_type):
    """Lower one aggregate element to an ir value (constant or runtime).

    Nested sequence literals recurse through ``materialize_sequence_value`` (which
    yields an ir.Constant when it can), so the caller's constness check stays
    accurate; every other leaf goes through ``TypeConverter.convert``.
    """
    from .valueref import ensure_ir, wrap_value

    nested = _carrier_of(elem)
    if nested is not None:
        return materialize_sequence_value(visitor, nested, target_pc_type)

    if not isinstance(elem, ValueRef):
        from .builtin_entities.python_type import PythonType
        elem = wrap_value(elem, kind="python",
                          type_hint=PythonType.wrap(elem, is_constant=True))
    return ensure_ir(visitor.type_converter.convert(elem, target_pc_type))


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
