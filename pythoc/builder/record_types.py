"""Materialize validated records using target storage representations."""

import threading
import weakref
from typing import NamedTuple

from llvmlite import ir


_materialization_lock = threading.RLock()


class RecordStorage(NamedTuple):
    elements: tuple
    packed: bool
    field_paths: tuple


def union_storage(layout, bit_storage, node):
    """Choose payload words independently of an over-aligned union's carrier."""
    from ..target_layout import get_target_layout

    target = get_target_layout()
    carrier = target.alignment_carrier(layout.alignment, node)
    word_size = target.primitive(carrier).allocation_size
    pointer_size = target.primitive(ir.IntType(8).as_pointer()).allocation_size
    if not bit_storage and word_size > pointer_size:
        word = target.alignment_carrier(pointer_size, node)
        payload = ir.ArrayType(word, layout.allocation_size // pointer_size)
        return ir.LiteralStructType([payload, ir.ArrayType(carrier, 0)])
    return ir.ArrayType(carrier, layout.allocation_size // word_size)


def record_storage(fields, pc_type, node):
    """Derive LLVM elements and field paths from the same record layout."""
    from ..target_layout import (
        align_up, byte_aggregate_layout, get_target_layout, get_type_layout, report_layout_errors,
    )

    visible = [(index, field) for index, field in enumerate(fields) if field is not None]
    paths = [None] * len(fields)
    config = pc_type._layout_config
    if config.is_natural:
        for slot, (index, _) in enumerate(visible):
            paths[index] = (slot,)
        return RecordStorage(tuple(field for _, field in visible), False, tuple(paths))

    # An identified LLVM field can still be opaque during recursive lowering.
    # Semantic layout follows by-value storage and stops at pointers.
    with report_layout_errors(node):
        field_layouts = tuple(get_type_layout(field) for field in pc_type._field_types)
        layout = byte_aggregate_layout(pc_type._field_names, field_layouts, config)
    offsets = layout.field_offsets
    members = [(index, field, field_layouts[index]) for index, field in visible]
    natural_alignment = max((member.alignment for _, _, member in members), default=1)
    packed = (layout.alignment < natural_alignment or any(
        offsets[index] % member.alignment for index, _, member in members))
    elements = []
    end = 0
    for index, field, member in members:
        offset = offsets[index]
        implicit_offset = end if packed else align_up(end, member.alignment)
        if implicit_offset != offset:
            elements.append(ir.ArrayType(ir.IntType(8), offset - end))
        paths[index] = (len(elements),)
        elements.append(field)
        end = offset + member.allocation_size

    if packed and layout.alignment > 1:
        payload = ir.LiteralStructType(elements, packed=True)
        elements = [payload]
        paths = [(0, *path) if path is not None else None for path in paths]
        natural_alignment = 1
        packed = False
    if not packed and layout.alignment > natural_alignment:
        carrier = get_target_layout().alignment_carrier(layout.alignment, node)
        elements.append(ir.ArrayType(carrier, 0))
    return RecordStorage(tuple(elements), packed, tuple(paths))


def materialize_record(definition, context, node):
    """Declare a record before lowering its possibly recursive fields."""
    from ..session import CompileSession
    with CompileSession.current().type_graph._lock, _materialization_lock:
        state = getattr(context, '_pythoc_records', None)
        if state is None:
            state = context._pythoc_records = (weakref.WeakKeyDictionary(), {})
        records, representations = state
        if definition in records:
            return records[definition]
        name, suffix = definition.name, 0
        while name in context.identified_types:
            suffix += 1
            name = f'{definition.name}.{suffix}'
        record = context.get_identified_type(name)
        records[definition] = record
        try:
            pc_type = definition.representative
            fields = tuple(field.get_llvm_type(context) for field in definition.field_types)
            storage = record_storage(fields, pc_type, node)
            key = definition.name, storage
            existing = representations.get(key)
            if existing is not None:
                records[definition] = existing
                return existing
            record.packed = storage.packed
            record.set_body(*storage.elements)
            record._pythoc_field_paths = storage.field_paths
            representations[key] = record
        except BaseException:
            del records[definition]
            raise
        return record
