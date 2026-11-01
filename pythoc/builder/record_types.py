"""Materialize validated records using target storage representations."""

import threading
import weakref

from llvmlite import ir


_materialization_lock = threading.RLock()


def storage_type(pc_type, context):
    """Describe storage without following a pointer's semantic pointee."""
    from ..type_converter import get_base_type
    from ..type_graph import record_view, named_record

    base = get_base_type(pc_type)
    if getattr(base, '_is_pointer', False):
        return ir.PointerType(ir.IntType(8))
    dimensions = getattr(base, 'dimensions', None)
    if dimensions is not None:
        result = storage_type(base.element_type, context)
        for count in reversed(dimensions):
            result = ir.ArrayType(result, count)
        return result
    view = record_view(base)
    if (view is not None and named_record(view) is None
            and view._layout_config.is_natural):
        view._ensure_field_types_resolved()
        fields = [storage_type(field, context) for field in view._field_types]
        return ir.LiteralStructType([field for field in fields if field is not None])
    return base.get_llvm_type(context)


def storage_value(builder, value, target):
    """Adapt a typed value to its field storage representation."""
    if value.type == target:
        return value
    if isinstance(value.type, ir.PointerType) and isinstance(target, ir.PointerType):
        return (value.bitcast(target) if builder is None
                else builder.bitcast(value, target))
    source = value.type
    if isinstance(target, ir.ArrayType) and isinstance(source, ir.ArrayType):
        if target.count != source.count:
            raise TypeError('Storage conversion cannot change array length')
        elements = (target.element,) * target.count
    elif (isinstance(target, ir.BaseStructType)
          and isinstance(source, ir.BaseStructType)):
        if len(target.elements) != len(source.elements):
            raise TypeError('Storage conversion cannot change field count')
        elements = target.elements
    else:
        raise TypeError(f'Invalid storage conversion: {source} to {target}')
    if builder is None:
        if value.constant is None or value.constant is ir.Undefined:
            return ir.Constant(target, value.constant)
        fields = [storage_value(None, field, field_type)
                  for field, field_type in zip(value.constant, elements)]
        return ir.Constant(target, fields)
    result = ir.Constant(target, ir.Undefined)
    for index, field_type in enumerate(elements):
        field = builder.extract_value(value, index)
        field = storage_value(builder, field, field_type)
        result = builder.insert_value(result, field, index)
    return result


def materialize_record(definition, context):
    """Intern physical storage while leaving PythoC identities independent.

    Pointer fields are addresses. Their pointee identity and operations are
    supplied by PythoC field types, including after forward completion.
    """
    with _materialization_lock:
        state = getattr(context, '_pythoc_records', None)
        if state is None:
            state = context._pythoc_records = (weakref.WeakKeyDictionary(), {})
        records, representations = state
        if definition in records:
            return records[definition]
        fields = tuple(storage_type(field, context)
                       for field in definition.field_types)
        fields = tuple(field for field in fields if field is not None)
        key = (definition.name, fields)
        record = representations.get(key)
        if record is None:
            name, suffix = definition.name, 0
            while name in context.identified_types:
                suffix += 1
                name = f'{definition.name}.{suffix}'
            record = context.get_identified_type(name)
            record.set_body(*fields)
            representations[key] = record
        records[definition] = record
        return record
