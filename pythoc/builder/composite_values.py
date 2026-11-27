"""Materialize typed composite values in LLVM storage."""

from llvmlite import ir

from ..ir_helpers import is_link_time_constant
from ..logger import logger
from .constant_storage import ConstantImageError, union_storage_constant


def aggregate_value(builder, pc_type, values, node, context):
    storage = pc_type.get_llvm_type(context)
    paths = getattr(storage, '_pythoc_field_paths', None)
    if paths is None:
        paths = tuple((index,) for index in range(len(storage.elements)))
    if len(values) != len(paths):
        logger.error('Aggregate initializer field count mismatch',
                     node=node, exc_type=TypeError)
    fields = {}
    for index, (value, path) in enumerate(zip(values, paths)):
        if path is None:
            continue
        field_type = storage
        for slot in path:
            field_type = field_type.elements[slot]
        if value.type != field_type:
            logger.error(
                f'Aggregate initializer field {index} type mismatch: '
                f'expected {field_type}, got {value.type}',
                node=node, exc_type=TypeError)
        fields[path] = value

    if all(is_link_time_constant(value) for value in fields.values()):
        def constant_for(typ, path):
            if path in fields:
                return fields[path]
            if isinstance(typ, (ir.BaseStructType, ir.ArrayType)):
                return ir.Constant(typ, [constant_for(child, (*path, index))
                                         for index, child in enumerate(typ.elements)])
            return ir.Constant(typ, None)

        return constant_for(storage, ())
    result = ir.Constant(storage, None)
    for path, value in fields.items():
        result = builder.insert_value(result, value, path)
    return result


def union_value(builder, pc_type, member, name, node, context, constant_only):
    storage = pc_type.get_llvm_type(context)
    if is_link_time_constant(member):
        try:
            return union_storage_constant(storage, member, context)
        except ConstantImageError as error:
            if constant_only:
                logger.error(f"Union initializer for {name!r}: {error}",
                             node=node, exc_type=ValueError)

    place = builder.alloca_in_entry(storage, 'union_initializer')
    builder.store(ir.Constant(storage, None), place)
    member_place = builder.bitcast(place, member.type.as_pointer())
    builder.store(member, member_place)
    return builder.load(place)


def vector_value(builder, pc_type, lanes, context):
    vector = pc_type.get_llvm_type(context)
    if all(isinstance(lane, ir.Constant) for lane in lanes):
        values = list(lanes)
        return ir.Constant(vector, values * vector.count if len(values) == 1 else values)
    if len(lanes) == 1:
        scalar = builder.insert_element(ir.Constant(vector, ir.Undefined),
                                        lanes[0], ir.Constant(ir.IntType(32), 0))
        mask = ir.Constant(ir.VectorType(ir.IntType(32), vector.count),
                           [0] * vector.count)
        return builder.shuffle_vector(scalar, ir.Constant(vector, ir.Undefined), mask)
    result = ir.Constant(vector, ir.Undefined)
    for index, lane in enumerate(lanes):
        result = builder.insert_element(result, lane, ir.Constant(ir.IntType(32), index))
    return result
