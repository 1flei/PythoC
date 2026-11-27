"""Construct constant target storage with bytes and whole-word relocations."""

from llvmlite import ir

from ..llvm_extensions import format_for_type
from ..target_layout import get_target_layout


class ConstantImageError(ValueError):
    pass


def _write_constant(value, offset, image, relocations, layout, order, context):
    typ = value.type
    if isinstance(value, (ir.GlobalValue, ir.FormattedConstant)):
        if not isinstance(typ, (ir.PointerType, ir.IntType)):
            raise ConstantImageError(f'unsupported relocation type {typ}')
        relocations[offset] = value
        return
    if not isinstance(value, ir.Constant):
        raise ConstantImageError('the initializer contains a runtime value')
    size = layout.primitive(typ).allocation_size
    payload = value.constant
    if payload is None:
        return
    if payload is ir.Undefined:
        raise ConstantImageError('the constant contains an undefined value')
    if isinstance(typ, ir.IntType) and isinstance(payload, int):
        bits = payload & ((1 << typ.width) - 1)
        image[offset:offset + size] = bits.to_bytes(size, order)
    elif (float_format := format_for_type(typ)) is not None:
        width = float_format.bits // 8
        image[offset:offset + width] = float_format.encode(payload).bits.to_bytes(width, order)
    elif isinstance(typ, ir.ArrayType) and isinstance(payload, (bytes, bytearray)):
        image[offset:offset + len(payload)] = payload
    elif isinstance(typ, (ir.ArrayType, ir.VectorType)):
        element_layout = layout.primitive(typ.element)
        if (isinstance(typ, ir.VectorType) and isinstance(typ.element, ir.IntType)
                and typ.element.width != element_layout.stride * 8):
            raise ConstantImageError('the constant uses packed integer vector lanes')
        for index, element in enumerate(payload):
            _write_constant(element, offset + index * element_layout.stride,
                            image, relocations, layout, order, context)
    elif isinstance(typ, ir.BaseStructType):
        for index, element in enumerate(payload):
            field_offset = typ.get_element_offset(layout.target_data, index, context)
            _write_constant(element, offset + field_offset, image, relocations,
                            layout, order, context)
    else:
        raise ConstantImageError(f'unsupported constant storage type {typ}')


def _carrier_value(typ, image, order):
    if isinstance(typ, ir.IntType):
        return ir.Constant(typ, int.from_bytes(image, order))
    # TargetLayout uses a byte vector as a carrier for larger alignments.
    return ir.Constant(typ, [ir.Constant(typ.element, value) for value in image])


def union_storage_constant(storage_type, member, context):
    """Place one alternative at its union offset and zero the remaining bytes."""
    layout = get_target_layout()
    order = 'big' if str(layout.target_data).startswith('E') else 'little'
    payload_type = storage_type
    if isinstance(storage_type, ir.LiteralStructType):
        payload_type, alignment = storage_type.elements
    carrier = payload_type.element
    stride = layout.primitive(carrier).allocation_size
    size = stride * payload_type.count
    if layout.primitive(member.type).allocation_size > size:
        raise ConstantImageError('the active member is larger than union storage')
    image = bytearray(size)
    relocations = {}
    _write_constant(member, 0, image, relocations, layout, order, context)
    values = [
        _carrier_value(carrier, image[offset:offset + stride], order)
        for offset in range(0, size, stride)
    ]
    for offset, value in relocations.items():
        value_layout = layout.primitive(value.type)
        if (not isinstance(carrier, ir.IntType) or offset % stride
                or value_layout.allocation_size != stride
                or carrier.width != stride * 8):
            raise ConstantImageError(
                f'a relocation at offset {offset} does not fit one union storage word')
        if isinstance(value.type, ir.PointerType):
            value = value.ptrtoint(carrier)
        elif value.type != carrier:
            raise ConstantImageError(f'relocation type {value.type} does not match {carrier}')
        values[offset // stride] = value
    payload = ir.Constant(payload_type, values)
    if payload_type is storage_type:
        return payload
    return ir.Constant(storage_type, [payload, ir.Constant(alignment, None)])
