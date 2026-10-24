"""Storage and non-addressable field projections for bit aggregates."""

from dataclasses import dataclass

from llvmlite import ir

from ..ir_helpers import check_const_store, safe_load, safe_store
from ..target_layout import field_segments, get_target_layout
from ..valueref import ValueRef, ensure_ir


def get_bit_storage_type(layout):
    target = get_target_layout()
    carrier = target.alignment_carrier(layout.alignment)
    size = target.primitive(carrier).allocation_size
    if layout.allocation_size % size:
        raise ValueError('bit storage is not divisible by its alignment carrier')
    return ir.ArrayType(carrier, layout.allocation_size // size)


def get_bit_storage_ctypes(layout, name):
    import ctypes

    carriers = (ctypes.c_uint8, ctypes.c_uint16, ctypes.c_uint32,
                ctypes.c_uint64, ctypes.c_longdouble)
    carrier = next((candidate for candidate in carriers
                    if ctypes.sizeof(candidate) == layout.alignment
                    and ctypes.alignment(candidate) == layout.alignment), None)
    if carrier is None:
        raise ValueError(
            f'ctypes has no carrier for {layout.alignment}-byte alignment')
    return type(name, (ctypes.Structure,), {
        '_fields_': [('_storage', carrier * (
            layout.allocation_size // ctypes.sizeof(carrier)))],
    })


def _resize(builder, value, target):
    if value.type.width < target.width:
        return builder.zext(value, target)
    if value.type.width > target.width:
        return builder.trunc(value, target)
    return value


def assemble_bit_fields(visitor, aggregate_type, values):
    """Pack constant and runtime initializers into the same field mapping."""
    from ..type_converter import strip_qualifiers
    from .union import UnionType

    if issubclass(getattr(aggregate_type, '_union_type', aggregate_type), UnionType):
        raise TypeError(
            'Initialize a bit-layout union by assigning its selected field')
    layout = aggregate_type.get_layout()
    llvm_type = aggregate_type.get_llvm_type(visitor.module.context)
    bit_order = aggregate_type._layout_config.bit_order
    if all(isinstance(value, ir.Constant) for value in values):
        data = bytearray(layout.allocation_size)
        byte_order = ('little' if str(get_target_layout().target_data)
                      .startswith('e') else 'big')
        for index, value in enumerate(values):
            unit = layout.field_storage_units_bits[index]
            segments = field_segments(
                layout.field_offsets_bits[index],
                layout.field_widths_bits[index], unit, bit_order)
            for start, bit, value_bit, width in segments:
                offset, size = start // 8, unit // 8
                word = int.from_bytes(data[offset:offset + size], byte_order)
                mask = (1 << width) - 1
                word |= ((int(value.constant) >> value_bit) & mask) << bit
                data[offset:offset + size] = word.to_bytes(size, byte_order)
        carrier = llvm_type.element
        size = get_target_layout().primitive(carrier).allocation_size
        words = []
        for offset in range(0, len(data), size):
            chunk = data[offset:offset + size]
            payload = (list(chunk) if isinstance(carrier, ir.VectorType)
                       else int.from_bytes(chunk, byte_order))
            words.append(ir.Constant(carrier, payload))
        return ir.Constant(llvm_type, words)

    pointer = visitor._create_alloca_in_entry(llvm_type, 'bit_initializer')
    visitor.builder.store(ir.Constant(llvm_type, None), pointer)
    base = ValueRef(kind='address', value=pointer,
                    type_hint=aggregate_type, address=pointer)
    for index, value in enumerate(values):
        field_type = strip_qualifiers(aggregate_type._field_types[index])
        field = access_bit_field(visitor, base, aggregate_type, index, field_type)
        field.store(visitor.builder, value)
    return visitor.builder.load(pointer)


@dataclass
class BitFieldStorage:
    visitor: object
    pointer: object
    llvm_type: object
    pc_type: object
    segments: tuple
    unit: int
    mutable: bool

    def _pointer(self, start):
        builder = self.visitor.builder
        byte_pointer = builder.bitcast(self.pointer, ir.IntType(8).as_pointer())
        address = builder.gep(
            byte_pointer, [ir.Constant(ir.IntType(64), start // 8)],
            inbounds=True,
        )
        return builder.bitcast(address, ir.IntType(self.unit).as_pointer())

    def load(self):
        builder = self.visitor.builder
        result = ir.Constant(self.llvm_type, 0)
        words = {}
        for start, bit, value_bit, width in self.segments:
            if start not in words:
                words[start] = safe_load(
                    builder, self._pointer(start), self.pc_type, align=1)
            word = words[start]
            part = builder.lshr(word, ir.Constant(word.type, bit))
            part = builder.and_(part, ir.Constant(word.type, (1 << width) - 1))
            part = _resize(builder, part, self.llvm_type)
            part = builder.shl(part, ir.Constant(self.llvm_type, value_bit))
            result = builder.or_(result, part)
        return result

    def store(self, value, *, node=None):
        if not self.mutable:
            raise ValueError('Cannot assign to a temporary bit field')
        check_const_store(self.pc_type, node)
        builder = self.visitor.builder
        word_type = ir.IntType(self.unit)
        masks = {}
        values = {}
        for start, bit, value_bit, width in self.segments:
            mask = (1 << width) - 1
            masks[start] = masks.get(start, 0) | (mask << bit)
            part = builder.lshr(value, ir.Constant(value.type, value_bit))
            part = builder.and_(part, ir.Constant(value.type, mask))
            part = _resize(builder, part, word_type)
            part = builder.shl(part, ir.Constant(word_type, bit))
            values[start] = builder.or_(
                values.get(start, ir.Constant(word_type, 0)), part)
        all_bits = (1 << self.unit) - 1
        for start, mask in masks.items():
            pointer = self._pointer(start)
            word = values[start]
            if mask != all_bits:
                old = safe_load(builder, pointer, self.pc_type, align=1)
                preserved = builder.and_(old, ir.Constant(word_type,
                                                         all_bits ^ mask))
                word = builder.or_(preserved, word)
            safe_store(builder, word, pointer, self.pc_type, align=1, node=node)


def access_bit_field(visitor, base, aggregate_type, field_index, field_type,
                     *, var_name=None, linear_path=None):
    layout = aggregate_type.get_layout()
    if base.is_pointer_typed():
        pointer, mutable = ensure_ir(base), True
    elif base.has_place():
        pointer, mutable = base.require_place(), True
    else:
        value = ensure_ir(base)
        pointer = visitor._create_alloca_in_entry(value.type, 'bit_value')
        visitor.builder.store(value, pointer)
        mutable = False
    unit = layout.field_storage_units_bits[field_index]
    storage = BitFieldStorage(
        visitor, pointer, field_type.get_llvm_type(visitor.module.context),
        field_type,
        field_segments(layout.field_offsets_bits[field_index],
                       layout.field_widths_bits[field_index], unit,
                       aggregate_type._layout_config.bit_order),
        unit, mutable,
    )
    return ValueRef(kind='value', value=None, type_hint=field_type,
                    var_name=var_name, linear_path=linear_path, storage=storage)
