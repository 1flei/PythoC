"""C aggregate classification for System V and Windows x64."""

from enum import IntEnum

from llvmlite import ir

from .base import ABIInfo, CoercedType, PassingKind
from ...llvm_extensions import is_float_type


class _EightbyteClass(IntEnum):
    """Register classes in merge priority order."""

    NO_CLASS = 0
    SSEUP = 1
    SSE = 2
    INTEGER = 3


class X86_64ABI(ABIInfo):
    def __init__(
        self,
        max_register_size: int = 16,
        use_byval_for_indirect_args: bool = True,
        pow2_register_sizes: bool = False,
        int64_vectors_in_sse: bool = True,
    ):
        self.max_register_size = max_register_size
        self._use_byval_for_indirect_args = use_byval_for_indirect_args
        self._pow2_register_sizes = pow2_register_sizes
        self._int64_vectors_in_sse = int64_vectors_in_sse

    def uses_byval_for_indirect_args(self) -> bool:
        return self._use_byval_for_indirect_args

    def _allocate_argument_registers(self, arguments, result):
        if self._pow2_register_sizes:
            yield from arguments
            return

        gp = 6 - int(result.is_indirect)
        sse = 8
        for argument in arguments:
            if argument.is_indirect:
                yield argument
                continue
            integer_regs, sse_regs = self._required_registers(argument.abi_type)
            if gp >= integer_regs and sse >= sse_regs:
                gp -= integer_regs
                sse -= sse_regs
            elif (argument.needs_coercion
                  and (sse_regs or self.get_type_size(argument.original_type) > 8)):
                # If an aggregate would only partially fit, all of it goes on the stack.
                argument = CoercedType(PassingKind.INDIRECT,
                                       original_type=argument.original_type)
            yield argument

    def _required_registers(self, typ):
        if self.is_struct_type(typ):
            # A coerced two-eightbyte aggregate has one carrier per eightbyte.
            parts = [self._required_registers(part) for part in typ.elements]
            return sum(gp for gp, _ in parts), sum(sse for _, sse in parts)
        if isinstance(typ, ir.IntType):
            return (typ.width + 63) // 64, 0
        if isinstance(typ, ir.PointerType):
            return 1, 0
        if is_float_type(typ) or isinstance(typ, ir.VectorType):
            return 0, 1
        return 0, 0

    def _classify(self, pc_type, llvm_type, module_context, *, is_return):
        if not self.is_aggregate_type(llvm_type):
            return CoercedType(PassingKind.DIRECT, original_type=llvm_type,
                               is_return=is_return)

        size = self.get_type_size(llvm_type)
        if size == 0:
            return CoercedType(PassingKind.DIRECT, ir.VoidType() if is_return else None,
                               llvm_type, is_return)
        indirect = CoercedType(PassingKind.INDIRECT, original_type=llvm_type,
                               is_return=is_return)
        if size > self.max_register_size:
            return indirect
        if self._pow2_register_sizes:
            if size not in (1, 2, 4, 8):
                return indirect
            return CoercedType(PassingKind.COERCE, ir.IntType(size * 8),
                               llvm_type, is_return)

        lanes = [[], []]
        classes = [_EightbyteClass.NO_CLASS, _EightbyteClass.NO_CLASS]
        for offset, field, field_size in self._abi_fields(pc_type, llvm_type, module_context):
            if offset % self.get_type_alignment(field):
                return indirect
            if self.is_aggregate_type(field):
                continue
            field_class = self._field_class(field, field_size)
            if field_class is None:
                return indirect
            first_lane = offset // 8
            for lane in range(first_lane, (offset + field_size - 1) // 8 + 1):
                lanes[lane].append((offset - lane * 8, field, field_size))
                lane_class = field_class
                if field_class == _EightbyteClass.SSE and lane != first_lane:
                    lane_class = _EightbyteClass.SSEUP
                classes[lane] = max(classes[lane], lane_class)

        lo_class, hi_class = classes
        if lo_class == _EightbyteClass.SSE and hi_class == _EightbyteClass.SSEUP:
            # Both eightbytes occupy the same XMM register.
            carrier = next(field for _, field, width in lanes[0] if width == 16)
            if getattr(carrier, 'element', None) == ir.IntType(128):
                carrier = ir.VectorType(ir.IntType(64), 2)
            return CoercedType(PassingKind.COERCE, carrier, llvm_type, is_return)
        if hi_class == _EightbyteClass.SSEUP:
            hi_class = _EightbyteClass.SSE

        lo = self._lane_type(lo_class, lanes[0], has_next=bool(lanes[1]))
        hi = self._lane_type(hi_class, lanes[1])
        if lo is None:
            return CoercedType(PassingKind.DIRECT, original_type=llvm_type,
                               is_return=is_return)
        coerced = lo if hi is None else ir.LiteralStructType([lo, hi])
        return CoercedType(PassingKind.COERCE, coerced, llvm_type, is_return)

    def _field_class(self, field, size):
        if isinstance(field, ir.VectorType) and size >= 8:
            if size == 8:
                if field.element == ir.DoubleType():
                    return None
                if field.element == ir.IntType(64) and not self._int64_vectors_in_sse:
                    return _EightbyteClass.INTEGER
            return _EightbyteClass.SSE
        if is_float_type(field):
            return _EightbyteClass.SSE
        return _EightbyteClass.INTEGER

    @staticmethod
    def _lane_type(lane_class, fields, *, has_next=False):
        if lane_class == _EightbyteClass.NO_CLASS:
            return None
        if lane_class == _EightbyteClass.INTEGER:
            size = 8 if has_next else max(offset + used for offset, _, used in fields)
            return ir.IntType(next(width for width in (8, 16, 32, 64) if width >= size * 8))
        if any(width >= 8 for _, _, width in fields):
            return ir.DoubleType()
        if all(width == 2 for _, _, width in fields):
            slots = 4 if has_next else max(offset // 2 + 1 for offset, _, _ in fields)
            lane = fields[0][1]
            return lane if slots == 1 else ir.VectorType(lane, 2 if slots == 2 else 4)
        if has_next or any(offset for offset, _, _ in fields):
            return ir.VectorType(ir.FloatType(), 2)
        return ir.FloatType()
