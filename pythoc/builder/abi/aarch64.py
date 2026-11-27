"""C aggregate classification for the AArch64 procedure call standard."""

from llvmlite import ir

from .base import ABIInfo, CoercedType, PassingKind
from ...llvm_extensions import format_for_type


class AArch64ABI(ABIInfo):
    def __init__(self, *, use_adjusted_argument_alignment=False):
        self._use_adjusted_argument_alignment = use_adjusted_argument_alignment

    def uses_byval_for_indirect_args(self) -> bool:
        return False

    def _classify(self, pc_type, llvm_type, module_context, *, is_return):
        if not self.is_aggregate_type(llvm_type):
            return CoercedType(PassingKind.DIRECT, original_type=llvm_type,
                               is_return=is_return)
        size = self.get_type_size(llvm_type)
        if size == 0:
            return CoercedType(PassingKind.DIRECT, ir.VoidType() if is_return else None,
                               llvm_type, is_return)
        homogeneous = self._homogeneous_carrier(pc_type, llvm_type, module_context, size)
        if homogeneous is not None:
            return CoercedType(PassingKind.COERCE, homogeneous, llvm_type, is_return)
        if size > 16:
            return CoercedType(PassingKind.INDIRECT, original_type=llvm_type,
                               is_return=is_return)
        if size <= 8:
            carrier = ir.IntType(8 * size if is_return else 64)
        elif self._register_alignment(llvm_type, pc_type, is_return) >= 16:
            carrier = ir.IntType(128)
        else:
            carrier = ir.ArrayType(ir.IntType(64), 2)
        return CoercedType(PassingKind.COERCE, carrier, llvm_type, is_return)

    def _homogeneous_carrier(self, pc_type, llvm_type, module_context, size):
        if size > 64:
            return None
        fields = set()
        for offset, field, width in self._abi_fields(pc_type, llvm_type, module_context):
            if self.is_aggregate_type(field):
                continue
            if isinstance(field, ir.VectorType) and width in (8, 16):
                # Short vectors of the same width are homogeneous across element types.
                field = ir.VectorType(ir.IntType(32), width // 4)
            else:
                float_format = format_for_type(field)
                if float_format is None:
                    return None
                if float_format.bits == 16:
                    field = ir.HalfType()
            fields.add((offset, field))
        if not 1 <= len(fields) <= 4:
            return None
        first = next(iter(fields))[1]
        width = self.get_type_size(first)
        if size != width * len(fields):
            return None
        if fields != {(i * width, first) for i in range(len(fields))}:
            return None
        return first if len(fields) == 1 else ir.ArrayType(first, len(fields))

    def _register_alignment(self, llvm_type, pc_type, is_return):
        if not is_return and not self._use_adjusted_argument_alignment:
            from ...target_layout import get_type_layout
            return get_type_layout(pc_type).member_alignment
        return self.get_type_alignment(llvm_type)
