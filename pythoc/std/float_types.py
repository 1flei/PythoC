"""Extended binary floats sharing the LLVM floating-point boundary."""

from ..builtin_entities.base import BuiltinType
from ..llvm_extensions import F16, BF16, F128


class f16(BuiltinType):
    _llvm_type = F16.llvm_type
    _size_bytes = F16.bits // 8
    _is_signed = True
    _is_float = True

    @classmethod
    def get_name(cls) -> str:
        return 'f16'


class bf16(BuiltinType):
    _llvm_type = BF16.llvm_type
    _size_bytes = BF16.bits // 8
    _is_signed = True
    _is_float = True

    @classmethod
    def get_name(cls) -> str:
        return 'bf16'


class f128(BuiltinType):
    _llvm_type = F128.llvm_type
    _size_bytes = F128.bits // 8
    _is_signed = True
    _is_float = True

    @classmethod
    def get_name(cls) -> str:
        return 'f128'
