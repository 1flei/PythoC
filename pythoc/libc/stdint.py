"""
Standard integer typedefs from <stdint.h> / <inttypes.h>.
"""

from ..builtin_entities import i8, i16, i32, i64, u8, u16, u32, u64
from ..forward_ref import mark_type_defined

__all__ = [
    'int8_t', 'int16_t', 'int32_t', 'int64_t',
    'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t',
    'intptr_t', 'uintptr_t', 'intmax_t', 'uintmax_t',
]

int8_t = i8
int16_t = i16
int32_t = i32
int64_t = i64
uint8_t = u8
uint16_t = u16
uint32_t = u32
uint64_t = u64
intptr_t = i64
uintptr_t = u64
intmax_t = i64
uintmax_t = u64

for _name in __all__:
    mark_type_defined(_name, globals()[_name])
