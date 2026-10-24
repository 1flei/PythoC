"""One invalid bit-field program per isolated compiler process."""

import sys

from pythoc import compile, const, i64, offsetof, ptr, struct, u3, u8, union


@struct(storage_unit=8)
class Bits:
    a: u3
    b: u3


@union(storage_unit=8)
class Overlap:
    a: u3
    b: u8


mode = sys.argv[1]

if mode == 'address_attribute':
    @compile
    def invalid() -> i64:
        value: Bits = (1, 2)
        return i64(ptr(value.a))
elif mode == 'address_subscript':
    @compile
    def invalid() -> i64:
        value: Bits = (1, 2)
        return i64(ptr(value[0]))
elif mode == 'offsetof_struct':
    @compile
    def invalid() -> i64:
        return offsetof(Bits, 'a')
elif mode == 'offsetof_union':
    @compile
    def invalid() -> i64:
        return offsetof(Overlap, 'b')
elif mode == 'const_attribute':
    @compile
    def invalid() -> i64:
        value: const[Bits] = (1, 2)
        value.a = u3(3)
        return 0
elif mode == 'const_subscript':
    @compile
    def invalid() -> i64:
        value: Overlap = Overlap()
        frozen: ptr[const[Overlap]] = ptr(value)
        frozen[0][0] = u3(3)
        return 0
else:
    raise ValueError(mode)

invalid()
