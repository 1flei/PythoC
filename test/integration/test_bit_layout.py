"""End-to-end decorator-configured bit aggregate tests."""

import unittest
from pathlib import Path
import shlex
import subprocess
import tempfile

from pythoc import (
    compile,
    i1,
    i31,
    i32,
    i64,
    sizeof,
    struct,
    u3,
    u5,
    u6,
    u8,
    union,
)
from pythoc import array, extern, i3, i8, ptr, static, u33, u64, volatile
from pythoc.utils.cc_utils import compile_c_to_object, find_available_cc
from pythoc.utils.link_utils import (
    get_executable_extension, get_shared_lib_extension, link_files,
)
from pythoc.target_layout import default_target_triple
from test.utils.c_bitfield_profiles import c_bitfield_options


_temp = tempfile.TemporaryDirectory(prefix='pc-bitfield-')
_source = Path(__file__).parents[1] / 'c_oracles' / 'bit_layout.c'
_build = Path(__file__).parents[2] / 'build/test/bit_layout'
_build.mkdir(parents=True, exist_ok=True)
_object = compile_c_to_object(str(_source), str(_build / 'oracle.o'))
_library = link_files([_object], str(_build / (
    'oracle' + get_shared_lib_extension())), shared=True, link_objects=[])
_c_options = c_bitfield_options(default_target_triple())


@struct(storage_unit=16)
class AllShort:
    a: u3
    b: u5
    c: u6


@struct(**_c_options['mixed'])
class MixedUnits:
    a: u3
    b: u5
    c: u6


@struct(**_c_options['boundary'])
class UnitBoundary:
    a: u3
    b: u3


BoundaryRecord = struct[('prefix', u8), ('value', UnitBoundary), ('suffix', u8)]


@struct(storage_unit=64)
class LargeBits:
    a: u64
    b: u64
    c: u64


@struct(storage_unit=8, cross_unit=True, bit_order='msb0')
class MsbCrossing:
    a: u6
    b: u6


@struct(storage_unit=8, cross_unit=True)
class Crossing:
    a: u6
    b: u6


@struct(storage_unit=8)
class SignedBits:
    a: i3
    b: u5


@union(storage_unit=32, cross_unit=True, abi_alignment=8)
class WideUnion:
    value: u33


Container = struct[('tag', i8), ('data', array[WideUnion, 2]), ('tail', u64)]


@extern(lib=_library, name='pc_bit_short')
def c_short(value: AllShort) -> AllShort:
    pass


@extern(lib=_library, name='pc_bit_mixed')
def c_mixed(value: MixedUnits) -> MixedUnits:
    pass


@extern(lib=_library, name='pc_bit_boundary')
def c_boundary(value: UnitBoundary) -> UnitBoundary:
    pass


@extern(lib=_library, name='pc_bit_large')
def c_large(value: LargeBits) -> LargeBits:
    pass


@compile
def c_round_trip() -> i64:
    small: AllShort = (5, 17, 33)
    mixed: MixedUnits = (5, 17, 33)
    boundary: UnitBoundary = (5, 3)
    large: LargeBits = (100, 200, 300)
    a = c_short(small)
    b = c_mixed(mixed)
    c = c_boundary(boundary)
    d = c_large(large)
    return (i64(a.a) + i64(a.b) + i64(a.c)
            + i64(b.a) + i64(b.b) + i64(b.c)
            + i64(c.a) + i64(c.b)
            + i64(d.a) + i64(d.b) + i64(d.c))


@compile
def initialized_short() -> AllShort:
    value: AllShort = (5, 17, 33)
    return value


@compile
def initialized_mixed() -> MixedUnits:
    value: MixedUnits = (5, 17, 33)
    return value


@compile
def initialized_boundary() -> UnitBoundary:
    value: UnitBoundary = (5, 3)
    return value


@compile
def boundary_array_stride() -> i64:
    values: array[UnitBoundary, 2]
    return i64(u64(ptr(values[1])) - u64(ptr(values[0])))


@compile
def runtime_initialization(x: u6) -> i64:
    value: AllShort = (u3(x), u5(x), x)
    other: AllShort = {'a': u3(x), 'b': u5(x), 'c': x}
    source: struct[u3, u5, u6] = (u3(x), u5(x), x)
    converted: AllShort = source
    return (i64(value.a) + i64(value.b) + i64(value.c)
            + i64(other.a) + i64(other.b) + i64(other.c)
            + i64(converted.a) + i64(converted.b) + i64(converted.c))


@compile
def static_initialization() -> i64:
    value: static[AllShort] = (5, 17, 33)
    return i64(value.a) + i64(value.b) + i64(value.c)


@compile
def signed_round_trip() -> i64:
    value: SignedBits = (-3, 17)
    value.a += i3(1)
    return i64(value.a) * 100 + i64(value.b)


@compile
def update_neighbor(value: ptr[SignedBits]) -> i3:
    value.b = u5(31)
    return i3(1)


@compile
def compound_assignment_preserves_rhs_neighbor_change() -> i64:
    value: SignedBits = (-3, 17)
    value.a += update_neighbor(ptr(value))
    return i64(value.a) * 100 + i64(value.b)


@compile
def volatile_field_round_trip() -> i64:
    value: AllShort = (5, 17, 33)
    view: ptr[volatile[AllShort]] = ptr(value)
    view.a = u3(3)
    view[0][1] = u5(7)
    return i64(view.a) + i64(view[0][1]) + i64(value.c)


@compile
def crossing_msb_round_trip() -> i64:
    value: MsbCrossing = (42, 21)
    value.b += u6(3)
    return i64(value.a) * 100 + i64(value.b)


@compile
def nested_arrays_preserve_neighbors() -> i64:
    value: Container
    value.tag = i8(23)
    value.tail = u64(0x123456789ABCDEF0)
    value.data[0].value = u33(0x100000005)
    value.data[1].value = u33(0x100000009)
    if value.tag != 23 or value.data[0].value != u33(0x100000005):
        return -1
    if value.data[1].value != u33(0x100000009):
        return -2
    return i64(value.tail)


@compile
def exhaustive_crossing_storage() -> i64:
    a: i64 = 0
    while a < 64:
        b: i64 = 0
        while b < 64:
            little: Crossing = (u6(a), u6(b))
            big: MsbCrossing = (u6(a), u6(b))
            lp: ptr[u8] = ptr[u8](ptr(little))
            bp: ptr[u8] = ptr[u8](ptr(big))
            if i64(lp[0]) != (a | ((b & 3) << 6)):
                return 1
            if i64(lp[1]) != b >> 2:
                return 2
            if i64(bp[0]) != ((a << 2) | (b >> 4)):
                return 3
            if i64(bp[1]) != ((b & 15) << 4):
                return 4
            if i64(little.a) != a or i64(little.b) != b:
                return 5
            if i64(big.a) != a or i64(big.b) != b:
                return 6
            b += 1
        a += 1
    return 0


@struct(alignment=1, storage_unit=32, abi_alignment=32)
class PackedWord:
    a: i1
    b: i31
    c: i32


@union(storage_unit=8, bit_order="lsb0")
class LittleBitOrder:
    field: u3
    raw: u8


@union(storage_unit=8, bit_order="msb0")
class BigBitOrder:
    field: u3
    raw: u8


@compile
def packed_round_trip() -> i64:
    value: PackedWord = PackedWord()
    value.a = i1(1)
    value.b = i31(100)
    value.b += i31(23)
    value.c = i32(1000)
    return i64(value.a) + i64(value.b) + i64(value.c)


@compile
def packed_size() -> i64:
    return sizeof(PackedWord)


@compile
def crossing_round_trip() -> i64:
    value: Crossing = Crossing()
    value.a = u6(42)
    value.b = u6(21)
    return i64(value.a) * i64(100) + i64(value.b)


@compile
def lsb0_storage() -> i64:
    value: LittleBitOrder = LittleBitOrder()
    value.raw = u8(0)
    value.field = u3(5)
    return i64(value.raw)


@compile
def msb0_storage() -> i64:
    value: BigBitOrder = BigBitOrder()
    value.raw = u8(0)
    value.field = u3(5)
    return i64(value.raw)


class TestBitLayoutIntegration(unittest.TestCase):
    def test_c_by_value_round_trip(self):
        self.assertEqual(c_round_trip(), 739)

    def test_all_crossing_values_match_independent_byte_formula(self):
        self.assertEqual(exhaustive_crossing_storage(), 0)

    def test_native_c_layout_and_python_entry_bytes(self):
        executable = str(Path(_temp.name) / (
            'oracle' + get_executable_extension()))
        subprocess.run(shlex.split(find_available_cc()) + [
            '-std=c11', str(_source), '-o', executable], check=True,
            capture_output=True, text=True)
        actual = subprocess.check_output([executable], text=True).splitlines()
        expected = []
        for name, ty, function in (
            ('all_short', AllShort._struct_type, initialized_short),
            ('mixed', MixedUnits._struct_type, initialized_mixed),
            ('boundary', UnitBoundary._struct_type, initialized_boundary),
        ):
            value = function()
            layout = ty.get_layout()
            raw = value._value
            self.assertIsInstance(raw, bytes)
            self.assertEqual(len(raw), layout.allocation_size)
            expected.append(f'{name} {layout.allocation_size} '
                            f'{layout.alignment} {raw.hex()}')
        nested = BoundaryRecord.get_layout()
        expected.append('boundary_layout {} {} {} {} {}'.format(
            array[UnitBoundary, 2].get_size_bytes(), nested.allocation_size,
            nested.alignment, *nested.field_offsets[1:],
        ))
        self.assertEqual(actual, expected)
        self.assertEqual(boundary_array_stride(),
                         UnitBoundary._struct_type.get_size_bytes())

    def test_runtime_tuple_dict_and_struct_initialization(self):
        self.assertEqual(runtime_initialization(21), 141)

    def test_static_bitfield_initializer(self):
        self.assertEqual(static_initialization(), 55)

    def test_signed_width_and_extension(self):
        self.assertEqual(signed_round_trip(), -183)

    def test_compound_assignment_preserves_rhs_neighbor_change(self):
        self.assertEqual(compound_assignment_preserves_rhs_neighbor_change(), -169)

    def test_volatile_attribute_and_subscript(self):
        self.assertEqual(volatile_field_round_trip(), 43)

    def test_msb0_cross_unit_preserves_other_field(self):
        self.assertEqual(crossing_msb_round_trip(), 4224)

    def test_unaligned_union_arrays_preserve_neighbors(self):
        self.assertEqual(nested_arrays_preserve_neighbors(),
                         0x123456789ABCDEF0)
    def test_packed_fields_and_augmented_assignment(self):
        self.assertEqual(packed_round_trip(), 1122)
        self.assertEqual(packed_size(), 8)

    def test_field_can_cross_storage_unit(self):
        self.assertEqual(crossing_round_trip(), 4221)

    def test_lsb0_bit_order(self):
        self.assertEqual(lsb0_storage(), 0b00000101)

    def test_msb0_bit_order(self):
        self.assertEqual(msb0_storage(), 0b10100000)

if __name__ == "__main__":
    unittest.main()
