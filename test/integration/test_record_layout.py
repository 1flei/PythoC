"""Exercise record layout across PythoC, LLVM, Python and the native C ABI."""

from contextlib import redirect_stderr
import ctypes
from io import StringIO
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from pythoc import (
    CompileSession, array, atomic_load, compile, extern, f32, f64, flush_all_pending_outputs,
    func, i32, offsetof, ptr, simd, simd_load, sizeof, static, struct,
    u8, u32, u64, union, volatile,
)
from pythoc.build.output_manager import get_output_manager
from pythoc.logger import logger
from pythoc.utils.cc_utils import compile_c_to_object
from pythoc.utils.link_utils import get_shared_lib_extension, link_files


_source = Path(__file__).parents[1] / 'c_oracles' / 'record_layout.c'
_build = Path(__file__).parents[2] / 'build/test/record_layout'
_build.mkdir(parents=True, exist_ok=True)
_clang_abi = sys.platform == 'win32' or shutil.which('clang') is not None
_cc = 'clang' if _clang_abi and sys.platform != 'win32' else None
_oracle = 'oracle-clang' if _clang_abi else 'oracle'
_object = compile_c_to_object(str(_source), str(_build / (_oracle + '.o')), cc=_cc)
_library = link_files(
    [_object], str(_build / (_oracle + get_shared_lib_extension())),
    shared=True, link_objects=[],
)


@struct
class Natural:
    tag: u8
    value: u32


@struct(alignment=8)
class Packed1:
    tag: u8
    value: u64


@struct(alignment=16)
class Packed2:
    tag: u8
    value: u32


@struct(abi_alignment=128)
class Aligned:
    value: u32


@struct(alignment=8, abi_alignment=32)
class PackedAligned:
    tag: u8
    value: u32


@union(alignment=8)
class PackedUnion:
    value: u64
    bytes: array[u8, 8]


@struct(alignment=8)
class WithUnion:
    tag: u8
    data: PackedUnion


@struct(alignment=8)
class FloatBox:
    value: f32


@struct(abi_alignment=64)
class AlignedMember:
    value: u32


@struct(alignment=8)
class PackedOuter:
    tag: u32
    data: AlignedMember


@struct(storage_unit=32, abi_alignment=8)
class WordFields:
    a: u8
    b: u8
    c: u8
    d: u8


Nested = struct[('tag', u8), ('values', array[Packed2, 2]), ('tail', u8)]
NestedAligned = struct[('tag', u8), ('value', Aligned), ('tail', u8)]
IntegerLanes = struct[u64, u64]
MixedLanes = struct[u64, f64]
IndirectResult = struct[u64, u64, u64]

IntegerCallback = func[u64, u64, u64, u64, u64, IntegerLanes, u64, u64]
MixedCallback = func[u64, u64, u64, u64, u64, u64, MixedLanes, f64, u64]
SretCallback = func[u64, u64, u64, u64, IntegerLanes, u64, IndirectResult]

Vector128 = simd[f32, 4]
Vector64 = simd[f32, 2]
VectorRecord = struct[('lanes', Vector128)]
VectorPair = struct[('first', Vector64), ('second', Vector64)]
VectorUnion = union[('lanes', Vector128), ('integer', u64)]
FloatUnion = union[('value', f64)]
IntegerUnion = union[('value', u64)]


@extern(lib=_library, name='pc_record_metric')
def c_metric(record: i32, metric: i32) -> u64:
    pass


@extern(lib=_library, name='pc_step_Natural')
def c_step_natural(value: Natural) -> Natural:
    pass


@extern(lib=_library, name='pc_step_Packed1')
def c_step_packed1(value: Packed1) -> Packed1:
    pass


@extern(lib=_library, name='pc_callback_Packed2')
def c_callback_packed2(callback: func[Packed2, Packed2], value: Packed2) -> Packed2:
    pass


@extern(lib=_library, name='pc_callback_WithUnion')
def c_callback_union(callback: func[WithUnion, WithUnion], value: WithUnion) -> WithUnion:
    pass


@extern(lib=_library, name='pc_callback_FloatUnion')
def c_callback_float_union(callback: func[FloatUnion, FloatUnion],
                           value: FloatUnion) -> FloatUnion:
    pass


@extern(lib=_library, name='pc_callback_IntegerUnion')
def c_callback_integer_union(callback: func[IntegerUnion, IntegerUnion],
                             value: IntegerUnion) -> IntegerUnion:
    pass


@extern(lib=_library, name='pc_callback_Aligned')
def c_callback_aligned(callback: func[Aligned, Aligned], value: Aligned) -> Aligned:
    pass


@extern(lib=_library, name='pc_step_PackedAligned')
def c_step_packed_aligned(value: PackedAligned) -> PackedAligned:
    pass


@extern(lib=_library, name='pc_step_FloatBox')
def c_step_float(value: FloatBox) -> FloatBox:
    pass


@extern(lib=_library, name='pc_step_PackedOuter')
def c_step_outer(value: PackedOuter) -> PackedOuter:
    pass


@extern(lib=_library, name='pc_stack_integer')
def c_stack_integer(a: u64, b: u64, c: u64, d: u64, e: u64,
                    value: IntegerLanes, tail: u64) -> u64:
    pass


@extern(lib=_library, name='pc_callback_integer')
def c_callback_integer(callback: IntegerCallback) -> u64:
    pass


@extern(lib=_library, name='pc_stack_mixed')
def c_stack_mixed(a: u64, b: u64, c: u64, d: u64, e: u64, f: u64,
                  value: MixedLanes, tail: f64) -> u64:
    pass


@extern(lib=_library, name='pc_callback_mixed')
def c_callback_mixed(callback: MixedCallback) -> u64:
    pass


@extern(lib=_library, name='pc_stack_sret')
def c_stack_sret(a: u64, b: u64, c: u64, d: u64,
                 value: IntegerLanes, tail: u64) -> IndirectResult:
    pass


@extern(lib=_library, name='pc_callback_sret')
def c_callback_sret(callback: SretCallback) -> IndirectResult:
    pass


@compile
def score_integer(a: u64, b: u64, c: u64, d: u64, e: u64,
                  value: IntegerLanes, tail: u64) -> u64:
    return 10000 * value[0] + 100 * value[1] + tail


@compile
def score_mixed(a: u64, b: u64, c: u64, d: u64, e: u64, f: u64,
                value: MixedLanes, tail: f64) -> u64:
    return 10000 * value[0] + 100 * u64(value[1]) + u64(tail)


@compile
def collect_sret(a: u64, b: u64, c: u64, d: u64,
                 value: IntegerLanes, tail: u64) -> IndirectResult:
    return (value[0], value[1], tail)


@compile
def invoke_integer(callback: IntegerCallback, value: IntegerLanes) -> u64:
    return callback(1, 2, 3, 4, 5, value, 12)


@compile
def native_argument_pressure(out: ptr[u64]) -> None:
    pair: IntegerLanes = (10, 20)
    mixed: MixedLanes = (10, 20.0)
    out[0] = c_stack_integer(1, 2, 3, 4, 5, pair, 12)
    out[1] = c_callback_integer(score_integer)
    out[2] = invoke_integer(c_stack_integer, pair)
    out[3] = c_stack_mixed(1, 2, 3, 4, 5, 6, mixed, 12.0)
    out[4] = c_callback_mixed(score_mixed)
    direct = c_stack_sret(1, 2, 3, 4, pair, 12)
    callback = c_callback_sret(collect_sret)
    out[5] = 10000 * direct[0] + 100 * direct[1] + direct[2]
    out[6] = 10000 * callback[0] + 100 * callback[1] + callback[2]


@extern(lib=_library, name='pc_step_vector')
def c_step_vector(value: VectorRecord, tail: f32) -> VectorRecord:
    pass


@extern(lib=_library, name='pc_callback_vector')
def c_callback_vector(callback: func[VectorRecord, f32, VectorRecord],
                      value: VectorRecord, tail: f32) -> VectorRecord:
    pass


@extern(lib=_library, name='pc_step_vector_pair')
def c_step_vector_pair(value: VectorPair, tail: f32) -> VectorPair:
    pass


@extern(lib=_library, name='pc_has_vector_pair_abi')
def c_has_vector_pair_abi() -> i32:
    pass


@extern(lib=_library, name='pc_step_vector_union')
def c_step_vector_union(value: VectorUnion, tail: f32) -> VectorUnion:
    pass


@extern(lib=_library, name='pc_last_vector_register')
def c_last_vector_register(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64,
                           g: f64, value: VectorRecord, tail: f32) -> f32:
    pass


@extern(lib=_library, name='pc_spilled_vector')
def c_spilled_vector(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, g: f64,
                     h: f64, value: VectorRecord, tail: f32) -> f32:
    pass


@compile
def bump_vector(value: VectorRecord, tail: f32) -> VectorRecord:
    return (value.lanes + tail,)


@compile
def native_vector_exchange(out: ptr[u64]) -> None:
    lanes: array[f32, 4] = (1.0, 2.0, 3.0, 4.0)
    vector = simd_load(ptr(lanes[0]), Vector128)
    record: VectorRecord = (vector,)
    direct = c_step_vector(record, 5.0)
    callback = c_callback_vector(bump_vector, record, 5.0)
    for i in range(4):
        out[i] = u64(direct.lanes[i])
        out[4 + i] = u64(callback.lanes[i])

    overlay: VectorUnion
    overlay.lanes = vector
    overlay = c_step_vector_union(overlay, 5.0)
    out[8] = u64(10.0 * overlay.lanes[1] + overlay.lanes[3])

    out[9] = u64(c_last_vector_register(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                         record, 5.0))
    out[10] = u64(c_spilled_vector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                                    record, 5.0))


@compile
def native_vector_pair_exchange() -> u64:
    lanes: array[f32, 4] = (1.0, 2.0, 3.0, 4.0)
    pair: VectorPair = (simd_load(ptr(lanes[0]), Vector64),
                        simd_load(ptr(lanes[2]), Vector64))
    result = c_step_vector_pair(pair, 5.0)
    return u64(100.0 * result.first[1] + 10.0 * result.second[0] + result.second[1])


@compile
def bump_packed(value: Packed2) -> Packed2:
    value[1] += u32(value.tag)
    return value


@compile
def bump_union(value: WithUnion) -> WithUnion:
    value.data.value += u64(value.tag)
    return value


@compile
def bump_float_union(value: FloatUnion) -> FloatUnion:
    value.value += 3.0
    return value


@compile
def bump_integer_union(value: IntegerUnion) -> IntegerUnion:
    value.value += 3
    return value


@compile
def native_union_exchange() -> u64:
    floating: FloatUnion = {'value': 38.5}
    integer: IntegerUnion = {'value': 37}
    a = c_callback_float_union(bump_float_union, floating)
    b = c_callback_integer_union(bump_integer_union, integer)
    return u64(100 * a.value) + b.value


@compile
def bump_aligned(value: Aligned) -> Aligned:
    value.value += 7
    return value


@compile
def native_exchange(out: ptr[u64]) -> None:
    natural: Natural = (2, 41)
    packed: Packed1 = {'tag': 3, 'value': 40}
    pair: Packed2 = (7, 32)
    shared: WithUnion
    shared.tag = 8
    shared.data.value = 30
    wide: Aligned = (30,)
    combined: PackedAligned = (2, 36)
    floating: FloatBox = (f32(1.0),)

    out[0] = u64(c_step_natural(natural).value)
    out[1] = c_step_packed1(packed).value
    out[2] = u64(c_callback_packed2(bump_packed, pair).value)
    out[3] = c_callback_union(bump_union, shared).data.value
    out[4] = u64(c_callback_aligned(bump_aligned, wide).value)
    out[5] = u64(c_step_packed_aligned(combined).value)
    out[6] = u64(c_step_float(floating).value * f32(28.0))


@compile
def native_packed_outer() -> u32:
    outer: PackedOuter = (10, (30,))
    return c_step_outer(outer).data.value


@compile
def pc_layout_metrics(out: ptr[u64], index: i32) -> None:
    values: Nested
    base = u64(ptr(values))
    out[0] = sizeof(Packed2)
    out[1] = offsetof(Packed2, 'value')
    out[2] = u64(ptr(values.values[index][1])) - u64(ptr(values.values[index]))
    out[3] = u64(ptr(values.values[1])) - u64(ptr(values.values[0]))
    out[4] = u64(ptr(values.tail)) - base
    out[5] = offsetof(WithUnion, 'data.value')
    wide: NestedAligned
    out[6] = u64(ptr(wide.value)) - u64(ptr(wide))
    out[7] = sizeof(NestedAligned)


@compile
def python_exchange(value: Packed2) -> Packed2:
    runtime: Packed2 = {'tag': value.tag, 'value': value[1] + 1}
    constant: static[Packed2] = (3, 4)
    return (runtime.tag + constant.tag, runtime.value + constant.value)


@compile
def packed_access_width(view: ptr[volatile[Packed2]]) -> u32:
    view[0][1] += 2
    return view.value


@compile
def explicit_word_access(view: ptr[volatile[WordFields]]) -> u8:
    view.b += 2
    return view.b + view.c


class TestRecordLayout(unittest.TestCase):
    def test_semantic_and_llvm_storage_match_the_c_compiler(self):
        from llvmlite import ir
        from pythoc.target_layout import get_target_layout

        context = ir.Context()
        target = get_target_layout()
        for index, (record, field) in enumerate((
            (Natural, 1), (Packed1, 1), (Packed2, 1), (Aligned, 0),
            (PackedAligned, 1), (Nested, 2), (NestedAligned, 2),
            (PackedUnion, 0), (WithUnion, 1), (FloatBox, 0), (PackedOuter, 1),
        )):
            with self.subTest(record=record):
                size, alignment, offset = (
                    c_metric(index, metric) for metric in range(3)
                )
                layout = record.get_layout()
                self.assertEqual((layout.allocation_size, layout.alignment),
                                 (size, alignment))
                self.assertEqual(layout.field_offsets[field], offset)
                llvm_type = record.get_llvm_type(context)
                physical = target.primitive(llvm_type)
                self.assertEqual((physical.allocation_size, physical.alignment),
                                 (size, alignment))
                if index in (0, 1, 2, 4, 5, 7, 8, 9):
                    storage = record.get_ctypes_type()
                    self.assertEqual((ctypes.sizeof(storage), ctypes.alignment(storage)),
                                     (size, alignment))
                    if index == 4:
                        self.assertEqual(storage.value.offset, offset)

    def test_gep_offsets_array_stride_and_nested_offsetof_match_c(self):
        result = (ctypes.c_uint64 * 8)()
        pc_layout_metrics(ctypes.addressof(result), 1)
        size = c_metric(2, 0)
        offset = c_metric(2, 2)
        self.assertEqual(tuple(result), (
            size, offset, offset, size, c_metric(5, 2), c_metric(8, 2),
            c_metric(3, 1), c_metric(6, 0),
        ))

    def test_native_by_value_and_callback_abi(self):
        result = (ctypes.c_uint64 * 7)()
        native_exchange(ctypes.addressof(result))
        self.assertEqual(tuple(result), (42,) * len(result))

    def test_same_storage_unions_cross_native_calls_and_callbacks(self):
        self.assertEqual(native_union_exchange(), 4242)

    def test_aggregate_abi_with_register_pressure_and_indirect_result(self):
        result = (ctypes.c_uint64 * 7)()
        native_argument_pressure(ctypes.addressof(result))
        self.assertEqual(tuple(result), (102012,) * len(result))

    def test_vector_aggregate_abi_with_native_c_and_register_pressure(self):
        result = (ctypes.c_uint64 * 11)()
        native_vector_exchange(ctypes.addressof(result))
        self.assertEqual(tuple(result), (6, 8, 10, 12, 12, 14, 16, 18, 29,
                                         1039, 1039))

    def test_pair_of_short_vectors_by_value(self):
        if not c_has_vector_pair_abi():
            self.skipTest('C oracle uses the pre-GCC 11.1 ABI for 64-bit float vectors')
        self.assertEqual(native_vector_pair_exchange(), 834)

    @unittest.skipUnless(_clang_abi, 'requires a Clang C ABI oracle')
    def test_by_value_abi_of_packed_record_with_overaligned_member(self):
        # Clang and GCC disagree when only the nested record boundary is unaligned.
        self.assertEqual(native_packed_outer(), 42)

    def test_python_boundary_runtime_and_static_initializers(self):
        result = python_exchange((2, 37))
        self.assertEqual((result.tag, result.value), (5, 42))

    def test_packed_volatile_access_alignment(self):
        value = Packed2.get_ctypes_type()(5, 40)
        self.assertEqual(packed_access_width(ctypes.addressof(value)), 42)
        self.assertEqual((value.tag, value.value), (5, 42))
        emitted = get_output_manager().get_ir_text(packed_access_width._group_key)
        body = emitted.split(' @packed_access_width(', 1)[1].split('\n}', 1)[0]
        self.assertRegex(body, r'load volatile i32, .*\balign 1\b')
        self.assertRegex(body, r'store volatile i32 .*\balign 1\b')

    def test_explicit_word_access_uses_the_requested_storage_unit(self):
        storage = (ctypes.c_ubyte * 4)(1, 2, 3, 4)
        self.assertEqual(explicit_word_access(ctypes.addressof(storage)), 7)
        self.assertEqual(tuple(storage), (1, 4, 3, 4))
        emitted = get_output_manager().get_ir_text(explicit_word_access._group_key)
        body = emitted.split(' @explicit_word_access(', 1)[1].split('\n}', 1)[0]
        self.assertRegex(body, r'load volatile i32, .*\balign 1\b')
        self.assertRegex(body, r'store volatile i32 .*\balign 1\b')

    def test_atomic_requires_aligned_member_address(self):
        output = StringIO()
        with (
            redirect_stderr(output),
            CompileSession(),
            patch.object(logger, 'raise_on_error', True),
            self.assertRaisesRegex(TypeError, 'requires 4-byte alignment'),
        ):
            @compile
            def unaligned_atomic(value: ptr[Packed2]) -> u32:
                return atomic_load(ptr(value.value))

            flush_all_pending_outputs()
        self.assertRegex(output.getvalue(), r'\[test_record_layout.py:\d+:\d+\]')

    @unittest.skipUnless(sys.platform.startswith('linux'), 'requires POSIX trap status')
    def test_atomic_checks_pointer_alignment_across_calls(self):
        with tempfile.TemporaryDirectory(prefix='pc-atomic-align-') as temporary:
            script = Path(temporary) / 'check.py'
            script.write_text('''\
import ctypes
import resource
import sys
from pythoc import atomic_load, compile, ptr, u32

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

@compile
def load(address: ptr[u32]) -> u32:
    return atomic_load(address)

@compile
def choose(address: ptr[u32], read: u32) -> u32:
    return atomic_load(address) if read else u32(9)

value = ctypes.c_uint32(42)
address = ctypes.addressof(value)
assert load(address) == 42
assert choose(address, 1) == 42
assert choose(address + 1, 0) == 9
if sys.argv[1] == 'misaligned':
    load(address + 1)
''')
            normal = subprocess.run([sys.executable, str(script), 'aligned'],
                                    cwd=temporary, capture_output=True)
            self.assertEqual(normal.returncode, 0, normal.stderr.decode())
            failed = subprocess.run([sys.executable, str(script), 'misaligned'],
                                    cwd=temporary, capture_output=True)
            self.assertIn(failed.returncode, (-signal.SIGILL, -signal.SIGTRAP),
                          failed.stderr.decode())

    def test_atomic_checks_constant_member_address(self):
        with (CompileSession(), patch.object(logger, 'raise_on_error', True),
              self.assertRaisesRegex(TypeError, 'requires 4-byte alignment, address provides 2')):
            @compile
            class Data:
                value: static[Packed2] = (1, 42)

            @compile
            def unaligned_static_atomic() -> u32:
                return atomic_load(ptr(Data.value.value))

            flush_all_pending_outputs()


if __name__ == '__main__':
    unittest.main()
