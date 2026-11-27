"""A native C client can use PythoC-defined data without an adapter."""

from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from llvmlite import binding, ir

import pythoc
from pythoc import (
    array, compile, extern_global, i8, i32, ptr, static, struct, u8, union,
)
from pythoc.artifact import LinkPlan
from pythoc.decorators.class_statics import _StaticMember
from pythoc.logger import logger

pythoc.init()


@compile
class UnionField:
    tag: u8
    count: i32


@union
class UnionSeed:
    small: UnionField
    bytes: array[u8, 16]


@compile
class RelocatedField:
    tag: u8
    element: ptr[i32]
    count: i32


@union
class RelocatedUnion:
    entry: RelocatedField
    bytes: array[u8, 32]


@union(abi_alignment=128)
class AlignedRelocation:
    address: ptr[i32]


@compile(static_member_symbol_prefix='', linkage='external')
class NativeData:
    pc_defined_counter: static[i32] = _StaticMember(37, alignment=32)
    pc_defined_array: static[array[i32, 3]] = (2, 5, 9)
    pc_defined_zero: static[i32]
    pc_counter_address: static[ptr[i32]] = lambda: ptr(NativeData.pc_defined_counter)
    pc_array_element: static[ptr[i32]] = lambda: ptr(NativeData.pc_defined_array[2])
    pc_aligned_relocation: static[AlignedRelocation] = lambda: {
        'address': ptr(NativeData.pc_defined_counter),
    }
    pc_union_seed: static[UnionSeed] = {'small': (9, -17)}
    pc_relocated_union: static[RelocatedUnion] = lambda: {
        'entry': (13, ptr[i32](ptr[i8](ptr(NativeData.pc_defined_array)) + 8), -21),
    }


@compile(static_member_symbol_prefix='', linkage='internal')
class SectionEntries:
    pc_entry_first: static[i32] = _StaticMember(17, section='pc_test_entries', used=True)
    pc_entry_second: static[i32] = _StaticMember(23, section='pc_test_entries', used=True)


@compile(static_member_symbol_prefix='abi', suffix='v2', linkage='external')
class Prefixed:
    marker: static[i32] = 28


@compile
def defined_global_read() -> i32:
    return NativeData.pc_defined_counter


@compile
def defined_array_read() -> i32:
    return NativeData.pc_defined_array[1]


@compile
def defined_union_read() -> i32:
    return NativeData.pc_union_seed.small.count


@compile
def defined_relocated_union_read() -> i32:
    return NativeData.pc_relocated_union.entry.element[0]


@compile
def defined_aligned_relocation_read() -> i32:
    return NativeData.pc_aligned_relocation.address[0]


@compile
def runtime_union_initializer(count: i32) -> i32:
    value: UnionSeed = {'small': (u8(3), count)}
    return value.small.count + i32(value.small.tag)


@unittest.skipUnless(shutil.which('cc'), 'requires a native C toolchain')
class DefinedGlobalTest(unittest.TestCase):
    def test_union_alignment_does_not_widen_pointer_relocations(self):
        self.assertEqual(defined_aligned_relocation_read(), 37)

    def test_static_only_group_replays_ir_on_cache_hit(self):
        with tempfile.TemporaryDirectory(prefix='pc-static-replay-') as temporary:
            directory = Path(temporary)
            (directory / 'check.py').write_text('''\
from pathlib import Path
from pythoc import compile, flush_all_pending_outputs, i32, static
from pythoc.build.output_manager import get_output_manager

@compile(static_member_symbol_prefix='', linkage='external')
class Globals:
    pc_replay_counter: static[i32] = 42

flush_all_pending_outputs()
manager = get_output_manager()
key = Globals._static_group_key
text = manager.get_ir_text(key)
assert 'global i32 42' in text, text
print(Path(manager.get_group(key)['obj_file']).stat().st_mtime_ns)
''')
            first = subprocess.check_output([sys.executable, 'check.py'], cwd=directory)
            second = subprocess.check_output([sys.executable, 'check.py'], cwd=directory)
            self.assertEqual(first, second)

    def test_duplicate_definition_points_to_second_static_member(self):
        output = StringIO()
        with (pythoc.CompileSession(), patch.object(logger, 'raise_on_error', True),
              redirect_stderr(output), self.assertRaisesRegex(ValueError, 'Duplicate definition')):
            @compile(static_member_symbol_prefix='', linkage='external')
            class First:
                pc_duplicate_data: static[i32] = 1

            @compile(static_member_symbol_prefix='', linkage='external')
            class Second:
                pc_duplicate_data: static[i32] = 2

            pythoc.flush_all_pending_outputs()
        self.assertRegex(output.getvalue(), r'\[test_defined_global.py:\d+:\d+\] Duplicate definition')

    def test_packed_union_initializer_diagnoses_unaligned_relocation(self):
        @struct(alignment=8)
        class PackedField:
            tag: u8
            element: ptr[i32]

        @union
        class PackedRelocation:
            entry: PackedField

        output = StringIO()
        with (pythoc.CompileSession(), patch.object(logger, 'raise_on_error', True),
              redirect_stderr(output), self.assertRaisesRegex(ValueError, 'relocation at offset 1')):
            @compile(static_member_symbol_prefix='', linkage='external')
            class PackedData:
                pc_packed_counter: static[i32] = 42
                pc_packed_relocation: static[PackedRelocation] = lambda: {
                    'entry': (1, ptr(PackedData.pc_packed_counter)),
                }

            pythoc.flush_all_pending_outputs()
        self.assertRegex(output.getvalue(), r'\[test_defined_global.py:\d+:\d+\] Union initializer')

    def test_conflicting_weak_declarations_are_rejected_in_both_orders(self):
        for weak in (True, False):
            with self.subTest(weak_first=weak), patch.object(logger, 'raise_on_error', True):
                module = ir.Module()
                first = extern_global(i32, 'pc_conflict', lib='', weak=weak)
                second = extern_global(i32, 'pc_conflict', lib='', weak=not weak)
                visitor = SimpleNamespace(module=module)
                first.lower_to_module(visitor)
                with self.assertRaisesRegex(TypeError, 'Conflicting linkage'):
                    second.lower_to_module(visitor)

    def test_named_runtime_union_initializer(self):
        self.assertEqual(runtime_union_initializer(39), 42)

    def test_deferred_initializer_must_name_its_actual_expression(self):
        def select(unused, *, initializer):
            return initializer

        with (pythoc.CompileSession(), patch.object(logger, 'raise_on_error', True),
              self.assertRaisesRegex(TypeError, 'inline zero-argument lambda')):
            @compile(static_member_symbol_prefix='', linkage='external')
            class Ambiguous:
                pc_ambiguous_init: static[i32] = select(lambda: 11, initializer=lambda: 22)

            pythoc.flush_all_pending_outputs()

    def test_deferred_initializer_rejects_discarded_runtime_work(self):
        with tempfile.TemporaryDirectory(prefix='pc-static-runtime-') as temporary:
            directory = Path(temporary)
            (directory / 'check.py').write_text('''\
from pythoc import compile, i32, static
from pythoc.artifact import LinkPlan

@compile
def runtime() -> i32:
    return 1

@compile(static_member_symbol_prefix='', linkage='external')
class Globals:
    counter: static[i32] = lambda: (runtime(), 7)[1]

LinkPlan.from_all_groups()
''')
            result = subprocess.run([sys.executable, 'check.py'], cwd=directory,
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('requires a compile-time constant initializer',
                          result.stdout + result.stderr)

    def test_cross_module_reference_links_the_storage_owner(self):
        with tempfile.TemporaryDirectory(prefix='pc-static-owner-') as temporary:
            directory = Path(temporary)
            (directory / 'provider.py').write_text('''\
from pythoc import compile, i32, static

@compile(static_member_symbol_prefix='', linkage='external')
class Globals:
    pc_cross_module_counter: static[i32] = 19
    pc_cross_module_unused: static[i32] = 23
''')
            (directory / 'consumer.py').write_text('''\
from provider import Globals
from pythoc import compile, i32

@compile
def read_counter() -> i32:
    return Globals.pc_cross_module_counter
''')
            (directory / 'check.py').write_text('''\
from pathlib import Path
import subprocess
from consumer import read_counter
from pythoc.artifact import LinkPlan

plan = LinkPlan.from_compiled_symbols([read_counter])
assert len(plan.obj_files) == 2
client = Path('client')
subprocess.run(['cc', 'client.c', *plan.obj_files, '-o', client], check=True)
subprocess.run([client.resolve()], check=True)
assert read_counter() == 19
''')
            (directory / 'client.c').write_text('''\
extern int pc_cross_module_counter, pc_cross_module_unused;
extern int read_counter(void);
int main(void) {
    if (pc_cross_module_unused != 23 || read_counter() != 19)
        return 1;
    pc_cross_module_counter = 37;
    return read_counter() != 37;
}
''')
            subprocess.run([sys.executable, 'check.py'], cwd=directory, check=True)

    def test_external_data_visibility_does_not_create_an_initializer(self):
        module = ir.Module()
        for visibility, weak in [('hidden', False), ('protected', True)]:
            symbol_name = 'pc_' + visibility
            global_binding = extern_global(array[i32, 0], symbol_name, lib='',
                                           visibility=visibility, weak=weak)
            global_binding.lower_to_module(SimpleNamespace(module=module))
            linkage = 'extern_weak' if weak else 'external'
            self.assertIn(f'@"{symbol_name}" = {linkage} {visibility} global [0 x i32]',
                          str(module).splitlines())
        binding.parse_assembly(str(module)).verify()

    @unittest.skipUnless(sys.platform.startswith('linux'), 'requires ELF start and stop symbols')
    def test_linker_discovers_retained_internal_entries(self):
        objects = LinkPlan.from_compiled_symbols([defined_global_read]).obj_files
        with tempfile.TemporaryDirectory(prefix='pc-defined-section-') as temporary:
            directory = Path(temporary)
            source = directory / 'consumer.c'
            source.write_text('''\
extern int __start_pc_test_entries[], __stop_pc_test_entries[];
int main(void) {
    int *begin = __start_pc_test_entries;
    return __stop_pc_test_entries - begin != 2 || begin[0] + begin[1] != 40;
}
''')
            output = directory / 'consumer'
            subprocess.run(['cc', source, *objects, '-o', output], check=True)
            subprocess.run([output], check=True)

    def test_native_client_reads_and_writes_defined_storage(self):
        roots = [defined_global_read, defined_array_read, defined_union_read,
                 defined_relocated_union_read, defined_aligned_relocation_read]
        objects = LinkPlan.from_compiled_symbols(roots).obj_files
        with tempfile.TemporaryDirectory(prefix='pc-defined-global-') as tmp:
            directory = Path(tmp)
            source = directory / 'consumer.c'
            source.write_text('''\
#include <stdint.h>
extern int pc_defined_counter, pc_defined_array[3], pc_defined_zero, *pc_counter_address;
extern int *pc_array_element;
extern int prefixed_marker __asm__("abi_v2.marker");
extern union { struct { uint8_t tag; int count; } small; uint8_t bytes[16]; } pc_union_seed;
extern union {
    struct { uint8_t tag; int *element; int count; } entry;
    uint8_t bytes[32];
} pc_relocated_union;
extern union __attribute__((aligned(16))) { int *address; } pc_aligned_relocation;
extern int defined_global_read(void), defined_array_read(void);
extern int defined_union_read(void), defined_relocated_union_read(void);
int main(void) {
    if (pc_defined_counter != 37 || pc_defined_array[2] != 9 || pc_defined_zero ||
        prefixed_marker != 28 ||
        pc_counter_address != &pc_defined_counter || (uintptr_t)pc_counter_address % 32 ||
        pc_array_element != &pc_defined_array[2] || *pc_array_element != 9 ||
        pc_union_seed.small.tag != 9 || pc_union_seed.small.count != -17 ||
        pc_union_seed.bytes[1] || pc_union_seed.bytes[12] || defined_union_read() != -17 ||
        pc_relocated_union.entry.tag != 13 || pc_relocated_union.entry.element != &pc_defined_array[2] ||
        pc_relocated_union.entry.count != -21 || pc_relocated_union.bytes[31] ||
        pc_aligned_relocation.address != &pc_defined_counter || (uintptr_t)&pc_aligned_relocation % 16 ||
        defined_relocated_union_read() != 9)
        return 1;
    pc_defined_counter = 12;
    pc_defined_array[1] = 13;
    return defined_global_read() != 12 || defined_array_read() != 13;
}
''')
            output = directory / 'consumer'
            subprocess.run(['cc', source, *objects, '-o', output], check=True)
            subprocess.run([output], check=True)


if __name__ == '__main__':
    unittest.main()
