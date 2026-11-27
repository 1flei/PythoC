"""An explicit ABI name leaves the Python binding callable."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from pythoc import compile, i32, param
from pythoc.artifact import LinkPlan


@compile(name='pc_native_symbol')
def python_binding() -> i32:
    return 31


@compile(name='pc_named_specialization')
def specialize(increment: param, value: i32) -> i32:
    return value + increment


@compile(name='abs')
def abs_binding(value: i32) -> i32:
    return 42


@unittest.skipUnless(shutil.which('cc'), 'requires a native C toolchain')
class NativeSymbolNameTest(unittest.TestCase):
    def test_specialization_preserves_native_base_name(self):
        specialized = specialize(7)
        self.assertEqual(specialized(35), 42)
        self.assertTrue(specialized._actual_func_name.startswith('pc_named_specialization_'))

    @unittest.skipUnless(sys.platform.startswith('linux'), 'requires ELF symbol preemption')
    def test_python_call_binds_its_own_kernel(self):
        self.assertEqual(abs_binding(-6), 42)

    def test_explicit_name_is_callable_from_python_and_c(self):
        self.assertEqual(python_binding(), 31)
        objects = LinkPlan.from_compiled_symbols([python_binding]).obj_files
        with tempfile.TemporaryDirectory(prefix='pc-symbol-name-') as temporary:
            directory = Path(temporary)
            source = directory / 'client.c'
            source.write_text('''\
extern int pc_native_symbol(void);
int main(void) { return pc_native_symbol() != 31; }
''')
            output = directory / 'client'
            subprocess.run(['cc', source, *objects, '-o', output], check=True)
            subprocess.run([output], check=True)


if __name__ == '__main__':
    unittest.main()
