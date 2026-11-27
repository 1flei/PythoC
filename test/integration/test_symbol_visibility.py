"""Compiled native definitions preserve ELF symbol visibility."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import pythoc
from pythoc import compile, i32
from pythoc.artifact import LinkPlan

pythoc.init()


@compile(visibility='hidden')
def pc_visibility_hidden() -> i32:
    return 7


@compile(visibility='protected')
def pc_visibility_protected() -> i32:
    return pc_visibility_hidden()


@unittest.skipUnless(sys.platform.startswith('linux') and shutil.which('readelf'),
                     'requires ELF symbol inspection')
class SymbolVisibilityTest(unittest.TestCase):
    def test_linked_library_exports_only_visible_function(self):
        objects = LinkPlan.from_compiled_symbols(
            [pc_visibility_hidden, pc_visibility_protected]).obj_files
        with tempfile.TemporaryDirectory(prefix='pc-visibility-') as temporary:
            output = Path(temporary) / 'symbols.so'
            subprocess.run(['cc', '-shared', *objects, '-o', output], check=True)
            names = subprocess.check_output(['readelf', '-W', '--dyn-syms', output],
                                            text=True)
            self.assertNotIn('pc_visibility_hidden', names)
            visible = next(line for line in names.splitlines()
                           if 'pc_visibility_protected' in line)
            self.assertIn('PROTECTED', visible)


if __name__ == '__main__':
    unittest.main()
