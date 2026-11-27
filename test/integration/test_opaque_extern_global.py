"""Taking the address of a foreign object needs no local layout."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from pythoc import compile, extern, extern_global, i32, ptr
from pythoc.artifact import LinkPlan
from pythoc.type_resolver import TypeReference


OpaqueRecord = TypeReference('ForeignOpaqueRecord', {})
opaque_declaration = extern_global(OpaqueRecord, 'pc_foreign_opaque', lib='')
opaque_by_name = extern_global('ForeignOpaqueRecord', 'pc_foreign_opaque', lib='')


@extern(lib='')
def pc_read_foreign_opaque(data: ptr[OpaqueRecord]) -> i32:
    pass


@compile
def read_foreign_opaque() -> i32:
    return (pc_read_foreign_opaque(ptr(opaque_declaration))
            + pc_read_foreign_opaque(ptr(opaque_by_name)))


@unittest.skipUnless(shutil.which('cc'), 'requires a native C toolchain')
class OpaqueExternGlobalTest(unittest.TestCase):
    def test_can_take_address_of_incomplete_external_storage(self):
        objects = LinkPlan.from_compiled_symbols([read_foreign_opaque]).obj_files
        with tempfile.TemporaryDirectory(prefix='pc-extern-opaque-') as temporary:
            directory = Path(temporary)
            source = directory / 'provider.c'
            source.write_text('''\
struct opaque { int value; };
struct opaque pc_foreign_opaque = {17};
int pc_read_foreign_opaque(struct opaque *data) { return data->value; }
extern int read_foreign_opaque(void);
int main(void) { return read_foreign_opaque() != 34; }
''')
            output = directory / 'client'
            subprocess.run(['cc', source, *objects, '-o', output], check=True)
            subprocess.run([output], check=True)


if __name__ == '__main__':
    unittest.main()
