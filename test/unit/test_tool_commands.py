"""Compiler, linker and archiver commands preserve executable boundaries."""

import shlex
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pythoc.utils.cc_utils import compile_c_to_object, find_available_cc
from pythoc.utils.link_utils import (
    build_archive_command, build_link_command, find_available_linker,
    get_default_archivers,
)


class TestToolCommands(unittest.TestCase):
    def test_c_compiler_dialect_has_an_explicit_default_and_can_be_overridden(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'source.c'
            source.write_text('', encoding='utf-8')
            output = source.with_suffix('.o')
            for flags in ([], ['-std=c99']):
                with self.subTest(flags=flags), patch(
                    'pythoc.utils.cc_utils.subprocess.run',
                ) as run:
                    compile_c_to_object(str(source), str(output), cc='cc', cflags=flags)
                    command = run.call_args.args[0]
                    self.assertEqual([flag for flag in command if flag.startswith('-std=')],
                                     ['-std=gnu11', *flags])

    def test_windows_zig_paths_survive_all_command_builders(self):
        for path in (r'C:\tools\zig.exe', r'C:\Program Files\Zig\zig.exe'):
            with self.subTest(path=path), patch('sys.platform', 'win32'), \
                    patch('pythoc.utils.cc_utils._zig_tool_cmd',
                          return_value=[path, 'cc']), patch(
                              'pythoc.utils.link_utils._zig_tool_cmd',
                              side_effect=lambda tool: [path, tool]), patch(
                                  'pythoc.utils.link_utils._which_cached',
                                  return_value=None):
                compiler = find_available_cc()
                linker = find_available_linker()
                archiver = get_default_archivers()[0]
                self.assertEqual(shlex.split(compiler), [path, 'cc'])
                command = build_link_command(
                    ['source.o'], 'program.exe', linker=linker,
                    link_libraries=[], link_objects=[],
                )
                self.assertEqual(command[:2], [path, 'cc'])
                self.assertEqual(build_archive_command(
                    ['source.o'], 'source.lib', archiver=archiver,
                )[:2], [path, 'ar'])


if __name__ == '__main__':
    unittest.main()
