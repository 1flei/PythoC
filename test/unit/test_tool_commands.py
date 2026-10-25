"""Compiler, linker and archiver commands preserve executable boundaries."""

import shlex
import unittest
from unittest.mock import patch

from pythoc.utils.cc_utils import find_available_cc
from pythoc.utils.link_utils import (
    build_archive_command, build_link_command, find_available_linker,
    get_default_archivers,
)


class TestToolCommands(unittest.TestCase):
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
