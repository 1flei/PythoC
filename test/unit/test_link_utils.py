"""Unit tests for link command construction in utils.link_utils."""

import unittest

from pythoc.utils.link_utils import build_link_command


class TestBuildLinkCommandExtraFlags(unittest.TestCase):
    def test_extra_flags_appended_last(self):
        cmd = build_link_command(
            ['a.o'], 'out', link_libraries=[],
            extra_flags=['-Wl,-dead_strip', '-v'],
        )
        self.assertEqual(cmd[-2:], ['-Wl,-dead_strip', '-v'])

    def test_extra_flags_default_off(self):
        with_flags = build_link_command(
            ['a.o'], 'out', link_libraries=[], extra_flags=['-v'],
        )
        without_flags = build_link_command(['a.o'], 'out', link_libraries=[])
        self.assertEqual(with_flags, without_flags + ['-v'])

    def test_extra_flags_shared(self):
        cmd = build_link_command(
            ['a.o'], 'out.so', shared=True, link_libraries=[],
            extra_flags=['-undefined', 'dynamic_lookup'],
        )
        self.assertEqual(cmd[-2:], ['-undefined', 'dynamic_lookup'])


if __name__ == '__main__':
    unittest.main()
