"""Layout probing must degrade to the C probe when libclang is absent.

libclang is an optional dependency; the layout it reads consists only of
compile-time constants, which a compile-and-run probe obtains with the
same toolchain that links the runtime extension anyway.
"""

import unittest
from unittest.mock import patch

from pythoc import callable_layout


class TestCallableLayoutProbeFallback(unittest.TestCase):
    def test_cc_probe_matches_libclang(self):
        include_dirs = callable_layout._include_dirs()
        try:
            reference = callable_layout._layout_via_libclang(include_dirs)
        except Exception as error:
            self.skipTest('libclang unavailable: {}'.format(error))
        probed = callable_layout._layout_via_cc_probe(include_dirs)
        self.assertEqual(probed, reference)

    def test_fallback_engages_without_libclang(self):
        def _boom(include_dirs):
            raise RuntimeError('no libclang')

        with patch.object(
            callable_layout, '_layout_via_libclang', _boom
        ):
            layout = callable_layout.python_abi_layout()
        # _check() already ran inside; spot-check the essential invariants.
        self.assertEqual(layout['pointer'], 8)
        self.assertEqual(layout['have_vectorcall'], 1)
        self.assertIn('tp_vectorcall_offset', layout)


if __name__ == '__main__':
    unittest.main()
