"""Unit tests for source context carried by ActiveCompileFrame.

During compilation, line-number semantics (debug info, linear events) read
the offset from the active compile frame, not the logger's global fields.
The logger fields remain as a fallback for decoration-time paths.
"""

import unittest

from pythoc.ast_visitor.base import LLVMIRVisitor
from pythoc.context import ActiveCompileFrame
from pythoc.logger import logger


class TestActualLineNumber(unittest.TestCase):
    def setUp(self):
        self.visitor = LLVMIRVisitor(
            module=None,
            builder=None,
            struct_types=None,
            compiler=None,
            user_globals={},
        )
        self._saved_offset = logger.current_line_offset

    def tearDown(self):
        logger.current_line_offset = self._saved_offset

    def test_frame_line_offset_used_during_compilation(self):
        """With an active frame, its line offset wins over the logger's."""
        self.visitor.func_state = ActiveCompileFrame(line_offset=41)
        logger.current_line_offset = 999  # stale decoration-time value
        self.assertEqual(self.visitor._get_actual_line_number(3), 44)

    def test_logger_fallback_without_frame(self):
        """Without a frame (decoration-time path), logger offset is used."""
        self.visitor.func_state = None
        logger.current_line_offset = 9
        self.assertEqual(self.visitor._get_actual_line_number(3), 12)

    def test_none_line_number_passthrough(self):
        self.visitor.func_state = ActiveCompileFrame(line_offset=41)
        self.assertIsNone(self.visitor._get_actual_line_number(None))

    def test_frame_defaults(self):
        """A default frame behaves like offset 0 (identity mapping)."""
        self.visitor.func_state = ActiveCompileFrame()
        self.assertEqual(self.visitor._get_actual_line_number(7), 7)


if __name__ == '__main__':
    unittest.main()
