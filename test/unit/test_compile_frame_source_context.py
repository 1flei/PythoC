"""Unit tests for source context carried by ActiveCompileFrame.

During compilation, line-number semantics (debug info, linear events) read
the offset from the active compile frame, not the logger's global fields.
The logger fields remain as a fallback for decoration-time paths.
"""

import ast
from contextlib import redirect_stderr
from io import StringIO
import unittest

from pythoc.ast_visitor.base import LLVMIRVisitor
from pythoc.context import ActiveCompileFrame
from pythoc.logger import Logger, logger


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


class TestPropagatedDiagnostics(unittest.TestCase):
    def test_reporting_retains_the_first_location_and_exception(self):
        compiler_logger = Logger()
        compiler_logger.set_raise_on_error(True)
        error = TypeError('invalid operand')
        expression = ast.parse('a + b', mode='eval').body
        compiler_logger.set_source_context('operand.py', 6)
        compiler_logger.remember_error_location(error, expression.right)
        compiler_logger.set_source_context('caller.py', 13)
        compiler_logger.remember_error_location(error, expression)
        output = StringIO()
        with redirect_stderr(output):
            for _ in range(2):
                with self.assertRaises(TypeError) as caught:
                    compiler_logger.error(error)
                self.assertIs(caught.exception, error)
        self.assertEqual(output.getvalue(), '[ERROR] [operand.py:7:4] invalid operand\n')


if __name__ == '__main__':
    unittest.main()
