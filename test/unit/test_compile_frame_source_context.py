"""Unit tests for source context carried by ActiveCompileFrame.

During compilation, line-number semantics (debug info, linear events) read
the offset from the active compile frame, not the logger's global fields.
The logger fields remain as a fallback for decoration-time paths.
"""

import ast
from contextlib import redirect_stderr
from io import StringIO
import unittest
from unittest.mock import patch

from pythoc import array, i32
from pythoc.ast_visitor.base import LLVMIRVisitor
from pythoc.builtin_entities.struct import create_struct_type
from pythoc.context import ActiveCompileFrame
from pythoc.logger import Logger, logger
from pythoc.source import SourceLocation, anchor_generated_ast, set_ast_source
from pythoc.type_resolver import TypeResolver


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
    def test_explicit_python_api_location_overrides_the_enclosing_node(self):
        compiler_logger = Logger()
        compiler_logger.set_raise_on_error(True)
        output = StringIO()
        with (redirect_stderr(output),
              compiler_logger.at_node(SourceLocation('outer.py', 3)),
              self.assertRaisesRegex(TypeError, 'invalid option')):
            compiler_logger.error('invalid option', node=SourceLocation('entry.py', 7),
                                  exc_type=TypeError)
        self.assertEqual(output.getvalue(), '[ERROR] [entry.py:7] invalid option\n')

    def test_python_record_metadata_accepts_positional_field_names(self):
        record = create_struct_type([i32], ['payload'])
        self.assertEqual(record.get_field_index('payload'), 0)
        self.assertIs(record.get_field_type(0), i32)

    def test_type_subscript_passes_its_node_to_the_constructor_diagnostic(self):
        syntax = ast.parse('array[i32, -1]', mode='eval').body
        set_ast_source(syntax, ('declaration.py', 20))
        resolver = TypeResolver(user_globals={'array': array, 'i32': i32})
        output = StringIO()
        with (patch.object(logger, 'raise_on_error', True),
              patch.object(logger, 'error', wraps=logger.error) as report,
              redirect_stderr(output),
              self.assertRaisesRegex(TypeError, 'dimensions must be non-negative')):
            resolver.parse_annotation(syntax)
        self.assertIs(report.call_args.kwargs['node'], syntax)
        self.assertIn('[declaration.py:21:0]', output.getvalue())

    def test_type_subscript_constructor_requires_node_argument(self):
        with self.assertRaisesRegex(TypeError, 'node'):
            array.handle_type_subscript(((None, i32), (None, 1)))

    def test_generated_coordinates_do_not_replace_authored_provenance(self):
        compiler_logger = Logger()
        compiler_logger.set_raise_on_error(True)
        expression = ast.parse('original + generated', mode='eval').body
        set_ast_source(expression.left, ('definition.py', 20))
        anchor_generated_ast(expression, SourceLocation('builder.py', 42))
        output = StringIO()
        with redirect_stderr(output), compiler_logger.at_node(
                expression, source=('consumer.py', 90)):
            with self.assertRaises(RuntimeError):
                compiler_logger.error('authored', node=expression.left)
            with self.assertRaises(RuntimeError):
                compiler_logger.error('generated', node=expression.right)
        self.assertEqual(output.getvalue().splitlines(), [
            '[ERROR] [definition.py:21:0] authored',
            '[ERROR] [builder.py:42] generated',
        ])

    def test_diagnostics_prefer_explicit_node_and_restore_enclosing_source(self):
        compiler_logger = Logger()
        compiler_logger.set_raise_on_error(True)
        expression = ast.parse('a + b', mode='eval').body
        inner = ast.parse('call()', mode='eval').body
        output = StringIO()

        with redirect_stderr(output), compiler_logger.at_node(
                expression, source=('outer.py', 10)):
            with self.assertRaises(RuntimeError):
                compiler_logger.error('explicit', node=expression.right)
            with compiler_logger.at_node(inner, source=('inner.py', 20)):
                with compiler_logger.at_node(ast.Name(id='generated', ctx=ast.Load())):
                    with self.assertRaises(RuntimeError):
                        compiler_logger.error('generated')
            with self.assertRaises(RuntimeError):
                compiler_logger.error('outer')
        with redirect_stderr(output), self.assertRaises(RuntimeError):
            compiler_logger.error('outside')

        self.assertEqual(output.getvalue().splitlines()[:3], [
            '[ERROR] [outer.py:11:4] explicit',
            '[ERROR] [inner.py:21:0] generated',
            '[ERROR] [outer.py:11:0] outer',
        ])
        self.assertIn('test_compile_frame_source_context.py:', output.getvalue().splitlines()[3])

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
