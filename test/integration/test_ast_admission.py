"""Unsupported syntax fails directly in source and expansion paths."""

import ast
from contextlib import redirect_stderr
from io import StringIO
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from pythoc import compile, i32, inline
from pythoc.build.output_manager import flush_all_pending_outputs
from pythoc.session import CompileSession
from test.utils.test_utils import DeferredTestCase


class AstAdmissionTest(DeferredTestCase):
    def test_unhandled_type_error_is_reported_at_the_expression(self):
        source = """from pythoc import compile, i32, struct

@compile
def bad(index: i32) -> i32:
    pair: struct[i32, i32] = (1, 2)
    return pair[index]
"""
        root = Path(__file__).resolve().parents[2]
        environment = dict(os.environ, PYTHONPATH=str(root), PC_RAISE_ON_ERROR='0')
        with TemporaryDirectory() as directory:
            script = Path(directory, 'bad_subscript.py')
            for eager in (True, False):
                with self.subTest(eager=eager):
                    trigger = ('bad(0)' if eager else
                               'from pythoc.build.output_manager import flush_all_pending_outputs\n'
                               'flush_all_pending_outputs()')
                    script.write_text(source + '\n' + trigger + '\n', encoding='utf-8')
                    result = subprocess.run(
                        [sys.executable, str(script)], cwd=directory, env=environment,
                        capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertRegex(result.stderr,
                                     r'\[bad_subscript.py:6:11\].*struct subscript index must be a constant')
                    self.assertEqual(result.stderr.count('[ERROR]'), 1)
                    self.assertNotIn('Traceback', result.stderr)

    def test_native_signature_rejects_unrepresented_parameters(self):
        with self.assertRaisesRegex(TypeError, r"positional-only parameters \(x\)"):
            @compile
            def positional(x: i32, /) -> i32:
                return 9

        with self.assertRaisesRegex(TypeError, r"keyword-only parameters \(factor\)"):
            @compile
            def keyword(value: i32, *, factor: i32 = 2) -> i32:
                return 9

    def test_generated_native_signature_reports_the_same_parameter(self):
        from pythoc.meta import compile_ast
        syntax = ast.parse("def generated(*, scale: i32 = 2) -> i32: return 9").body[0]
        output = StringIO()
        with CompileSession(), redirect_stderr(output):
            with self.assertRaisesRegex(TypeError, "keyword-only.*scale"):
                compile_ast(syntax, user_globals={"i32": i32},
                            source_file="generated_signature.py", start_line=42)
        self.assertRegex(output.getvalue(), r"\[generated_signature.py:42:\d+\].*keyword-only")

    def test_rejects_assert(self):
        with CompileSession(), self.assertRaisesRegex(
                SyntaxError, 'Unsupported Python syntax: Assert'):
            @compile
            def selected(value: i32) -> i32:
                assert value > 0
                return value

            flush_all_pending_outputs()

    def test_rejects_try(self):
        with CompileSession(), self.assertRaisesRegex(
                SyntaxError, 'Unsupported Python syntax: Try'):
            @compile
            def selected(value: i32) -> i32:
                try:
                    return value + 1
                except ValueError:
                    return value + 2

            flush_all_pending_outputs()

    def test_rejects_raise(self):
        with CompileSession(), self.assertRaisesRegex(
                SyntaxError, 'Unsupported Python syntax: Raise'):
            @compile
            def selected(value: i32) -> i32:
                raise ValueError('unsupported')

            flush_all_pending_outputs()

    def test_rejects_import(self):
        with CompileSession(), self.assertRaisesRegex(
                SyntaxError, 'Unsupported Python syntax: Import'):
            @compile
            def selected(value: i32) -> i32:
                import math
                return value

            flush_all_pending_outputs()

    def test_rejects_delete(self):
        with CompileSession(), self.assertRaisesRegex(
                SyntaxError, 'Unsupported Python syntax: Delete'):
            @compile
            def selected(value: i32) -> i32:
                del value
                return 1

            flush_all_pending_outputs()

    def test_rejects_named_expression(self):
        with CompileSession(), self.assertRaisesRegex(
                SyntaxError, 'Unsupported Python syntax: NamedExpr'):
            @compile
            def selected(value: i32) -> i32:
                if (assigned := value) > 0:
                    return assigned
                return 0

            flush_all_pending_outputs()

    def test_rejects_inline(self):
        with CompileSession(), self.assertRaisesRegex(
                SyntaxError, 'Unsupported Python syntax: Assert'):
            @inline
            def inline_assert(value):
                assert value > 0
                return value

            @compile
            def selected(value: i32) -> i32:
                return inline_assert(value)

            flush_all_pending_outputs()

    def test_rejects_generated(self):
        with CompileSession(), self.assertRaisesRegex(
                SyntaxError, 'Unsupported Python syntax: Assert'):
            from pythoc import meta

            generated = meta.func(
                name='generated_assert',
                params=[('value', i32)],
                return_type=i32,
                body=[
                    ast.Assert(test=ast.Compare(
                        left=ast.Name(id='value', ctx=ast.Load()),
                        ops=[ast.Gt()],
                        comparators=[ast.Constant(value=0)],
                    )),
                    ast.Return(value=ast.Name(id='value', ctx=ast.Load())),
                ],
            )
            selected = meta.compile_generated(generated, user_globals=globals())

            flush_all_pending_outputs()

    def test_pass_remains_a_valid_noop(self):
        with CompileSession():
            @compile
            def selected(value: i32) -> i32:
                pass
                return value + 1

            self.assertEqual(selected(3), 4)


if __name__ == '__main__':
    unittest.main()
