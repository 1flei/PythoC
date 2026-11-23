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
    def test_python_compiler_entries_locate_the_call_without_a_traceback(self):
        root = Path(__file__).resolve().parents[2]
        environment = dict(os.environ, PYTHONPATH=str(root), PC_RAISE_ON_ERROR='0')
        cases = (
            ('inline(cls_method=True, method=True)', 'Cannot use both cls_method=True and method=True'),
            ('i32()', 'i32() takes exactly 1 positional argument (0 given)'),
            ("i32('bad')", 'invalid literal for int()'),
            ('enum[1]', 'Invalid enum variant: 1'),
            ("struct['value': i32].get_field_index('missing')", 'struct_i32_value has no field named'),
            ('instantiate([])', 'instantiate cannot determine the element type of an empty iterable'),
            ("extern(calling_convention='invalid')", 'Unsupported calling convention: invalid'),
            ('FlatHashMap(f64, i32)', 'no default hash for key type f64'),
        )
        imports = ('from pythoc import enum, extern, f64, i32, inline, instantiate, struct\n'
                   'from pythoc.std.map import FlatHashMap\n')
        with TemporaryDirectory() as directory:
            script = Path(directory, 'compiler_api.py')
            for expression, message in cases:
                with self.subTest(expression=expression):
                    script.write_text(
                        imports + expression + '\n',
                        encoding='utf-8',
                    )
                    result = subprocess.run(
                        [sys.executable, str(script)], cwd=directory, env=environment,
                        capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertIn(f'[ERROR] [compiler_api.py:3] {message}', result.stderr)
                    self.assertEqual(result.stderr.count('[ERROR]'), 1)
                    self.assertNotIn('Traceback', result.stderr)

    def test_type_helper_diagnostics_locate_body_and_deferred_annotation(self):
        body = """from pythoc import array, compile, i32

@compile
def invalid() -> i32:
    bad: array[i32, -1]
    return 0

invalid()
"""
        signature = """from __future__ import annotations
from pythoc import array, compile, i32

@compile
def invalid(
    bad: array[i32, -1],
) -> i32:
    return 0
"""
        nested = """from __future__ import annotations
from pythoc import array, compile, i32, param

@compile
def specialize(N: param, data: array[i32, N]) -> i32:
    return 0

@compile
def caller() -> i32:
    return specialize(-1, 0)

caller()
"""
        root = Path(__file__).resolve().parents[2]
        environment = dict(os.environ, PYTHONPATH=str(root), PC_RAISE_ON_ERROR='0')
        with TemporaryDirectory() as directory:
            for label, source, location in (
                    ('body', body, '5:9'), ('signature', signature, '6:9'),
                    ('nested', nested, '5:31')):
                with self.subTest(label=label):
                    script = Path(directory, f'{label}.py')
                    script.write_text(source, encoding='utf-8')
                    result = subprocess.run(
                        [sys.executable, str(script)], cwd=directory, env=environment,
                        capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertIn(
                        f'[ERROR] [{script.name}:{location}] array dimensions must be non-negative',
                        result.stderr,
                    )
                    self.assertEqual(result.stderr.count('[ERROR]'), 1)
                    self.assertNotIn('Traceback', result.stderr)

    def test_subscript_error_is_reported_at_the_expression(self):
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

    def test_invalid_struct_field_reports_its_annotation(self):
        output = StringIO()
        with redirect_stderr(output), self.assertRaisesRegex(
                TypeError, "cannot be used as a struct field"):
            from pythoc import param

            @compile
            class Invalid:
                field: param

        field_line = type(self).test_invalid_struct_field_reports_its_annotation.__code__.co_firstlineno + 8
        self.assertRegex(output.getvalue(), rf"\[test_ast_admission.py:{field_line}:\d+\]")

    def test_inline_error_keeps_the_callee_source_across_modules(self):
        callee = """from pythoc import array, i32, inline

@inline
def invalid():
    local: array[i32, -1]
    return 0
"""
        caller = """from pythoc import compile, i32
from inline_callee import invalid

@compile
def selected() -> i32:
    return invalid()

selected()
"""
        root = Path(__file__).resolve().parents[2]
        environment = dict(os.environ, PYTHONPATH=str(root), PC_RAISE_ON_ERROR='0')
        with TemporaryDirectory() as directory:
            Path(directory, 'inline_callee.py').write_text(callee, encoding='utf-8')
            script = Path(directory, 'inline_caller.py')
            script.write_text(caller, encoding='utf-8')
            result = subprocess.run(
                [sys.executable, str(script)], cwd=directory, env=environment,
                capture_output=True, text=True, check=False,
            )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn(
            '[ERROR] [inline_callee.py:5:11] array dimensions must be non-negative',
            result.stderr,
        )
        self.assertEqual(result.stderr.count('[ERROR]'), 1)
        self.assertNotIn('Traceback', result.stderr)

    def test_python_meta_error_reports_only_the_source_diagnostic(self):
        source = """from pythoc import compile, i32

def check():
    raise ValueError('invalid compile-time input')

@compile
def selected() -> i32:
    return check()

selected()
"""
        root = Path(__file__).resolve().parents[2]
        environment = dict(os.environ, PYTHONPATH=str(root), PC_RAISE_ON_ERROR='0')
        with TemporaryDirectory() as directory:
            script = Path(directory, 'meta_error.py')
            script.write_text(source, encoding='utf-8')
            result = subprocess.run(
                [sys.executable, str(script)], cwd=directory, env=environment,
                capture_output=True, text=True, check=False,
            )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(
            result.stderr.strip(),
            '[ERROR] [meta_error.py:8:11] Python call failed in check: invalid compile-time input',
        )

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
        output = StringIO()
        with redirect_stderr(output), CompileSession(), self.assertRaisesRegex(
                SyntaxError, 'Unsupported Python syntax: Assert'):
            from pythoc import meta

            origin_line = sys._getframe().f_lineno + 1
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
            meta.compile_generated(generated, user_globals=globals())

            flush_all_pending_outputs()
        self.assertIn(f'[test_ast_admission.py:{origin_line}]', output.getvalue())

    def test_quoted_fragment_keeps_its_authored_source(self):
        from pythoc import meta

        @meta.quote
        def bad_body(value):
            assert value

        assert_line = bad_body.origin_line + 2
        output = StringIO()
        with redirect_stderr(output), CompileSession(), self.assertRaisesRegex(
                SyntaxError, 'Unsupported Python syntax: Assert'):
            generated = meta.func(
                'bad_quote', [('value', i32)], i32, bad_body('value'))
            meta.compile_generated(generated)
            flush_all_pending_outputs()
        self.assertRegex(output.getvalue(), rf'\[test_ast_admission.py:{assert_line}:\d+\]')

    def test_parsed_generated_ast_without_filename_uses_compilation_site(self):
        from pythoc import meta

        syntax = ast.parse('def bad():\n    assert False').body[0]
        output = StringIO()
        with redirect_stderr(output), CompileSession(), self.assertRaisesRegex(
                SyntaxError, 'Unsupported Python syntax: Assert'):
            origin_line = sys._getframe().f_lineno + 1
            meta.compile_ast(syntax)
            flush_all_pending_outputs()
        self.assertIn(f'[test_ast_admission.py:{origin_line}]', output.getvalue())

    def test_unresolved_signature_reports_the_parameter_during_group_preparation(self):
        output = StringIO()
        with redirect_stderr(output), CompileSession(), self.assertRaisesRegex(
                NameError, 'Type annotation name.*UnresolvedSignatureParameter'):
            parameter_line = sys._getframe().f_lineno + 3
            @compile(suffix='invalid_signature_location')
            def bad(
                value: 'UnresolvedSignatureParameter',
            ) -> i32:
                return 0

            flush_all_pending_outputs()
        self.assertRegex(output.getvalue(), rf'\[test_ast_admission.py:{parameter_line}:\d+\]')
        self.assertEqual(output.getvalue().count('[ERROR]'), 1)

    def test_invalid_annotation_is_reported_by_the_compiler(self):
        output = StringIO()
        with redirect_stderr(output), CompileSession(), self.assertRaisesRegex(
                TypeError, 'Cannot use as type: 17'):
            annotation_line = sys._getframe().f_lineno + 3
            @compile(suffix='invalid_annotation_diagnostic')
            def bad() -> i32:
                value: '17' = 0
                return 0

            flush_all_pending_outputs()
        self.assertRegex(output.getvalue(), rf'\[test_ast_admission.py:{annotation_line}:\d+\]')
        self.assertEqual(output.getvalue().count('[ERROR]'), 1)

    def test_pass_remains_a_valid_noop(self):
        with CompileSession():
            @compile
            def selected(value: i32) -> i32:
                pass
                return value + 1

            self.assertEqual(selected(3), 4)


if __name__ == '__main__':
    unittest.main()
