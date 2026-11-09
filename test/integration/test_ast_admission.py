"""Unsupported syntax fails directly in source and expansion paths."""

import ast
import unittest

from pythoc import compile, i32, inline
from pythoc.build.output_manager import flush_all_pending_outputs
from pythoc.session import CompileSession
from test.utils.test_utils import DeferredTestCase


class AstAdmissionTest(DeferredTestCase):
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
