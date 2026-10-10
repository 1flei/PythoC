import ast
import unittest

from pythoc._inline.kernel import _build_loop_var_declarations


class TestYieldLoopDeclarations(unittest.TestCase):
    def test_reuses_visible_loop_target(self):
        declarations = _build_loop_var_declarations(
            ast.Name(id="i", ctx=ast.Store()),
            ast.Name(id="i32", ctx=ast.Load()),
            ast.Constant(value=None),
            {"i"},
        )
        self.assertEqual(declarations, [])

    def test_declares_only_new_tuple_targets(self):
        declarations = _build_loop_var_declarations(
            ast.Tuple(
                elts=[
                    ast.Name(id="key", ctx=ast.Store()),
                    ast.Name(id="value", ctx=ast.Store()),
                ],
                ctx=ast.Store(),
            ),
            ast.Subscript(
                value=ast.Name(id="struct", ctx=ast.Load()),
                slice=ast.Tuple(
                    elts=[
                        ast.Name(id="i32", ctx=ast.Load()),
                        ast.Name(id="f32", ctx=ast.Load()),
                    ],
                    ctx=ast.Load(),
                ),
                ctx=ast.Load(),
            ),
            ast.Constant(value=None),
            {"key"},
        )
        self.assertEqual(len(declarations), 1)
        self.assertEqual(declarations[0].target.id, "value")
        self.assertEqual(declarations[0].annotation.id, "f32")


if __name__ == "__main__":
    unittest.main()
