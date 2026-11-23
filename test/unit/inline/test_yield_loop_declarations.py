import ast
import unittest

from pythoc._inline.kernel import _build_loop_var_declarations
from pythoc.utils.ast_targets import iter_unpack_leaves


class TestYieldLoopDeclarations(unittest.TestCase):
    def test_reuses_visible_loop_target(self):
        declarations = _build_loop_var_declarations(
            iter_unpack_leaves(ast.Name(id="i", ctx=ast.Store())),
            ast.Name(id="i32", ctx=ast.Load()),
            {"i"},
        )
        self.assertEqual(declarations, [])

    def test_declares_only_new_tuple_targets(self):
        declarations = _build_loop_var_declarations(
            iter_unpack_leaves(ast.Tuple(
                elts=[
                    ast.Name(id="key", ctx=ast.Store()),
                    ast.Name(id="value", ctx=ast.Store()),
                ],
                ctx=ast.Store(),
            )),
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
            {"key"},
        )
        self.assertEqual(len(declarations), 1)
        self.assertEqual(declarations[0].target.id, "value")
        from pythoc import f32, i32, struct
        from pythoc.type_resolver import TypeResolver
        resolver = TypeResolver(user_globals={'struct': struct, 'i32': i32,
                                              'f32': f32})
        self.assertIs(resolver.parse_annotation(declarations[0].annotation), f32)

    def test_projects_nested_fields_and_declares_repeated_names_once(self):
        from pythoc import f32, i32, i64, struct
        from pythoc.type_resolver import TypeResolver

        target = ast.parse('((a, b), a) = values').body[0].targets[0]
        leaves = list(iter_unpack_leaves(target))
        self.assertEqual([(path, node.id) for path, node in leaves],
                         [((0, 0), 'a'), ((0, 1), 'b'), ((1,), 'a')])
        annotation = ast.parse('struct[struct[i32, f32], i64]', mode='eval').body
        declarations = _build_loop_var_declarations(leaves, annotation)
        resolver = TypeResolver(user_globals={'struct': struct, 'i32': i32,
                                              'f32': f32, 'i64': i64})
        self.assertEqual([(decl.target.id, resolver.parse_annotation(decl.annotation))
                          for decl in declarations], [('a', i32), ('b', f32)])


if __name__ == "__main__":
    unittest.main()
