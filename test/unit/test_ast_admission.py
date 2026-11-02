"""The lowering visitor must reject unknown nodes before visiting children."""

import ast
import unittest
from unittest.mock import patch

from pythoc.ast_visitor.visitor_impl import LLVMIRVisitor
from pythoc.logger import logger


class AstAdmissionVisitorTest(unittest.TestCase):
    def setUp(self):
        self.visitor = object.__new__(LLVMIRVisitor)
        self.visitor._current_node = None

    def test_unknown_statement_is_rejected_before_its_test_runs(self):
        node = ast.Assert(test=ast.Call(
            func=ast.Name(id='side_effect', ctx=ast.Load()),
            args=[], keywords=[],
        ))
        with patch.object(logger, 'error', side_effect=SyntaxError) as error:
            with self.assertRaises(SyntaxError):
                self.visitor.visit(node)
        self.assertEqual(error.call_args.args[0],
                         'Unsupported Python syntax: Assert')
        self.assertIs(error.call_args.kwargs['node'], node)
        self.assertIsNone(self.visitor._current_node)

    def test_unknown_expression_is_rejected(self):
        node = ast.NamedExpr(
            target=ast.Name(id='value', ctx=ast.Store()),
            value=ast.Constant(value=3),
        )
        with patch.object(logger, 'error', side_effect=SyntaxError):
            with self.assertRaises(SyntaxError):
                self.visitor.visit(node)

    def test_pass_is_an_explicit_noop(self):
        self.assertIsNone(self.visitor.visit(ast.Pass()))


if __name__ == '__main__':
    unittest.main()
