import ast
import unittest

from llvmlite import ir

from pythoc.ast_visitor.base import LLVMIRVisitor as BaseVisitor
from pythoc.ast_visitor.expressions import ExpressionsMixin
from pythoc.ast_visitor.value_dispatcher import ValueRefDispatcher
from pythoc.builtin_entities import i32
from pythoc.builtin_entities.python_type import PythonType
from pythoc.valueref import wrap_value


class _MockVisitor:
    def __init__(self):
        self.fallback_calls = []

    def _perform_binary_operation(self, op, left, right, node):
        self.fallback_calls.append((op, left, right, node))
        return wrap_value(
            "fallback",
            kind="python",
            type_hint=PythonType.wrap("fallback", is_constant=True),
        )


class _LeftAddType:
    @classmethod
    def handle_add(cls, visitor, left, right, node):
        return wrap_value(
            "left-handler",
            kind="python",
            type_hint=PythonType.wrap("left-handler", is_constant=True),
        )


class _RightAddType:
    @classmethod
    def handle_radd(cls, visitor, left, right, node):
        return wrap_value(
            "right-handler",
            kind="python",
            type_hint=PythonType.wrap("right-handler", is_constant=True),
        )


class _NoBinOpType:
    pass


class TestValueRefDispatcher(unittest.TestCase):
    def setUp(self):
        self.visitor = _MockVisitor()
        self.dispatcher = ValueRefDispatcher(self.visitor)

    def _wrap_python(self, value):
        return wrap_value(
            value,
            kind="python",
            type_hint=PythonType.wrap(value, is_constant=True),
        )

    def test_handle_binop_folds_python_values(self):
        left = self._wrap_python(7)
        right = self._wrap_python(3)
        node = ast.BinOp(left=ast.Constant(7), op=ast.Add(), right=ast.Constant(3))

        result = self.dispatcher.handle_binop(left, right, node)

        self.assertTrue(result.is_python_value())
        self.assertEqual(result.get_python_value(), 10)
        self.assertEqual(self.visitor.fallback_calls, [])

    def test_handle_binop_prefers_left_handler(self):
        left = wrap_value(ir.Constant(ir.IntType(32), 1), kind="value", type_hint=_LeftAddType)
        right = wrap_value(ir.Constant(ir.IntType(32), 2), kind="value", type_hint=_RightAddType)
        node = ast.BinOp(left=ast.Name(id="lhs", ctx=ast.Load()), op=ast.Add(), right=ast.Name(id="rhs", ctx=ast.Load()))

        result = self.dispatcher.handle_binop(left, right, node)

        self.assertEqual(result.get_python_value(), "left-handler")
        self.assertEqual(self.visitor.fallback_calls, [])

    def test_handle_binop_uses_reverse_handler(self):
        left = wrap_value(ir.Constant(ir.IntType(32), 1), kind="value", type_hint=_NoBinOpType)
        right = wrap_value(ir.Constant(ir.IntType(32), 2), kind="value", type_hint=_RightAddType)
        node = ast.BinOp(left=ast.Name(id="lhs", ctx=ast.Load()), op=ast.Add(), right=ast.Name(id="rhs", ctx=ast.Load()))

        result = self.dispatcher.handle_binop(left, right, node)

        self.assertEqual(result.get_python_value(), "right-handler")
        self.assertEqual(self.visitor.fallback_calls, [])

    def test_handle_binop_falls_back_to_default_operation(self):
        left = wrap_value(ir.Constant(ir.IntType(32), 4), kind="value", type_hint=_NoBinOpType)
        right = wrap_value(ir.Constant(ir.IntType(32), 2), kind="value", type_hint=_NoBinOpType)
        node = ast.BinOp(left=ast.Name(id="lhs", ctx=ast.Load()), op=ast.Add(), right=ast.Name(id="rhs", ctx=ast.Load()))

        result = self.dispatcher.handle_binop(left, right, node)

        self.assertEqual(result.get_python_value(), "fallback")
        self.assertEqual(len(self.visitor.fallback_calls), 1)
        op, fallback_left, fallback_right, fallback_node = self.visitor.fallback_calls[0]
        self.assertIsInstance(op, ast.Add)
        self.assertIs(fallback_left, left)
        self.assertIs(fallback_right, right)
        self.assertIs(fallback_node, node)


class _DispatcherSpy:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def handle_binop(self, left, right, node):
        self.calls.append((left, right, node))
        return self.result


class _ExpressionVisitor(ExpressionsMixin):
    def __init__(self, left, right, dispatcher):
        self._left = left
        self._right = right
        self.value_dispatcher = dispatcher

    def visit_rvalue_expression(self, expr):
        if isinstance(expr, ast.Name) and expr.id == "left":
            return self._left
        if isinstance(expr, ast.Name) and expr.id == "right":
            return self._right
        raise AssertionError(f"Unexpected expr: {ast.dump(expr)}")


class TestExpressionsBinOpDelegation(unittest.TestCase):
    def test_visit_binop_delegates_to_value_dispatcher(self):
        left = wrap_value(ir.Constant(ir.IntType(32), 11), kind="value", type_hint=i32)
        right = wrap_value(ir.Constant(ir.IntType(32), 22), kind="value", type_hint=i32)
        expected = wrap_value(33, kind="python", type_hint=PythonType.wrap(33, is_constant=True))
        dispatcher = _DispatcherSpy(expected)
        visitor = _ExpressionVisitor(left, right, dispatcher)
        node = ast.BinOp(
            left=ast.Name(id="left", ctx=ast.Load()),
            op=ast.Add(),
            right=ast.Name(id="right", ctx=ast.Load()),
        )

        result = visitor.visit_BinOp(node)

        self.assertIs(result, expected)
        self.assertEqual(dispatcher.calls, [(left, right, node)])


class _ReadRValueDispatcherSpy:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def read_rvalue(self, value_ref, *, name=None):
        self.calls.append((value_ref, name))
        return self.result


class _BaseVisitorForRValue(BaseVisitor):
    def __init__(self, result, dispatcher):
        self._result = result
        self.value_dispatcher = dispatcher

    def visit_expression(self, expr):
        return self._result


class TestBaseVisitorRValueDelegation(unittest.TestCase):
    def test_visit_rvalue_expression_delegates_to_dispatcher(self):
        binding = wrap_value(ir.Constant(ir.IntType(32), 5), kind="value", type_hint=i32)
        expected = wrap_value(ir.Constant(ir.IntType(32), 6), kind="value", type_hint=i32)
        dispatcher = _ReadRValueDispatcherSpy(expected)
        visitor = _BaseVisitorForRValue(binding, dispatcher)
        expr = ast.Name(id="binding", ctx=ast.Load())

        result = visitor.visit_rvalue_expression(expr)

        self.assertIs(result, expected)
        self.assertEqual(dispatcher.calls, [(binding, "binding")])

    def test_read_rvalue_delegates_to_dispatcher(self):
        binding = wrap_value(ir.Constant(ir.IntType(32), 5), kind="value", type_hint=i32)
        expected = wrap_value(ir.Constant(ir.IntType(32), 6), kind="value", type_hint=i32)
        dispatcher = _ReadRValueDispatcherSpy(expected)
        visitor = _BaseVisitorForRValue(binding, dispatcher)

        result = visitor.read_rvalue(binding, name="tmp")

        self.assertIs(result, expected)
        self.assertEqual(dispatcher.calls, [(binding, "tmp")])


if __name__ == "__main__":
    unittest.main()
