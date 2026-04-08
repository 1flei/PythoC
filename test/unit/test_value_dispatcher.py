import ast
import unittest

from llvmlite import ir

from pythoc.ast_visitor.base import LLVMIRVisitor as BaseVisitor
from pythoc.ast_visitor.expressions import ExpressionsMixin
from pythoc.value_dispatcher import ValueRefDispatcher
from pythoc.builtin_entities import bool as pc_bool
from pythoc.builtin_entities import i32, ptr
from pythoc.builtin_entities.python_type import PythonType
from pythoc.valueref import wrap_python_constant, wrap_value


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


class _AssignDecayType:
    @classmethod
    def handle_assign_decay(cls, visitor, value_ref):
        return wrap_value(
            "decayed",
            kind="python",
            type_hint=PythonType.wrap("decayed", is_constant=True),
        )


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

    def test_handle_compare_folds_python_values(self):
        left = self._wrap_python(2)
        right = self._wrap_python(5)

        result = self.dispatcher.handle_compare(ast.Lt(), left, right, ast.Constant(value=5))

        self.assertTrue(result.is_python_value())
        self.assertTrue(result.get_python_value())

    def test_prepare_assignment_rvalue_uses_type_handler(self):
        value_ref = wrap_value(
            ir.Constant(ir.IntType(32), 1),
            kind="value",
            type_hint=_AssignDecayType,
        )

        result = self.dispatcher.prepare_assignment_rvalue(value_ref)

        self.assertTrue(result.is_python_value())
        self.assertEqual(result.get_python_value(), "decayed")


class _ExpressionDispatcherSpy:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def handle_binop(self, left, right, node):
        self.calls.append(("binop", left, right, node))
        return self.result

    def handle_unary(self, operand, node):
        self.calls.append(("unary", operand, node))
        return self.result

    def handle_compare_chain(self, left, ops, comparators, node):
        self.calls.append(("compare", left, tuple(ops), tuple(comparators), node))
        return self.result


class _ExpressionVisitor(ExpressionsMixin):
    def __init__(self, dispatcher, values=None):
        self.value_dispatcher = dispatcher
        self._values = values or {}

    def visit_rvalue_expression(self, expr):
        if isinstance(expr, ast.Name) and expr.id in self._values:
            return self._values[expr.id]
        raise AssertionError(f"Unexpected expr: {ast.dump(expr)}")


class TestExpressionsDispatcherDelegation(unittest.TestCase):
    def test_visit_binop_delegates_to_value_dispatcher(self):
        left = wrap_value(ir.Constant(ir.IntType(32), 11), kind="value", type_hint=i32)
        right = wrap_value(ir.Constant(ir.IntType(32), 22), kind="value", type_hint=i32)
        expected = wrap_value(33, kind="python", type_hint=PythonType.wrap(33, is_constant=True))
        dispatcher = _ExpressionDispatcherSpy(expected)
        visitor = _ExpressionVisitor(dispatcher, {"left": left, "right": right})
        node = ast.BinOp(
            left=ast.Name(id="left", ctx=ast.Load()),
            op=ast.Add(),
            right=ast.Name(id="right", ctx=ast.Load()),
        )

        result = visitor.visit_BinOp(node)

        self.assertIs(result, expected)
        self.assertEqual(dispatcher.calls, [("binop", left, right, node)])

    def test_visit_unaryop_delegates_to_value_dispatcher(self):
        operand = wrap_value(ir.Constant(ir.IntType(32), 11), kind="value", type_hint=i32)
        expected = wrap_value(-11, kind="python", type_hint=PythonType.wrap(-11, is_constant=True))
        dispatcher = _ExpressionDispatcherSpy(expected)
        visitor = _ExpressionVisitor(dispatcher, {"value": operand})
        node = ast.UnaryOp(op=ast.USub(), operand=ast.Name(id="value", ctx=ast.Load()))

        result = visitor.visit_UnaryOp(node)

        self.assertIs(result, expected)
        self.assertEqual(dispatcher.calls, [("unary", operand, node)])

    def test_visit_compare_delegates_to_value_dispatcher(self):
        left = wrap_value(ir.Constant(ir.IntType(32), 1), kind="value", type_hint=i32)
        expected = wrap_value(True, kind="python", type_hint=PythonType.wrap(True, is_constant=True))
        dispatcher = _ExpressionDispatcherSpy(expected)
        visitor = _ExpressionVisitor(dispatcher, {"left": left})
        comparator = ast.Name(id="right", ctx=ast.Load())
        node = ast.Compare(left=ast.Name(id="left", ctx=ast.Load()), ops=[ast.Lt()], comparators=[comparator])

        result = visitor.visit_Compare(node)

        self.assertIs(result, expected)
        self.assertEqual(dispatcher.calls, [("compare", left, (node.ops[0],), (comparator,), node)])


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


class _TypeConverterSpy:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def convert(self, value, target_type, node=None):
        self.calls.append((value, target_type, node))
        return self.result


class _TypeCallTarget:
    calls = []

    @classmethod
    def reset(cls):
        cls.calls = []

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node):
        cls.calls.append((visitor, func_ref, list(args), node))
        return "type-call-result"


class _TypeCallVisitor:
    def __init__(self, converter_result=None):
        self.type_converter = _TypeConverterSpy(converter_result)
        self.value_dispatcher = ValueRefDispatcher(self)


class _DirectValueDispatcherSpy:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def handle_type_call(self, func_ref, args, node):
        self.calls.append((func_ref, list(args), node))
        return self.result


class _ExplicitCastDispatcherSpy:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def explicit_cast(self, target_type, value, node=None):
        self.calls.append((target_type, value, node))
        return self.result


class _NoAstEvalVisitor:
    def __init__(self, dispatcher_result=None):
        self.value_dispatcher = _DirectValueDispatcherSpy(dispatcher_result)

    def visit_rvalue_expression(self, expr):
        raise AssertionError("visit_rvalue_expression should not be called")


class TestTypeCallsThroughValueDispatcher(unittest.TestCase):
    def test_handle_type_call_delegates_to_protocol(self):
        _TypeCallTarget.reset()
        visitor = _TypeCallVisitor(converter_result=None)
        arg = wrap_value(ir.Constant(ir.IntType(32), 7), kind="value", type_hint=i32)
        func_ref = wrap_python_constant(_TypeCallTarget)
        node = ast.Call(func=ast.Name(id="target", ctx=ast.Load()), args=[], keywords=[])

        result = visitor.value_dispatcher.handle_type_call(func_ref, [arg], node)

        self.assertEqual(result, "type-call-result")
        self.assertEqual(len(_TypeCallTarget.calls), 1)
        _, called_func_ref, args, call_node = _TypeCallTarget.calls[0]
        self.assertTrue(called_func_ref.is_python_value())
        self.assertIs(called_func_ref.get_python_value(), _TypeCallTarget)
        self.assertEqual(args, [arg])
        self.assertIs(call_node, node)

    def test_explicit_cast_delegates_to_type_converter(self):
        value = wrap_value(ir.Constant(ir.IntType(32), 5), kind="value", type_hint=i32)
        expected = wrap_value(ir.Constant(ir.IntType(32), 6), kind="value", type_hint=i32)
        visitor = _TypeCallVisitor(converter_result=expected)

        result = visitor.value_dispatcher.explicit_cast(i32, value, node=None)

        self.assertIs(result, expected)
        self.assertEqual(visitor.type_converter.calls, [(value, i32, None)])

    def test_builtin_type_handle_type_call_uses_pre_evaluated_args(self):
        arg = wrap_value(ir.Constant(ir.IntType(32), 5), kind="value", type_hint=i32)
        expected = wrap_value(ir.Constant(ir.IntType(32), 5), kind="value", type_hint=i32)
        dispatcher = _ExplicitCastDispatcherSpy(expected)
        visitor = type("_Visitor", (), {"value_dispatcher": dispatcher})()
        node = ast.Call(func=ast.Name(id="i32", ctx=ast.Load()), args=[ast.Name(id="x", ctx=ast.Load())], keywords=[])

        result = i32.handle_type_call(visitor, None, [arg], node)

        self.assertIs(result, expected)
        self.assertEqual(dispatcher.calls, [(i32, arg, node)])

    def test_python_type_handle_call_delegates_type_calls_to_value_dispatcher(self):
        arg = wrap_value(ir.Constant(ir.IntType(32), 5), kind="value", type_hint=i32)
        expected = wrap_value("ok", kind="python", type_hint=PythonType.wrap("ok", is_constant=True))
        visitor = _NoAstEvalVisitor(dispatcher_result=expected)
        python_type = PythonType.wrap(_TypeCallTarget, is_constant=True)
        func_ref = wrap_value(_TypeCallTarget, kind="python", type_hint=python_type)
        node = ast.Call(func=ast.Name(id="target", ctx=ast.Load()), args=[ast.Name(id="x", ctx=ast.Load())], keywords=[])

        result = python_type.handle_call(visitor, func_ref, [arg], node)

        self.assertIs(result, expected)
        self.assertEqual(visitor.value_dispatcher.calls, [(func_ref, [arg], node)])

    def test_python_type_compile_time_call_uses_pre_evaluated_args(self):
        visitor = _NoAstEvalVisitor(dispatcher_result=None)
        python_type = PythonType.wrap(lambda x: x + 1, is_constant=True)
        func_ref = wrap_value(python_type.get_python_object(), kind="python", type_hint=python_type)
        arg = wrap_value(7, kind="python", type_hint=PythonType.wrap(7, is_constant=True))
        node = ast.Call(func=ast.Name(id="inc", ctx=ast.Load()), args=[ast.Name(id="x", ctx=ast.Load())], keywords=[])

        result = python_type.handle_call(visitor, func_ref, [arg], node)

        self.assertTrue(result.is_python_value())
        self.assertEqual(result.get_python_value(), 8)


class TestValueRefDispatcherRuntimeSemantics(unittest.TestCase):
    def _make_runtime_visitor(self):
        module = ir.Module(name="dispatcher_test")
        func_type = ir.FunctionType(ir.VoidType(), [])
        function = ir.Function(module, func_type, name="test")
        block = function.append_basic_block("entry")
        builder = ir.IRBuilder(block)
        return BaseVisitor(module=module, builder=builder, user_globals={})

    def test_handle_compare_runtime_path_returns_bool_value(self):
        visitor = self._make_runtime_visitor()
        left = wrap_value(ir.Constant(ir.IntType(32), 1), kind="value", type_hint=i32)
        right = wrap_value(ir.Constant(ir.IntType(32), 2), kind="value", type_hint=i32)

        result = visitor.value_dispatcher.handle_compare(ast.Lt(), left, right, None)

        self.assertIs(result.type_hint, pc_bool)
        self.assertEqual(result.value.type, ir.IntType(1))

    def test_to_boolean_converts_pointer_values(self):
        visitor = self._make_runtime_visitor()
        pointee = visitor.builder.alloca(ir.IntType(32), name="x")
        visitor.builder.store(ir.Constant(ir.IntType(32), 1), pointee)
        ptr_value = wrap_value(pointee, kind="value", type_hint=ptr[i32])

        result = visitor.value_dispatcher.to_boolean(ptr_value)

        self.assertEqual(result.type, ir.IntType(1))

    def test_handle_unary_not_returns_bool_value(self):
        visitor = self._make_runtime_visitor()
        operand = wrap_value(ir.Constant(ir.IntType(32), 7), kind="value", type_hint=i32)
        node = ast.UnaryOp(op=ast.Not(), operand=ast.Name(id="x", ctx=ast.Load()))

        result = visitor.value_dispatcher.handle_unary(operand, node)

        self.assertIs(result.type_hint, pc_bool)
        self.assertEqual(result.value.type, ir.IntType(1))


if __name__ == "__main__":
    unittest.main()
