import ast
import unittest

from pythoc import i32
from pythoc.inline.closure_capture import build_closure_capture_plan
from pythoc.inline.state_field_rewriter import (
    StateFieldRewriter,
    StateFieldRewritePolicy,
)
from pythoc.meta.instantiate import _compile_closure_fn_pipeline


class _FakeValueRef:
    def __init__(self, *, python_value=None, pc_type=None):
        self._python_value = python_value
        self._pc_type = pc_type

    def is_python_value(self):
        return self._pc_type is None

    def is_pcvalue(self):
        return self._pc_type is not None

    def get_python_value(self):
        return self._python_value

    def get_pc_type(self):
        return self._pc_type


class _FakeVarInfo:
    def __init__(self, value_ref):
        self.value_ref = value_ref


class TestClosureCapturePlan(unittest.TestCase):
    def test_splits_compiletime_and_runtime_captures(self):
        visible = {
            "scale": _FakeVarInfo(_FakeValueRef(python_value=3)),
            "offset": _FakeVarInfo(_FakeValueRef(pc_type=i32)),
        }

        plan = build_closure_capture_plan({"scale", "offset"}, visible)

        self.assertEqual(plan.bindings, {"scale": 3})
        self.assertEqual(len(plan.runtime), 1)
        self.assertEqual(plan.runtime[0].name, "offset")
        self.assertIs(plan.runtime[0].pc_type, i32)


class TestStateFieldRewriter(unittest.TestCase):
    def test_preserves_nested_function_and_rewrites_state_capture(self):
        fn = ast.parse(
            """
def outer(x: i32) -> i32:
    local: i32 = 3
    def inner(y: i32) -> i32:
        return y + local + offset
    return inner(x)
"""
        ).body[0]

        rewriter = StateFieldRewriter(StateFieldRewritePolicy(
            field_names={"offset"},
            protect_names={"s"},
            preserve_nested_functions=True,
        ))
        rewritten = [rewriter.visit(stmt) for stmt in fn.body]
        src = "\n".join(ast.unparse(stmt) for stmt in rewritten)

        self.assertIn("def inner", src)
        self.assertIn("s.offset", src)
        self.assertIn("local", src)


class TestClosureGeneratorBoundary(unittest.TestCase):
    def test_closure_with_yield_is_rejected(self):
        fn = ast.parse(
            """
def bad() -> i32:
    yield 1
"""
        ).body[0]

        with self.assertRaises((TypeError, SystemExit)):
            _compile_closure_fn_pipeline(
                fn,
                capture_bindings=None,
                source_object_id=1,
                func_name_hint="bad",
                callee_globals={"i32": i32},
            )


if __name__ == "__main__":
    unittest.main()
