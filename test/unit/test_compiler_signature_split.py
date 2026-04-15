"""
Unit tests for compiler declaration resolution and wrapper state layers.
"""

import ast
import unittest
from unittest.mock import patch

from pythoc import f64, i32, struct
from pythoc.compiler import LLVMCompiler
from pythoc.context import ActiveCompileFrame, FunctionBindingState
from pythoc.registry import FunctionInfo


class TestFunctionStateLayers(unittest.TestCase):
    """Verify semantic info, binding state, and compile frame are distinct."""

    def test_function_info_binding_and_frame_are_separated(self):
        wrapper = object()
        binding = FunctionBindingState(
            source_file="demo.py",
            original_name="demo",
            actual_func_name="demo_impl",
            compilation_globals={"demo": wrapper},
            wrapper=wrapper,
        )
        func_info = FunctionInfo(
            name="demo",
            source_file="demo.py",
            return_type_hint=i32,
            param_type_hints={"x": i32},
            param_names=["x"],
            binding_state=binding,
        )
        frame = ActiveCompileFrame()

        self.assertIs(func_info.wrapper, wrapper)
        self.assertIs(func_info.compilation_globals, binding.compilation_globals)
        self.assertIs(func_info.binding_state, binding)
        self.assertFalse(hasattr(binding, "current_function"))
        self.assertFalse(hasattr(binding, "varargs_info"))
        self.assertEqual(frame.all_inlined_stmts, [])

        callable_type = func_info.callable_pc_type
        self.assertEqual(callable_type.param_types, (i32,))
        self.assertIs(callable_type.return_type, i32)


class TestCompilerDeclarationResolution(unittest.TestCase):
    """Verify declaration resolution uses one parsed PC-type source."""

    def setUp(self):
        self.compiler = LLVMCompiler(user_globals={"i32": i32, "f64": f64})

    def test_resolve_function_declaration_does_not_use_legacy_parse_path(self):
        fn_ast = ast.parse(
            "def add(x: i32) -> i32:\n"
            "    return x\n"
        ).body[0]

        with patch.object(
            self.compiler,
            "_parse_type_annotation",
            side_effect=AssertionError("legacy parse path should be unused"),
        ):
            resolved = self.compiler._resolve_function_declaration(
                fn_ast,
                user_globals={"i32": i32},
            )

        self.assertEqual(resolved.param_names, ["x"])
        self.assertIs(resolved.param_type_hints["x"], i32)
        self.assertIs(resolved.return_type_hint, i32)

    def test_resolve_function_declaration_expands_struct_varargs_with_pc_types(self):
        Pair = struct[i32, f64]
        fn_ast = ast.parse(
            "def pack(x: i32, *args: Pair) -> i32:\n"
            "    return x\n"
        ).body[0]

        resolved = self.compiler._resolve_function_declaration(
            fn_ast,
            user_globals={"i32": i32, "f64": f64, "Pair": Pair},
        )

        self.assertEqual(resolved.varargs.kind, "typed")
        self.assertFalse(resolved.has_llvm_varargs)
        self.assertEqual(resolved.param_names, ["x", "args_elem0", "args_elem1"])
        self.assertEqual(resolved.varargs.element_types, [i32, f64])
        self.assertIs(resolved.param_type_hints["args_elem0"], i32)
        self.assertIs(resolved.param_type_hints["args_elem1"], f64)

    def test_resolve_function_declaration_preserves_empty_struct_varargs(self):
        Empty = struct[tuple([])]
        fn_ast = ast.parse(
            "def acquire(*args: Empty) -> i32:\n"
            "    return 0\n"
        ).body[0]

        resolved = self.compiler._resolve_function_declaration(
            fn_ast,
            user_globals={"i32": i32, "Empty": Empty},
        )

        self.assertEqual(resolved.varargs.kind, "typed")
        self.assertFalse(resolved.has_llvm_varargs)
        self.assertEqual(resolved.varargs.element_types, [])
        self.assertEqual(resolved.param_names, [])


if __name__ == "__main__":
    unittest.main()
