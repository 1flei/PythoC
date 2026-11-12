"""Default pipelines must optimize code while preserving observable behavior."""

import ctypes
import unittest
from unittest.mock import patch

from llvmlite import binding, ir
from pythoc.compiler import LLVMCompiler
from pythoc.optimization import optimize_ir, optimization_identity


def sample_module():
    compiler = LLVMCompiler()
    module = compiler.module
    word = ir.IntType(64)
    helper = ir.Function(module, ir.FunctionType(word, [word]), name='helper')
    helper.linkage = 'internal'
    builder = ir.IRBuilder(helper.append_basic_block('entry'))
    builder.ret(builder.add(helper.args[0], ir.Constant(word, 7)))
    caller = ir.Function(module, helper.function_type, name='evaluate')
    builder = ir.IRBuilder(caller.append_basic_block('entry'))
    builder.ret(builder.call(helper, caller.args))
    return compiler


class OptimizationTest(unittest.TestCase):
    def test_o0_retains_calls_and_o2_inlines_ordinary_helpers(self):
        source = str(sample_module().module)
        with binding.parse_assembly(optimize_ir(source, 0)) as module:
            self.assertIn('call', str(module.get_function('evaluate')))
        with binding.parse_assembly(optimize_ir(source, 2)) as module:
            body = str(module.get_function('evaluate'))
            self.assertNotIn('call', body)
            self.assertIn('add', body)

    def test_all_levels_execute_with_identical_results(self):
        for level in range(4):
            with self.subTest(level=level):
                compiler = sample_module()
                compiler.optimize_module(level)
                module = binding.parse_assembly(compiler.get_ir())
                target = binding.Target.from_triple(module.triple)
                machine = target.create_target_machine()
                with binding.create_mcjit_compiler(module, machine) as engine:
                    engine.finalize_object()
                    function = ctypes.CFUNCTYPE(ctypes.c_int64, ctypes.c_int64)(
                        engine.get_function_address('evaluate'))
                    for value in [-1000, -7, 0, 17, 1 << 40]:
                        self.assertEqual(function(value), value + 7)

    @unittest.skipUnless(hasattr(binding, 'create_pass_manager_builder'),
                         'legacy pass manager is not present')
    def test_legacy_fallback_runs_the_real_installed_pipeline(self):
        with patch('pythoc.optimization._HAS_DEFAULT_NEW_PM', False):
            self.test_all_levels_execute_with_identical_results()
            self.test_o0_retains_calls_and_o2_inlines_ordinary_helpers()

    def test_invalid_levels_fail_before_entering_llvm(self):
        source = str(sample_module().module)
        for level in [-1, 4, True, 2.0]:
            with self.subTest(level=level):
                with self.assertRaisesRegex(ValueError, 'integer from 0 to 3'):
                    optimize_ir(source, level)

    def test_identity_distinguishes_output_affecting_settings(self):
        self.assertNotEqual(optimization_identity(0), optimization_identity(2))
        self.assertIn('llvmlite=', optimization_identity(2))
        self.assertIn('llvm=', optimization_identity(2))

    def test_invalid_ir_is_not_silently_retried(self):
        with self.assertRaises(RuntimeError):
            optimize_ir('not LLVM IR', 2)


if __name__ == '__main__':
    unittest.main()
