"""Extern declarations reuse matching C symbols without rewriting snapshots."""

import unittest

from llvmlite import ir

from pythoc import extern, i32, i64, struct
from pythoc.builder import LLVMBuilder
from pythoc.callable_lowering import lower_extern_wrapper
from pythoc.valueref import ensure_ir


@extern(lib="c", name="native_function")
def declared(value: i32) -> i32:
    pass


@extern(lib="c", name="native_function")
def wrong_signature(value: i64) -> i64:
    pass


@extern(lib="c", name="native_function", calling_convention="stdcall")
def wrong_convention(value: i32) -> i32:
    pass


Pair = struct[i32, i32]


@extern(lib="c", name="pair_function")
def pair_declared(value: Pair) -> Pair:
    pass


class TestExternDefinition(unittest.TestCase):
    def test_matching_local_definition_is_reused(self):
        module = ir.Module()
        local = LLVMBuilder().declare_function(
            module, "native_function", [ir.IntType(32)], ir.IntType(32),
        ).ir_function
        ir.IRBuilder(local.append_basic_block("entry")).ret(local.args[0])
        self.assertIs(ensure_ir(lower_extern_wrapper(declared, module)), local)

    def test_mismatched_signature_is_rejected(self):
        module = ir.Module()
        lower_extern_wrapper(declared, module)
        with self.assertRaisesRegex(TypeError, "Conflicting function signature"):
            lower_extern_wrapper(wrong_signature, module)

    def test_calling_convention_mismatch_is_rejected(self):
        module = ir.Module()
        lower_extern_wrapper(declared, module)
        with self.assertRaisesRegex(TypeError, "calling convention"):
            lower_extern_wrapper(wrong_convention, module)

    def test_data_symbol_is_not_callable(self):
        module = ir.Module()
        ir.GlobalVariable(module, ir.IntType(32), "native_function")
        with self.assertRaisesRegex(TypeError, "data symbol"):
            lower_extern_wrapper(declared, module)

    def test_aggregate_signature_uses_the_existing_c_abi(self):
        module = ir.Module()
        first = ensure_ir(lower_extern_wrapper(pair_declared, module))
        self.assertIs(ensure_ir(lower_extern_wrapper(pair_declared, module)),
                      first)


if __name__ == "__main__":
    unittest.main()
