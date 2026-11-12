"""Extern symbol reuse, signature checks, and Python fallback calls."""

import ctypes
import unittest

from llvmlite import ir

from pythoc import extern, f16, f32, i24, i32, i64, i128, struct, u8, u64
from pythoc.builder import LLVMBuilder
from pythoc.callable_lowering import lower_extern_wrapper
from pythoc.decorators.extern import ExternFunctionWrapper
from pythoc.libc import stdlib, string
from pythoc.libc.math import sqrt
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

    def test_calling_convention_is_a_pythoc_signature_property(self):
        from pythoc.callable_lowering import extern_callable_type
        from pythoc.type_descriptor import same_type, can_convert
        from pythoc.type_id import get_type_id
        ordinary, _, _ = extern_callable_type(declared)
        alternate, _, _ = extern_callable_type(wrong_convention)
        self.assertEqual(ordinary.calling_convention, 'cdecl')
        self.assertEqual(alternate.calling_convention, 'stdcall')
        self.assertFalse(same_type(ordinary, alternate))
        self.assertNotEqual(get_type_id(ordinary), get_type_id(alternate))
        for mode in ('implicit', 'explicit'):
            self.assertFalse(can_convert(ordinary, alternate, mode=mode))
            self.assertFalse(can_convert(alternate, ordinary, mode=mode))

    def test_unknown_convention_and_variadic_stdcall_are_rejected(self):
        from pythoc.callable_lowering import extern_callable_type
        with self.assertRaisesRegex(TypeError, 'Unsupported calling convention'):
            extern(calling_convention='typo')(lambda: None)

        @extern(calling_convention='stdcall')
        def variadic(value: i32, *args) -> i32:
            pass

        with self.assertRaisesRegex(TypeError, 'require the cdecl'):
            extern_callable_type(variadic)

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


@extern(lib='c', name='abs')
def absolute(value: i32 = -17) -> i32:
    pass


Division = struct['quot': i32, 'rem': i32]


@extern(lib='c', name='div')
def integer_divide(numerator: i32, denominator: i32) -> Division:
    pass


class TestExternPythonCall(unittest.TestCase):
    def test_declared_integer_return_preserves_all_bits(self):
        self.assertEqual(stdlib.strtoull(b'18446744073709551615', None, 10), (1 << 64) - 1)
        self.assertEqual(stdlib.strtoll(b'-4294967297', None, 10), -4294967297)

    def test_float_return_and_pointer_output(self):
        buffer = ctypes.create_string_buffer(b'1.25tail')
        end = ctypes.c_void_p()
        self.assertEqual(stdlib.strtod(buffer, ctypes.byref(end)), 1.25)
        self.assertEqual(end.value, ctypes.addressof(buffer) + 4)

    def test_math_library_name_is_resolved_to_a_loadable_image(self):
        self.assertEqual(sqrt(2.25), 1.5)

    def test_natural_aggregate_return_uses_the_c_signature(self):
        result = integer_divide(-17, 5)
        self.assertEqual((result.quot, result.rem), (-3, -2))

    def test_pointer_returns_can_be_used_without_truncation(self):
        buffer = ctypes.create_string_buffer(b'abcdef')
        address = string.strchr(buffer, ord('d'))
        self.assertEqual(address, ctypes.addressof(buffer) + 3)

    def test_python_binding_defaults_keywords_and_errors(self):
        self.assertEqual(absolute(), 17)
        self.assertEqual(absolute(value=-23), 23)
        for args, kwargs in [((1, 2), {}), ((1,), {'value': 2}), ((), {'unknown': 1})]:
            with self.subTest(args=args, kwargs=kwargs):
                with self.assertRaises(TypeError):
                    absolute(*args, **kwargs)

    def test_variadic_values_use_c_default_promotions(self):
        for value, ctype, expected in (
            (u8(255), ctypes.c_int32, 255),
            (f32(1.25), ctypes.c_double, 1.25),
            (u64(4294967297), ctypes.c_uint64, 4294967297),
        ):
            with self.subTest(value=value):
                promoted = ExternFunctionWrapper._ctypes_vararg(value)
                self.assertIsInstance(promoted, ctype)
                self.assertEqual(promoted.value, expected)

    def test_plain_python_float_is_a_double_vararg(self):
        promoted = ExternFunctionWrapper._ctypes_vararg(1.25)
        self.assertIsInstance(promoted, ctypes.c_double)
        self.assertEqual(promoted.value, 1.25)

    def test_unsupported_carriers_fail_before_symbol_lookup(self):
        for pc_type in (f16, i24, i128, struct[i24], struct[i128]):
            with self.subTest(pc_type=pc_type):
                @extern(lib='c', name='nonexistent_unsupported_symbol')
                def unsupported(value: pc_type) -> i32:
                    pass

                with self.assertRaises(TypeError):
                    unsupported(1)

    def test_long_and_size_t_declarations_match_the_host_abi(self):
        self.assertEqual(stdlib.strtol.return_type._bit_width, ctypes.sizeof(ctypes.c_long) * 8)
        self.assertEqual(stdlib.strtoul.return_type._bit_width, ctypes.sizeof(ctypes.c_ulong) * 8)
        self.assertFalse(stdlib.strtoul.return_type._is_signed)
        self.assertFalse(stdlib.strtoull.return_type._is_signed)
        size = dict(stdlib.malloc.param_types)['size']
        self.assertEqual(size._bit_width, ctypes.sizeof(ctypes.c_size_t) * 8)
        self.assertFalse(size._is_signed)


if __name__ == "__main__":
    unittest.main()
