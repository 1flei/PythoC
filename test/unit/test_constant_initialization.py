"""Zero aggregates and scalar constant normalization."""

from types import SimpleNamespace
import unittest

from llvmlite import binding, ir

from pythoc import array, bool as pc_bool, f64, i32, ptr, struct, union
from pythoc.type_converter import TypeConverter
from pythoc.valueref import ensure_ir


class TestConstantInitialization(unittest.TestCase):
    def setUp(self):
        self.module = ir.Module(context=ir.Context())
        self.converter = TypeConverter(SimpleNamespace(module=self.module))

    def verify(self, value, target):
        module = ir.Module(context=ir.Context())
        converter = TypeConverter(SimpleNamespace(module=module))
        result = ensure_ir(converter._promote_python_to_pc(value, target))
        global_value = ir.GlobalVariable(module, result.type, "value")
        global_value.initializer = result
        binding.parse_assembly(str(module)).verify()
        return result

    def test_zero_shapes_all_form_valid_constants(self):
        targets = (
            array[i32, 2], array[ptr[i32], 2], array[array[i32, 2], 2],
            array[struct[i32, ptr[i32]], 2], struct[ptr[i32], i32],
            union[i32, f64],
        )
        for target in targets:
            with self.subTest(target=target):
                result = self.verify(0, target)
                self.assertIsNone(result.constant)

    def test_nonzero_scalar_requires_explicit_aggregate_values(self):
        for value in (7, 1.75, True):
            for target in (array[i32, 2], struct[i32, i32]):
                with self.subTest(value=value, target=target):
                    with self.assertRaisesRegex(TypeError, "explicit sequence"):
                        self.converter._promote_python_to_pc(value, target)

    def test_bool_to_wide_integer_is_numeric(self):
        self.assertEqual(self.verify(True, i32).constant, 1)

    def test_bool_target_is_preserved(self):
        self.assertEqual(self.verify(True, pc_bool).constant, 1)

    def test_bool_to_float_is_valid(self):
        self.assertEqual(self.verify(True, f64).constant, 1.0)


if __name__ == "__main__":
    unittest.main()
