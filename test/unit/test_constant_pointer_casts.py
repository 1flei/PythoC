"""The same pointer conversions preserve link-time addresses and runtime values."""

from types import SimpleNamespace
import unittest

from llvmlite import binding, ir

from pythoc import func, i32, ptr, void
from pythoc.ast_visitor.control_flow_builder import ControlFlowBuilder
from pythoc.ast_visitor.pcir import VReg
from pythoc.builder.llvm_simple_builder import LLVMBuilder
from pythoc.ir_helpers import is_link_time_constant
from pythoc.type_converter import TypeConverter
from pythoc.valueref import ensure_ir, wrap_value


class ConstantPointerCastsTest(unittest.TestCase):
    def setUp(self):
        self.module = ir.Module(context=ir.Context())
        self.function = ir.Function(self.module,
            ir.FunctionType(ir.PointerType(ir.IntType(8)), [ir.PointerType(ir.IntType(32))]),
            'fixture')
        self.real = LLVMBuilder(ir.IRBuilder(self.function.append_basic_block('entry')))
        self.visitor = SimpleNamespace(module=self.module, current_function=self.function)
        self.recorded = ControlFlowBuilder(self.real, self.visitor, 'fixture')
        self.visitor.builder = self.recorded
        self.converter = TypeConverter(self.visitor)

    def verify_static_pointer(self, value):
        self.assertTrue(is_link_time_constant(value))
        variable = ir.GlobalVariable(self.module, value.type, 'alias')
        variable.initializer = value
        self.real.ret(ir.Constant(self.function.function_type.return_type, None))
        binding.parse_assembly(str(self.module)).verify()

    def test_function_address_survives_conversion_before_recording(self):
        function = ir.Function(self.module, ir.FunctionType(ir.IntType(32), [ir.IntType(32)]),
                               'external_function')
        value = wrap_value(function, kind='value', type_hint=func[i32, i32])
        result = ensure_ir(self.converter.convert(value, ptr[void]))
        self.assertFalse(isinstance(result, VReg))
        self.verify_static_pointer(result)

    def test_data_address_survives_conversion_before_recording(self):
        storage = ir.GlobalVariable(self.module, ir.IntType(32), 'counter')
        storage.initializer = ir.Constant(ir.IntType(32), 23)
        value = wrap_value(storage, kind='value', type_hint=ptr[i32])
        result = ensure_ir(self.converter.convert(value, ptr[void]))
        self.verify_static_pointer(result)

    def test_runtime_pointer_remains_a_recorded_conversion(self):
        value = wrap_value(self.function.args[0], kind='value', type_hint=ptr[i32])
        result = ensure_ir(self.converter.convert(value, ptr[void]))
        self.assertIsInstance(result, VReg)
        self.assertFalse(is_link_time_constant(result))
        block = self.recorded.cfg.blocks[self.recorded.cfg.entry_id]
        self.assertEqual([instruction.op for instruction in block.pcir], ['bitcast'])

    def test_null_pointer_stays_constant(self):
        value = ir.Constant(ir.PointerType(ir.IntType(32)), None)
        self.verify_static_pointer(self.recorded.bitcast(value, ir.PointerType(ir.IntType(8))))

    def test_direct_backend_preserves_constant_and_runtime_casts(self):
        storage = ir.GlobalVariable(self.module, ir.IntType(32), 'counter')
        storage.initializer = ir.Constant(ir.IntType(32), 23)
        constant = self.real.bitcast(storage, ir.PointerType(ir.IntType(8)))
        self.assertTrue(is_link_time_constant(constant))
        runtime = self.real.bitcast(self.function.args[0], ir.PointerType(ir.IntType(8)))
        self.assertFalse(is_link_time_constant(runtime))
        self.real.ret(runtime)
        binding.parse_assembly(str(self.module)).verify()


if __name__ == '__main__':
    unittest.main()
