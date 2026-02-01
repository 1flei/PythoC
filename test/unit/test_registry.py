"""
Unit tests for unified registry system
"""

import unittest
from llvmlite import ir
import ast

from pythoc.registry import (
    VariableInfo, FunctionInfo, StructInfo, UnifiedCompilationRegistry,
    get_unified_registry
)


class TestVariableInfo(unittest.TestCase):
    """Test VariableInfo dataclass"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.module = ir.Module(name="test_module")
        self.func_type = ir.FunctionType(ir.VoidType(), [])
        self.func = ir.Function(self.module, self.func_type, name="test_func")
        self.block = self.func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(self.block)
    
    def test_variable_info_creation(self):
        """Test basic VariableInfo creation"""
        from pythoc.valueref import ValueRef
        from pythoc.builtin_entities import i32
        
        alloca = self.builder.alloca(ir.IntType(32), name="x")
        value_ref = ValueRef(kind='address', value=alloca, type_hint=i32, address=alloca)
        var_info = VariableInfo(
            name="x",
            value_ref=value_ref,
            alloca=alloca
        )
        
        self.assertEqual(var_info.name, "x")
        self.assertEqual(var_info.alloca, alloca)
        self.assertEqual(var_info.type_hint, i32)
        self.assertEqual(var_info.llvm_type, ir.IntType(32))
    
    def test_variable_info_with_type_hint(self):
        """Test VariableInfo with type hint"""
        from pythoc.builtin_entities import i32
        from pythoc.valueref import ValueRef
        
        alloca = self.builder.alloca(ir.IntType(32), name="x")
        value_ref = ValueRef(kind='address', value=alloca, type_hint=i32, address=alloca)
        var_info = VariableInfo(
            name="x",
            value_ref=value_ref,
            alloca=alloca,
            source="annotation"
        )
        
        self.assertEqual(var_info.type_hint, i32)
        self.assertEqual(var_info.source, "annotation")


class TestStructInfo(unittest.TestCase):
    """Test StructInfo dataclass"""
    
    def test_struct_info_creation(self):
        """Test basic StructInfo creation"""
        fields = [("x", ir.IntType(32)), ("y", ir.IntType(32))]
        struct_info = StructInfo(
            name="Point",
            fields=fields,
            field_indices={"x": 0, "y": 1}
        )
        
        self.assertEqual(struct_info.name, "Point")
        self.assertEqual(len(struct_info.fields), 2)
        self.assertEqual(struct_info.get_field_index("x"), 0)
        self.assertEqual(struct_info.get_field_index("y"), 1)
    
    def test_struct_info_field_count(self):
        """Test get_field_count method"""
        fields = [("x", ir.IntType(32)), ("y", ir.IntType(32)), ("z", ir.IntType(32))]
        struct_info = StructInfo(name="Point3D", fields=fields)
        
        self.assertEqual(struct_info.get_field_count(), 3)
    
    def test_struct_info_field_names(self):
        """Test get_field_names method"""
        fields = [("x", ir.IntType(32)), ("y", ir.IntType(32))]
        struct_info = StructInfo(
            name="Point",
            fields=fields,
            field_indices={"x": 0, "y": 1}
        )
        
        names = struct_info.get_field_names()
        self.assertEqual(names, ["x", "y"])


class TestUnifiedRegistry(unittest.TestCase):
    """Test UnifiedCompilationRegistry functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.registry = UnifiedCompilationRegistry()
        self.module = ir.Module(name="test_module")
        self.func_type = ir.FunctionType(ir.VoidType(), [])
        self.func = ir.Function(self.module, self.func_type, name="test_func")
        self.block = self.func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(self.block)
    
    def test_variable_registration(self):
        """Test variable registration and lookup"""
        from pythoc.valueref import ValueRef
        from pythoc.builtin_entities import i32
        
        alloca = self.builder.alloca(ir.IntType(32), name="x")
        value_ref = ValueRef(kind='address', value=alloca, type_hint=i32, address=alloca)
        var_info = VariableInfo(name="x", value_ref=value_ref, alloca=alloca)
        
        var_registry = self.registry.get_variable_registry()
        var_registry.declare(var_info)
        
        retrieved = var_registry.lookup("x")
        self.assertEqual(retrieved, var_info)
    
    def test_variable_scope(self):
        """Test variable scope management"""
        from pythoc.valueref import ValueRef
        from pythoc.builtin_entities import i32
        
        var_registry = self.registry.get_variable_registry()
        
        alloca1 = self.builder.alloca(ir.IntType(32), name="x")
        value_ref1 = ValueRef(kind='address', value=alloca1, type_hint=i32, address=alloca1)
        var_info1 = VariableInfo(name="x", value_ref=value_ref1, alloca=alloca1, scope_level=0)
        
        var_registry.declare(var_info1, allow_shadow=True)
        var_registry.enter_scope()
        
        alloca2 = self.builder.alloca(ir.IntType(32), name="x")
        value_ref2 = ValueRef(kind='address', value=alloca2, type_hint=i32, address=alloca2)
        var_info2 = VariableInfo(name="x", value_ref=value_ref2, alloca=alloca2, scope_level=1)
        var_registry.declare(var_info2, allow_shadow=True)
        
        # Should get the inner scope variable
        retrieved = var_registry.lookup("x")
        self.assertEqual(retrieved, var_info2)
        
        var_registry.exit_scope()
        
        # Should get the outer scope variable
        retrieved = var_registry.lookup("x")
        self.assertEqual(retrieved, var_info1)
    
    def test_builtin_entity_registration(self):
        """Test builtin entity registration"""
        from pythoc.builtin_entities import i32
        
        self.registry.register_builtin_entity("i32", i32)
        
        retrieved = self.registry.get_builtin_entity("i32")
        self.assertEqual(retrieved, i32)
    
    def test_struct_registration(self):
        """Test struct registration"""
        fields = [("x", ir.IntType(32)), ("y", ir.IntType(32))]
        struct_info = StructInfo(
            name="Point",
            fields=fields,
            field_indices={"x": 0, "y": 1}
        )
        
        self.registry.register_struct(struct_info)
        
        self.assertTrue(self.registry.has_struct("Point"))
        retrieved = self.registry.get_struct("Point")
        self.assertEqual(retrieved, struct_info)
    
    def test_extern_function_registration(self):
        """Test extern function registration - deprecated, extern info now stored on wrapper"""
        # This test is kept for documentation purposes
        # In the new design, extern function info is stored on the wrapper object
        # (wrapper._extern_config) and passed directly to compiler._declare_extern_function
        pass

    def test_global_registry_singleton(self):
        """Test global registry singleton"""
        registry1 = get_unified_registry()
        registry2 = get_unified_registry()
        
        self.assertIs(registry1, registry2)


if __name__ == '__main__':
    unittest.main()
