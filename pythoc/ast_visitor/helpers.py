"""
Helpers mixin for LLVMIRVisitor
"""

import ast
import builtins
from typing import Optional, Any
from llvmlite import ir
from ..valueref import ValueRef, ensure_ir, wrap_value, get_type, get_type_hint
from ..builtin_entities import (
    i8, i16, i32, i64,
    u8, u16, u32, u64,
    f32, f64, ptr,
    sizeof, nullptr,
    get_builtin_entity,
    is_builtin_type,
    is_builtin_function,
    TYPE_MAP,
)
from ..builtin_entities import bool as pc_bool
from ..registry import get_unified_registry, infer_struct_from_access
from ..logger import logger


class HelpersMixin:
    """Mixin containing helpers-related visitor methods"""
    
    def _get_pow_intrinsic(self, type_):
        """Get pow intrinsic for the given type"""
        if isinstance(type_, ir.FloatType):
            intrinsic_name = "llvm.pow.f32"
        elif isinstance(type_, ir.DoubleType):
            intrinsic_name = "llvm.pow.f64"
        else:
            # For integers, convert to double, pow, then back
            intrinsic_name = "llvm.pow.f64"
        
        try:
            return self.module.get_global(intrinsic_name)
        except KeyError:
            # Declare the intrinsic
            func_type = ir.FunctionType(type_, [type_, type_])
            return ir.Function(self.module, func_type, intrinsic_name)
    
    def _create_string_constant(self, value: str):
        """Create a global string constant"""
        # Convert string to byte array
        byte_array = bytearray(value.encode('utf-8'))
        byte_array.append(0)  # Null terminator
        
        # Create global constant
        char_array_type = ir.ArrayType(ir.IntType(8), len(byte_array))
        global_str = ir.GlobalVariable(self.module, char_array_type, f"str_{len(self.module.globals)}")
        global_str.linkage = 'internal'
        global_str.global_constant = True
        global_str.initializer = ir.Constant(char_array_type, byte_array)
        
        # Return pointer to first element
        gep_result = self.builder.gep(ensure_ir(global_str), [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)])
        return gep_result
    
    def _get_floor_intrinsic(self, type_):
        """Get floor intrinsic for the given type"""
        if isinstance(type_, ir.FloatType):
            intrinsic_name = "llvm.floor.f32"
        elif isinstance(type_, ir.DoubleType):
            intrinsic_name = "llvm.floor.f64"
        else:
            intrinsic_name = "llvm.floor.f64"
        
        try:
            return self.module.get_global(intrinsic_name)
        except KeyError:
            # Declare the intrinsic
            func_type = ir.FunctionType(type_, [type_])
            return ir.Function(self.module, func_type, intrinsic_name)
    

    def _create_alloca_in_entry(self, llvm_type, name):
        """Create an alloca instruction in the entry block of the current function"""
        # Save current builder position
        current_block = self.builder.block
        
        # Move to entry block and position at start (before any existing instructions)
        entry_block = self.current_function.entry_basic_block
        self.builder.position_at_start(entry_block)
        
        # Create alloca instruction
        alloca = self.builder.alloca(llvm_type, name=name)
        
        # Restore builder position to the original block
        self.builder.position_at_end(current_block)
        
        return alloca
    

    def _align_to(self, size: int, alignment: int) -> int:
        """Align size to the specified alignment boundary"""
        return (size + alignment - 1) & ~(alignment - 1)
