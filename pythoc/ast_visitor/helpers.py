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
    ptr,
    sizeof, nullptr,
    get_builtin_entity,
    is_builtin_type,
    is_builtin_function,
    TYPE_MAP,
)
from ..builtin_entities import bool as pc_bool
from ..logger import logger


class HelpersMixin:
    """Mixin containing helpers-related visitor methods"""
    
    def _get_pow_intrinsic(self, type_):
        from ..llvm_extensions import intrinsic
        return intrinsic(self.module, 'llvm.pow', type_)

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
        
        # Return pointer to first element as a constant expression.  Python string
        # literals are compile-time constants, so the corresponding ptr[i8] value
        # can be folded and used in static/global initializers.
        gep_result = global_str.gep([ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)])
        return gep_result
    
    def _get_floor_intrinsic(self, type_):
        from ..llvm_extensions import intrinsic
        return intrinsic(self.module, 'llvm.floor', type_)

    def _create_alloca_in_entry(self, llvm_type, name):
        """Create an alloca instruction in the entry block of the current function"""
        # Save current builder position
        current_block = self.builder.block
        
        # Move to entry block and position at start (before any existing instructions)
        entry_block = self.builder.entry_block
        self.builder.position_at_start(entry_block)
        
        # Create alloca instruction
        alloca = self.builder.alloca(llvm_type, name=name)
        
        # Restore builder position to the original block
        self.builder.position_at_end(current_block)
        
        return alloca
    

    def _align_to(self, size: int, alignment: int) -> int:
        """Align size to the specified alignment boundary"""
        return (size + alignment - 1) & ~(alignment - 1)
