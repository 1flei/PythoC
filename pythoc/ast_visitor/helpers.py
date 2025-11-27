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
    
    def _ensure_struct_pointer(self, pointer_value):
        """Ensure pointer_value ultimately refers to a struct pointer.

        Returns a tuple of (struct_pointer, struct_type). The pointer is loaded
        as needed to strip intermediate pointer layers.
        """
        ptr_type = get_type(pointer_value)
        if not isinstance(ptr_type, ir.PointerType):
            return ensure_ir(pointer_value), None
        
        current_pointer = ensure_ir(pointer_value)
        current_type = ptr_type
        
        while isinstance(current_type, ir.PointerType):
            pointee_type = current_type.pointee
            if isinstance(pointee_type, (ir.LiteralStructType, ir.IdentifiedStructType)):
                return current_pointer, pointee_type
            if isinstance(pointee_type, ir.PointerType):
                current_pointer = self.builder.load(current_pointer)
                current_type = current_pointer.type
                continue
            break
        
        # Handle the case where we have a single pointer to a struct
        if isinstance(current_type, ir.PointerType):
            pointee_type = current_type.pointee
            if isinstance(pointee_type, (ir.LiteralStructType, ir.IdentifiedStructType)):
                return current_pointer, pointee_type
            return current_pointer, pointee_type
        
        # If we still don't have a struct type, check if we have type hint information
        if hasattr(pointer_value, 'type_hint') and pointer_value.type_hint is not None:
            try:
                if isinstance(pointer_value.type_hint, (ir.LiteralStructType, ir.IdentifiedStructType)):
                    return current_pointer, pointer_value.type_hint
                
                if hasattr(pointer_value.type_hint, 'get_llvm_type'):
                    struct_type = pointer_value.type_hint.get_llvm_type(self.module.context)
                    if isinstance(struct_type, (ir.LiteralStructType, ir.IdentifiedStructType)):
                        return current_pointer, struct_type
                
                if hasattr(pointer_value.type_hint, '__name__'):
                    type_name = pointer_value.type_hint.__name__
                    struct_info = get_unified_registry().get_struct(type_name)
                    if struct_info and struct_info.llvm_type:
                        return current_pointer, struct_info.llvm_type
            except Exception:
                pass
        
        return ensure_ir(pointer_value), None
    
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
    

    def _get_struct_metadata(self, pointee_type, field_name=None):
        """Get struct metadata for a given LLVM type"""
        
        # Try to get struct metadata by name
        if isinstance(pointee_type, ir.IdentifiedStructType):
            struct_name = pointee_type.name.strip('"')
            struct_metadata = get_unified_registry().get_struct(struct_name)
            if struct_metadata:
                return struct_metadata
        
        # Fallback: infer from field access pattern
        if field_name:
            return get_unified_registry().infer_struct_from_access(pointee_type, field_name)
        
        return None

    



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
    

    def _get_type_alignment(self, field_type):
        """Get alignment for a field type"""
        from ..builtin_entities import get_type_alignment
        try:
            return get_type_alignment(field_type)
        except:
            return 4  # Default alignment
    

    def _get_type_size_and_alignment(self, type_str: str):
        """Get size and alignment for a type string"""
        from ..builtin_entities import get_type_size, get_type_alignment
        
        # Handle pointer types
        if type_str.startswith('ptr[') or type_str.endswith('*'):
            return (8, 8)  # 64-bit pointer
        
        # Try to get from builtin_entities
        try:
            size = get_type_size(type_str)
            alignment = get_type_alignment(type_str)
            return (size, alignment)
        except Exception:
            # Default fallback
            return (4, 4)
    

    def _align_to(self, size: int, alignment: int) -> int:
        """Align size to the specified alignment boundary"""
        return (size + alignment - 1) & ~(alignment - 1)
    

    def _get_or_declare_strlen(self):
        """Get or declare the strlen function"""
        try:
            return self.module.get_global('strlen')
        except KeyError:
            # Declare strlen: size_t strlen(const char* str)
            char_ptr_type = ir.PointerType(ir.IntType(8))
            strlen_type = ir.FunctionType(ir.IntType(64), [char_ptr_type])
            strlen_func = ir.Function(self.module, strlen_type, 'strlen')
            return strlen_func
    

    def _get_sqrt_intrinsic(self, float_type):
        """Get LLVM sqrt intrinsic for the given float type"""
        if isinstance(float_type, ir.FloatType):
            intrinsic_name = 'llvm.sqrt.f32'
        elif isinstance(float_type, ir.DoubleType):
            intrinsic_name = 'llvm.sqrt.f64'
        else:
            raise TypeError(f"sqrt intrinsic not available for type {float_type}")
        
        try:
            return self.module.get_global(intrinsic_name)
        except KeyError:
            func_type = ir.FunctionType(float_type, [float_type])
            return ir.Function(self.module, func_type, intrinsic_name)
