"""
Subscripts mixin for LLVMIRVisitor
"""

import ast
import builtins
from typing import Optional, Any
from llvmlite import ir
from ..valueref import ValueRef, ensure_ir, wrap_value, get_type, get_type_hint, extract_constant_index
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


class SubscriptsMixin:
    """Mixin containing subscripts-related visitor methods"""
    
    def visit_Subscript(self, node: ast.Subscript):
        """Handle subscript operations with unified duck typing protocol.

        The visitor stays at the AST layer: it evaluates the base and index,
        while `ValueRefDispatcher.handle_subscript()` performs protocol
        normalization and dispatch.
        """
        # Special case: union/enum varargs access (args[i])
        # Struct varargs are now handled as normal struct, no special case needed
        if isinstance(node.value, ast.Name):
            var_name = node.value.id
            if self.current_varargs_info:
                if var_name == self.current_varargs_info['name']:
                    # Only handle union/enum varargs specially (they use va_list)
                    if self.current_varargs_info['kind'] in ('union', 'enum'):
                        return self._handle_varargs_subscript(node)

        result = self.visit_expression(node.value)
        index = self.visit_rvalue_expression(node.slice)
        return self.value_dispatcher.handle_subscript(result, index, node)
    
    def visit_Attribute(self, node: ast.Attribute):
        """Handle attribute access with unified duck typing protocol.

        The visitor only evaluates the base expression. `ValueRefDispatcher`
        handles protocol lookup, special enum cases, and the final dispatch to
        `handle_attribute`.
        """
        # Struct varargs are now normal structs, no special handling needed
        
        # Evaluate the base expression
        result = self.visit_expression(node.value)

        return self.value_dispatcher.handle_attribute(result, node.attr, node)
    
    def _handle_varargs_subscript(self, node: ast.Subscript) -> ValueRef:
        """Handle args[i] for union/enum varargs (va_list based)
        
        Generates LLVM va_arg instruction to access varargs parameters.
        Struct varargs are handled as normal structs, not through this path.
        """
        varargs_info = self.current_varargs_info
        
        # Evaluate index
        index_val = self.visit_expression(node.slice)
        
        # Extract constant index (supports both PythonType and ir.Constant)
        index = extract_constant_index(index_val, "varargs subscript")
        
        if index < 0:
            logger.error(f"Varargs index must be a non-negative integer, got {index}",
                        node=node, exc_type=IndexError)
        # Check if we have type information for this index
        element_types = varargs_info['element_types']
        if not element_types:
            logger.error("Cannot access union varargs without type information", node=node, exc_type=TypeError)
        
        # For union varargs, all elements have the same possible types
        # Use the first type in the list as default, or cycle through if multiple
        if len(element_types) == 1:
            # Homogeneous varargs: all elements are the same type
            target_pc_type = element_types[0]
        else:
            # Heterogeneous: cycle through types (simple strategy)
            target_pc_type = element_types[index % len(element_types)]
        
        # Initialize va_list on first access
        if varargs_info['va_list'] is None:
            va_list_type = ir.PointerType(ir.IntType(8))
            va_list = self._create_alloca_in_entry(va_list_type, "va_list")
            
            # Call llvm.va_start
            va_start_name = "llvm.va_start"
            try:
                va_start = self.module.get_global(va_start_name)
            except KeyError:
                # Declare va_start intrinsic
                va_start_type = ir.FunctionType(ir.VoidType(), [va_list_type])
                va_start = ir.Function(self.module, va_start_type, va_start_name)
            
            # Bitcast va_list to i8* and call va_start in entry block
            # Must be in entry block to dominate all uses.
            # Use the same position_at_start/position_at_end pattern as _create_alloca_in_entry.
            current_block = self.builder.block
            entry_block = self.builder.entry_block
            self.builder.position_at_start(entry_block)
            va_list_i8 = self.builder.bitcast(va_list, va_list_type)
            self.builder.call(va_start, [va_list_i8])
            self.builder.position_at_end(current_block)
            
            varargs_info['va_list'] = va_list
            varargs_info['va_list_i8'] = va_list_i8
            varargs_info['access_count'] = 0
        
        # Generate va_arg for this access
        # Note: LLVM's va_arg is a bit tricky - we use the higher-level approach
        target_llvm_type = target_pc_type.get_llvm_type(self.module.context)
        
        # Use LLVM's va_arg instruction (through llvmlite)
        # For now, we'll use a simpler approach: sequential access
        access_count = varargs_info.get('access_count', 0)
        
        if index != access_count:
            logger.error(f"Union varargs must be accessed sequentially. Expected args[{access_count}], got args[{index}]",
                        node=node, exc_type=TypeError)
        
        # Read the argument using platform-specific varargs handling
        # This is a simplified version - full implementation would need platform ABI
        value = self._va_arg(varargs_info['va_list_i8'], target_llvm_type)
        
        varargs_info['access_count'] = access_count + 1
        
        return wrap_value(value, kind="value", type_hint=target_pc_type)
    
    def _va_arg(self, va_list_i8, target_type):
        """Read an argument from va_list
        
        Simplified x86_64 implementation that reads from overflow_arg_area.
        Full implementation would handle register save area and different platforms.
        """
        # On x86_64, va_list is a struct with:
        # - gp_offset (i32)
        # - fp_offset (i32)  
        # - overflow_arg_area (i8*)
        # - reg_save_area (i8*)
        
        # For simplicity, we assume all args are in overflow_arg_area (offset 8)
        # This works for args beyond the first 6 integer/8 FP args
        
        # Cast va_list (i8*) to i8** to access overflow_arg_area
        # Structure layout: [i32, i32, i8*, i8*]
        # overflow_arg_area is at byte offset 8 (after two i32s)
        
        # Get pointer to overflow_arg_area field
        va_list_struct_ptr = self.builder.bitcast(va_list_i8, ir.PointerType(ir.IntType(8)))
        
        # Access overflow_arg_area at offset 8
        # First, get the va_list as i64* to access 64-bit aligned fields
        va_list_as_i64_ptr = self.builder.bitcast(va_list_i8, ir.PointerType(ir.IntType(64)))
        
        # overflow_arg_area is the second i64 field (index 1)
        overflow_ptr_field = self.builder.gep(
            va_list_as_i64_ptr,
            [ir.Constant(ir.IntType(32), 1)],
            inbounds=False
        )
        
        # Load the overflow_arg_area pointer (i64 that represents i8*)
        overflow_area_i64 = self.builder.load(overflow_ptr_field)
        
        # Convert i64 to i8*
        overflow_area = self.builder.inttoptr(overflow_area_i64, ir.PointerType(ir.IntType(8)))
        
        # Cast to target type pointer
        typed_ptr = self.builder.bitcast(overflow_area, ir.PointerType(target_type))
        
        # Load the value
        value = self.builder.load(typed_ptr)
        
        # Advance overflow_arg_area pointer
        # Calculate size of target type (aligned to 8 bytes on x86_64)
        # Simple size calculation based on type
        if isinstance(target_type, ir.IntType):
            type_size = (target_type.width + 7) // 8
        elif isinstance(target_type, ir.DoubleType):
            type_size = 8
        elif isinstance(target_type, ir.FloatType):
            type_size = 4
        elif isinstance(target_type, ir.PointerType):
            type_size = 8
        else:
            type_size = 8  # Default
        
        aligned_size = (type_size + 7) & ~7  # Round up to 8-byte boundary
        
        new_overflow_i8 = self.builder.gep(
            overflow_area,
            [ir.Constant(ir.IntType(32), aligned_size)],
            inbounds=False
        )
        
        # Convert back to i64 and store
        new_overflow_i64 = self.builder.ptrtoint(new_overflow_i8, ir.IntType(64))
        self.builder.store(new_overflow_i64, overflow_ptr_field)
        
        return value

