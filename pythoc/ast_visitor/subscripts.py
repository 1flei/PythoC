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
        """Handle subscript operations with unified duck typing protocol
        
        Design principle (unified protocol):
        1. Get subscriptable object from node.value
        2. ALWAYS visit index (no special case for type subscripts)
        3. Delegate to type_hint.handle_subscript(visitor, base, index, node)
        
        Key insight: The dispatch is determined by base.type_hint:
        - base.type_hint is PythonType -> type subscript (ptr[i32], struct[x: i32])
        - base.type_hint is PC type instance -> value subscript (arr[0], p[i])
        
        Protocol implementers:
            - PythonType.handle_subscript: always type subscript
            - PCType.handle_subscript (array, ptr, struct): always value subscript
        """
        # Special case: union/enum varargs access (args[i])
        # Struct varargs are now handled as normal struct, no special case needed
        if isinstance(node.value, ast.Name):
            var_name = node.value.id
            if hasattr(self, 'current_varargs_info') and self.current_varargs_info:
                if var_name == self.current_varargs_info['name']:
                    # Only handle union/enum varargs specially (they use va_list)
                    if self.current_varargs_info['kind'] in ('union', 'enum'):
                        return self._handle_varargs_subscript(node)
        
        # Get the subscriptable object (evaluates node.value)
        result = self.visit_expression(node.value)
        
        # ALWAYS visit index - unified handling
        index = self.visit_expression(node.slice)
        
        # Delegate to type_hint's handle_subscript
        if result.type_hint and hasattr(result.type_hint, 'handle_subscript'):
            return result.type_hint.handle_subscript(self, result, index, node)
        
        logger.error(f"Object does not support subscripting: valueref: {result}", node=node, exc_type=TypeError)
    
    def visit_Attribute(self, node: ast.Attribute):
        """Handle attribute access with unified duck typing protocol
        
        Design principle (unified protocol, similar to visit_Call and visit_Subscript):
        1. Get the object from node.value
        2. Pre-evaluate base (node.value) and extract attr_name (node.attr)
        3. Delegate to handle_attribute(visitor, base, attr_name, node)
        
        All attribute-accessible types implement: handle_attribute(visitor, base, attr_name, node) -> ValueRef
        where base is a pre-evaluated ValueRef and attr_name is a string.
        
        Protocol implementers:
            - Builtin types (struct, union): type class with handle_attribute
            - Python types: PythonType instance with handle_attribute
            - Pointer types (ptr[struct]): ptr type with handle_attribute delegation
        """
        # Struct varargs are now normal structs, no special handling needed
        
        # Evaluate the base expression
        result = self.visit_expression(node.value)
        
        # Special case: enum class attribute access (e.g., MyEnum.VARIANT)
        # Delegate to handle_attribute for proper type handling
        if (result.kind == 'python' and 
            isinstance(result.value, type) and 
            getattr(result.value, '_is_enum', False)):
            # Use handle_attribute if available
            attr_name = node.attr
            if hasattr(result.value, 'handle_attribute') and callable(result.value.handle_attribute):
                return result.value.handle_attribute(self, result, attr_name, node)
            logger.error(f"Enum '{result.value.__name__}' has no attribute '{attr_name}' or handle_attribute method",
                        node=node, exc_type=AttributeError)
        
        # Extract the object that implements handle_attribute
        attributable = None
        if hasattr(result.value, 'handle_attribute'):
            attributable = result.value
        elif result.type_hint and hasattr(result.type_hint, 'handle_attribute'):
            attributable = result.type_hint
        else:
            logger.error(f"Object does not support attribute access: valueref: {result}", node=node, exc_type=TypeError)
        
        # Extract attribute name
        attr_name = node.attr
        
        return attributable.handle_attribute(self, result, attr_name, node)
    
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
            
            # CRITICAL: Bitcast and va_start MUST be in entry block to dominate all uses
            # Save current builder position
            current_block = self.builder.block
            
            # Move to entry block and position after allocas
            entry_block = self.current_function.entry_basic_block
            # Find the last alloca instruction in entry block
            last_alloca = None
            for instr in entry_block.instructions:
                if isinstance(instr, ir.AllocaInstr):
                    last_alloca = instr
            
            if last_alloca:
                # Position after the last alloca
                self.builder.position_after(last_alloca)
            else:
                # No allocas, position at start
                self.builder.position_at_start(entry_block)
            
            # Bitcast va_list to i8* in entry block
            va_list_i8 = self.builder.bitcast(va_list, va_list_type)
            self.builder.call(va_start, [va_list_i8])
            
            # Restore builder position
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

