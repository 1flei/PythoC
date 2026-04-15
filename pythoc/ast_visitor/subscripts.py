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

