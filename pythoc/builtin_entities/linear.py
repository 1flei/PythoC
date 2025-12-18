"""
Linear token type for compile-time resource tracking

Linear tokens are zero-size markers that enforce explicit resource cleanup.
They must be consumed exactly once before leaving scope.
"""

import ast
from llvmlite import ir
from .base import BuiltinType
from ..logger import logger


class linear(BuiltinType):
    """Linear token type - zero-size compile-time tracking marker
    
    Must be explicitly consumed before leaving scope.
    Cannot be copied or duplicated.
    Ownership transferred via move() or function returns.
    """
    _llvm_type = ir.LiteralStructType([])  # Empty struct = zero size
    _size_bytes = 0
    _is_linear = True
    
    @classmethod
    def get_name(cls) -> str:
        return 'linear'
    
    @classmethod
    def can_be_called(cls) -> bool:
        return True  # Can call linear() to create
    
    @classmethod
    def get_ctypes_type(cls):
        """Get ctypes type for linear (zero-size, returns None)."""
        return None
    
    @classmethod
    def get_llvm_type(cls, module_context=None) -> ir.Type:
        """Get LLVM type (zero-size empty struct)"""
        return ir.LiteralStructType([])
    
    @classmethod
    def handle_call(cls, visitor, func_ref, args, node: ast.Call):
        """Create a new linear token: linear()
        
        Args:
            visitor: AST visitor
            func_ref: ValueRef of the callable (linear type)
            args: Pre-evaluated arguments (should be empty)
            node: ast.Call node
        
        Returns:
            ValueRef containing zero-size token (will be tracked automatically)
        """
        if len(args) != 0:
            logger.error("linear() takes no arguments", node=node, exc_type=TypeError)
        
        # Return zero-size token (compile-time only)
        # Tracking happens automatically in visit_expression
        from ..valueref import wrap_value
        token_value = ir.Constant(ir.LiteralStructType([]), [])
        result = wrap_value(token_value, kind='value', type_hint=linear)
        
        return result
    
    @classmethod
    def is_linear(cls) -> bool:
        """Check if this is a linear type"""
        return True
