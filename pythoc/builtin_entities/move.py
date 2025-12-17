"""
move() intrinsic for linear type ownership transfer

move(x) transfers ownership of a linear value, marking the source as consumed
and returning the value for use elsewhere.
"""
import ast
from .base import BuiltinFunction
from ..valueref import wrap_value
from ..logger import logger


class move(BuiltinFunction):
    """move(x) -> x
    
    Transfer ownership of a linear value.
    
    This is an identity function at runtime (returns its argument unchanged),
    but signals to the linear type checker that ownership is being transferred.
    
    Works with:
    - linear tokens directly
    - Structs containing linear fields
    - Any type with linear components
    
    For non-linear types, move() is a no-op that just returns the value.
    
    Implementation note:
    The ownership transfer happens in two steps:
    1. visit_Call transfers ownership from the argument (marks source as consumed)
    2. move() returns a NEW ValueRef without var_name, so the assignment
       treats it as a fresh value (not a variable reference)
    """
    
    @classmethod
    def get_name(cls) -> str:
        return 'move'
    
    @classmethod
    def handle_call(cls, visitor, func_ref, args, node: ast.Call):
        """Handle move(value) call
        
        move() returns a NEW ValueRef without var_name tracking.
        This is critical because:
        1. visit_Call already transferred ownership from the argument
        2. The return value should be treated as a fresh value, not a variable
        3. This prevents double-consumption when the result is assigned
        
        Args:
            visitor: AST visitor
            func_ref: ValueRef of the callable (move function)
            args: Pre-evaluated arguments (ownership already transferred by visit_Call)
            node: ast.Call node
        
        Returns:
            New ValueRef with the same value but no var_name tracking
        """
        if len(args) != 1:
            logger.error("move() takes exactly 1 argument", node=node, exc_type=TypeError)
        
        arg_value = args[0]
        
        # Return a NEW ValueRef without var_name tracking
        # This ensures the assignment treats it as a fresh value, not a variable reference
        # The ownership has already been transferred from the source by visit_Call
        
        # Handle different kinds appropriately
        if arg_value.kind == 'address':
            return wrap_value(
                arg_value.value,
                kind='address',
                type_hint=arg_value.type_hint,
                address=arg_value.address
            )
        else:
            return wrap_value(
                arg_value.value,
                kind=arg_value.kind,
                type_hint=arg_value.type_hint
            )
        # Note: We intentionally do NOT copy var_name or linear_path
