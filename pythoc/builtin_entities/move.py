"""
move() intrinsic for linear type ownership transfer

move(x) transfers ownership of a linear value, marking the source as consumed
and returning the value for use elsewhere.
"""
import ast
from .base import BuiltinFunction
from ..valueref import ValueRef
from ..logger import logger


def _fresh_move_value(value):
    from ..literal_protocol import (
        get_mapping_entries,
        get_sequence_elements,
        is_mapping_carrier,
        is_sequence_carrier,
        rebuild_mapping_carrier,
        rebuild_sequence_carrier,
    )

    if isinstance(value, ValueRef):
        return value.clone(var_name=None, linear_path=None)

    if is_sequence_carrier(value):
        fresh_elements = [_fresh_move_value(elem) for elem in get_sequence_elements(value)]
        return rebuild_sequence_carrier(value, fresh_elements)

    if is_mapping_carrier(value):
        fresh_entries = [
            (_fresh_move_value(key), _fresh_move_value(mapped_value))
            for key, mapped_value in get_mapping_entries(value)
        ]
        return rebuild_mapping_carrier(value, fresh_entries)

    return value


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
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
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
        
        fresh_value = _fresh_move_value(arg_value.value)
        return arg_value.clone(value=fresh_value, var_name=None, linear_path=None)
