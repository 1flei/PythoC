"""
defer intrinsic for deferred execution

defer(f, a, b, c) - Register f(a, b, c) to be called when the current block exits

Deferred calls are executed in FIFO order (first registered, first executed)
when the block exits via:
- Normal flow (end of block/scope)
- return (executes all defers from current scope up)
- break/continue (executes defers for loop scope)
- goto (executes defers when jumping out of scope)

Usage:
    @compile
    def example() -> void:
        defer(cleanup, resource)  # Will be called at block exit
        # ... do work ...
        # cleanup(resource) is called here automatically

Note:
- defer captures the arguments at registration time, not at execution time
- Deferred calls are scope-bound: they execute when their scope exits
- Multiple defers in same scope execute in FIFO order (first defer runs first)
"""
import ast
from .base import BuiltinFunction
from .types import void
from ..valueref import wrap_value, ValueRef
from ..logger import logger


def _init_defer_registry(visitor):
    """Initialize defer registry if not exists
    
    The defer registry tracks:
    - _defer_stack: List of (scope_depth, callable_obj, func_ref, args, node)
    """
    if not hasattr(visitor, '_defer_stack'):
        visitor._defer_stack = []


def _get_defers_for_scope(visitor, scope_depth: int):
    """Get all deferred calls for a specific scope depth (without removing)
    
    Returns list of (callable_obj, func_ref, args, node) tuples in FIFO order
    """
    _init_defer_registry(visitor)
    defers = []
    
    for item in visitor._defer_stack:
        depth, callable_obj, func_ref, args, node = item
        if depth == scope_depth:
            defers.append((callable_obj, func_ref, args, node))
    
    return defers


def _get_defers_for_scope_and_above(visitor, min_scope_depth: int):
    """Get all deferred calls for scopes >= min_scope_depth (without removing)
    
    Used when exiting multiple scopes (e.g., return from nested scope)
    Returns list of (scope_depth, callable_obj, func_ref, args, node) tuples in FIFO order
    """
    _init_defer_registry(visitor)
    defers = []
    
    for item in visitor._defer_stack:
        depth, callable_obj, func_ref, args, node = item
        if depth >= min_scope_depth:
            defers.append((depth, callable_obj, func_ref, args, node))
    
    return defers


def unregister_defers_for_scope(visitor, scope_depth: int):
    """Remove deferred calls for a specific scope from the stack (without executing)
    
    Called when exiting a block to clean up defers registered in that block.
    The defers should have already been emitted at all exit points.
    """
    _init_defer_registry(visitor)
    visitor._defer_stack = [
        item for item in visitor._defer_stack
        if item[0] != scope_depth
    ]
    logger.debug(f"Unregistered defers for scope {scope_depth}")


def emit_deferred_calls(visitor, scope_depth: int = None, all_scopes: bool = False):
    """Emit IR for deferred calls (without removing from stack)
    
    This generates the IR to execute defers at the current position.
    The defers remain in the stack until explicitly unregistered.
    
    Args:
        visitor: AST visitor
        scope_depth: Specific scope depth to emit defers for
        all_scopes: If True, emit all defers from scope 0 and above
    
    Deferred calls are emitted in FIFO order (first registered, first executed)
    """
    _init_defer_registry(visitor)
    
    if all_scopes:
        # Get all defers from scope 0 (function scope) and above
        defers = _get_defers_for_scope_and_above(visitor, 0)
    elif scope_depth is not None:
        # Get defers only for the specific scope
        defers = [
            (scope_depth, c, f, a, n) 
            for c, f, a, n in _get_defers_for_scope(visitor, scope_depth)
        ]
    else:
        return
    
    if not defers:
        return
    
    # Emit in FIFO order (list is already in registration order)
    for item in defers:
        if len(item) == 5:
            depth, callable_obj, func_ref, args, node = item
        else:
            callable_obj, func_ref, args, node = item
        
        # Generate the call
        _execute_single_defer(visitor, callable_obj, func_ref, args, node)


# Keep old name for backward compatibility during transition
def execute_deferred_calls(visitor, scope_depth: int = None, all_scopes: bool = False):
    """Emit and unregister deferred calls (legacy API)
    
    This is the old behavior that both emits and removes defers.
    New code should use emit_deferred_calls + unregister_defers_for_scope.
    """
    emit_deferred_calls(visitor, scope_depth=scope_depth, all_scopes=all_scopes)
    if all_scopes:
        # Clear all defers
        visitor._defer_stack = []
    elif scope_depth is not None:
        unregister_defers_for_scope(visitor, scope_depth)


def _execute_single_defer(visitor, callable_obj, func_ref: ValueRef, args: list, node: ast.AST):
    """Execute a single deferred call
    
    Uses the same handle_call protocol as visit_Call for consistency.
    
    Args:
        visitor: AST visitor
        callable_obj: The callable object with handle_call method
        func_ref: ValueRef of the function
        args: List of ValueRef arguments
        node: Original AST node for error reporting
    """
    cf = visitor._get_cf_builder()
    
    # Skip if block is terminated
    if cf.is_terminated():
        return
    
    # Use the standard handle_call protocol
    callable_obj.handle_call(visitor, func_ref, args, node)


def _get_callable_obj(func_arg: ValueRef):
    """Extract callable object from function argument
    
    Returns the object that implements handle_call protocol.
    """
    # Check if it's a Python value with handle_call
    if func_arg.is_python_value():
        py_val = func_arg.get_python_value()
        if hasattr(py_val, 'handle_call'):
            return py_val
    
    # Check value for handle_call (e.g., ExternFunctionWrapper, @compile wrapper)
    if hasattr(func_arg, 'value') and hasattr(func_arg.value, 'handle_call'):
        return func_arg.value
    
    # Check type_hint for handle_call (e.g., func type)
    if hasattr(func_arg, 'type_hint') and func_arg.type_hint and hasattr(func_arg.type_hint, 'handle_call'):
        return func_arg.type_hint
    
    return None


class defer(BuiltinFunction):
    """defer(f, *args) - Register a deferred call
    
    The call f(*args) will be executed when the current block exits.
    Arguments are evaluated at defer() time, not at execution time.
    
    Multiple defers in the same scope execute in FIFO order.
    """
    
    @classmethod
    def get_name(cls) -> str:
        return 'defer'
    
    @classmethod
    def handle_call(cls, visitor, func_ref, args, node: ast.Call):
        """Handle defer(f, *args) call
        
        This registers the deferred call but doesn't execute it.
        The actual execution happens at scope exit.
        
        Args:
            visitor: AST visitor
            func_ref: ValueRef of the defer function itself (not used)
            args: Pre-evaluated arguments [callable, arg1, arg2, ...]
            node: ast.Call node
        
        Returns:
            void
        """
        if len(args) < 1:
            logger.error(
                "defer() requires at least 1 argument (the function to call)",
                node=node, exc_type=TypeError
            )
        
        # First argument is the function to defer
        deferred_func = args[0]
        deferred_args = args[1:]  # Remaining arguments
        
        # Get callable object for later execution
        callable_obj = _get_callable_obj(deferred_func)
        if callable_obj is None:
            logger.error(
                f"defer() first argument must be callable, got {deferred_func}",
                node=node, exc_type=TypeError
            )
        
        # Initialize defer registry
        _init_defer_registry(visitor)
        
        # Register the deferred call with current scope depth
        visitor._defer_stack.append((
            visitor.scope_depth,
            callable_obj,
            deferred_func,
            deferred_args,
            node
        ))
        
        logger.debug(
            f"Registered deferred call at scope depth {visitor.scope_depth}: "
            f"{deferred_func} with {len(deferred_args)} args"
        )
        
        return wrap_value(None, kind='python', type_hint=void)


def check_defers_at_function_end(visitor, func_node):
    """Check that all deferred calls have been executed
    
    Called at the end of function compilation to ensure no defers are leaked.
    This should not happen if the implementation is correct.
    
    Args:
        visitor: The visitor instance
        func_node: The function AST node (for error location)
    """
    _init_defer_registry(visitor)
    
    if visitor._defer_stack:
        # This indicates a bug in the implementation
        logger.warning(
            f"Internal: {len(visitor._defer_stack)} deferred calls were not executed. "
            f"This may indicate a bug in the defer implementation.",
            node=func_node
        )
        # Clear the stack to avoid affecting other functions
        visitor._defer_stack = []
