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
- All defer management is done through ScopeManager (no legacy _defer_stack)
"""
import ast
from .base import BuiltinFunction
from .types import void
from ..valueref import wrap_value, ValueRef
from ..logger import logger


# ============================================================================
# ScopeManager-based defer functions (current implementation)
# ============================================================================

def _get_defers_for_scope_via_manager(visitor, scope_depth: int):
    """Get defers for a specific scope depth from ScopeManager
    
    Returns list of (callable_obj, func_ref, args, node) tuples
    """
    if not hasattr(visitor, 'scope_manager'):
        return []
    
    for scope in visitor.scope_manager._scopes:
        if scope.depth == scope_depth:
            return [
                (d.callable_obj, d.func_ref, d.args, d.node)
                for d in scope.defers
            ]
    return []


def _get_all_defers_via_manager(visitor):
    """Get all defers from all scopes via ScopeManager
    
    Returns list of (depth, callable_obj, func_ref, args, node) tuples
    """
    if not hasattr(visitor, 'scope_manager'):
        return []
    
    result = []
    for scope in visitor.scope_manager._scopes:
        for d in scope.defers:
            result.append((scope.depth, d.callable_obj, d.func_ref, d.args, d.node))
    return result


def emit_deferred_calls(visitor, scope_depth: int = None, all_scopes: bool = False):
    """Emit IR for deferred calls via ScopeManager
    
    This generates the IR to execute defers at the current position.
    
    Args:
        visitor: AST visitor
        scope_depth: Specific scope depth to emit defers for
        all_scopes: If True, emit all defers from all scopes
    
    Deferred calls are emitted in FIFO order (first registered, first executed)
    """
    if not hasattr(visitor, 'scope_manager'):
        logger.error("emit_deferred_calls: visitor has no scope_manager", node=None)
        return
    
    if all_scopes:
        # Emit all defers from all scopes (innermost first for proper cleanup order)
        for scope in reversed(visitor.scope_manager._scopes):
            visitor.scope_manager._emit_defers_for_scope(scope)
    elif scope_depth is not None:
        # Find and emit defers for specific scope
        for scope in visitor.scope_manager._scopes:
            if scope.depth == scope_depth:
                visitor.scope_manager._emit_defers_for_scope(scope)
                break


def unregister_defers_for_scope(visitor, scope_depth: int):
    """Clear defers for a specific scope (they should already be emitted)
    
    This is called when exiting a scope to ensure defers don't get double-executed.
    With ScopeManager, defers are cleared when emitted, so this is mostly a no-op.
    """
    if not hasattr(visitor, 'scope_manager'):
        return
    
    for scope in visitor.scope_manager._scopes:
        if scope.depth == scope_depth:
            scope.defers.clear()
            logger.debug(f"Cleared defers for scope {scope_depth}")
            break


# Legacy API - now just delegates to ScopeManager
def execute_deferred_calls(visitor, scope_depth: int = None, all_scopes: bool = False):
    """Emit deferred calls (legacy API, delegates to emit_deferred_calls)"""
    emit_deferred_calls(visitor, scope_depth=scope_depth, all_scopes=all_scopes)


def _execute_single_defer(visitor, callable_obj, func_ref: ValueRef, args: list, node: ast.AST):
    """Execute a single deferred call
    
    Uses the same handle_call protocol as visit_Call for consistency.
    
    Args:
        visitor: AST visitor
        callable_obj: The callable object with handle_call method
        func_ref: ValueRef of the function
        args: List of ValueRef arguments
        node: Original AST node for error reporting
    
    Linear semantics:
    - Linear ownership is transferred HERE at defer execution time
    - NOT at defer registration time (visit_Call skips it for defer)
    - This matches the semantic that linear tokens are consumed when defer runs
    """
    cf = visitor._get_cf_builder()
    
    # Skip if block is terminated
    if cf.is_terminated():
        return
    
    logger.debug(f"_execute_single_defer: {callable_obj}")
    
    # Transfer linear ownership NOW at execution time
    # This is deferred from registration time to match the semantic
    # that linear tokens are consumed when the deferred call actually runs
    for arg in args:
        visitor._transfer_linear_ownership(arg, reason="deferred function argument", node=node)
    
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
    
    Linear type semantics:
    - Linear arguments are NOT consumed at defer registration time
    - Linear arguments are consumed when the defer actually executes
    - This allows linear tokens to be "held" by defer until scope exit
    """
    
    # Flag to indicate defer should skip linear transfer at registration
    # Linear transfer happens at execution time instead
    defer_linear_transfer = True
    
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
        
        # Register with ScopeManager (required - no fallback)
        if not hasattr(visitor, 'scope_manager') or visitor.scope_manager.current_scope is None:
            logger.error(
                "defer() called outside of any scope (no scope_manager)",
                node=node, exc_type=RuntimeError
            )
        
        visitor.scope_manager.register_defer(
            callable_obj, deferred_func, deferred_args, node
        )
        logger.debug(
            f"Registered deferred call via ScopeManager at scope depth {visitor.scope_manager.current_depth}: "
            f"{deferred_func} with {len(deferred_args)} args"
        )
        
        return wrap_value(None, kind='python', type_hint=void)


def check_defers_at_function_end(visitor, func_node):
    """Clean up defer tracking at function end
    
    Called at the end of function compilation.
    With ScopeManager, this is mostly a no-op since scopes handle cleanup.
    
    Args:
        visitor: The visitor instance
        func_node: The function AST node (for error location)
    """
    # ScopeManager handles cleanup automatically
    # This function is kept for API compatibility
    pass
