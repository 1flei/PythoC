"""
Unified Scope Manager for pythoc

This module provides unified scope management that combines:
- Variable registry (var_registry)
- Defer stack management
- Linear type tracking
- Loop context (break/continue targets)

The key insight is that defer execution time == variable lifetime end.
When a scope exits:
1. Deferred calls for that scope are executed
2. Linear tokens must have been consumed
3. Variables go out of scope

This replaces the separate scope_depth, _defer_stack, and var_registry.enter_scope()/exit_scope()
with a single unified mechanism.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import ast

from .logger import logger
from .valueref import ValueRef


class ScopeType(Enum):
    """Type of scope for debugging and special handling"""
    FUNCTION = auto()   # Function body
    LOOP = auto()       # Loop body (while, for)
    IF = auto()         # If/else block
    LABEL = auto()      # Scoped label block
    MATCH = auto()      # Match case block


@dataclass
class DeferInfo:
    """Information about a deferred call"""
    callable_obj: Any           # Object with handle_call method
    func_ref: ValueRef          # Function reference
    args: List[ValueRef]        # Arguments (captured at registration)
    node: ast.AST               # AST node for error reporting


@dataclass
class Scope:
    """A single scope in the scope stack
    
    Contains all information needed for scope management:
    - Variables declared in this scope
    - Deferred calls registered in this scope
    - Linear tokens created in this scope
    - Loop targets (for break/continue)
    """
    depth: int
    scope_type: ScopeType
    
    # Variables declared in this scope (name -> VariableInfo)
    variables: Dict[str, Any] = field(default_factory=dict)
    
    # Deferred calls registered in this scope
    defers: List[DeferInfo] = field(default_factory=list)
    
    # For loops: targets for break/continue
    continue_target: Optional[Any] = None  # Block to jump to on continue
    break_target: Optional[Any] = None     # Block to jump to on break
    
    def __repr__(self):
        return f"Scope(depth={self.depth}, type={self.scope_type.name}, vars={len(self.variables)}, defers={len(self.defers)})"


class ScopeManager:
    """Unified scope manager for variables, defers, and linear types
    
    This replaces the separate mechanisms:
    - var_registry.enter_scope() / exit_scope()
    - scope_depth tracking
    - _defer_stack management
    - loop_scope_stack
    
    Usage:
        with scope_manager.scope(ScopeType.IF) as scope:
            # Variables, defers automatically managed
            ...
        # Defers executed, linear checked, scope cleaned up
    
    Or manually:
        scope = scope_manager.enter_scope(ScopeType.LOOP)
        scope.continue_target = header_block
        scope.break_target = exit_block
        try:
            # ... loop body ...
        finally:
            scope_manager.exit_scope(cf)
    """
    
    def __init__(self, var_registry: Any):
        """Initialize scope manager
        
        Args:
            var_registry: The VariableRegistry to sync with
        """
        self._var_registry = var_registry
        self._scopes: List[Scope] = []
        self._visitor: Optional[Any] = None  # Set by visitor
    
    def set_visitor(self, visitor: Any):
        """Set the visitor for defer execution"""
        self._visitor = visitor
    
    @property
    def current_depth(self) -> int:
        """Get current scope depth"""
        return len(self._scopes)
    
    @property
    def current_scope(self) -> Optional[Scope]:
        """Get current scope"""
        return self._scopes[-1] if self._scopes else None
    
    def enter_scope(self, scope_type: ScopeType, 
                    continue_target: Any = None,
                    break_target: Any = None) -> Scope:
        """Enter a new scope
        
        Args:
            scope_type: Type of scope (FUNCTION, LOOP, IF, etc.)
            continue_target: For loops, the block to jump to on continue
            break_target: For loops, the block to jump to on break
        
        Returns:
            The new Scope object
        """
        # Sync with var_registry
        self._var_registry.enter_scope()
        
        scope = Scope(
            depth=len(self._scopes) + 1,
            scope_type=scope_type,
            continue_target=continue_target,
            break_target=break_target
        )
        self._scopes.append(scope)
        
        logger.debug(f"Entered scope {scope}")
        return scope
    
    def exit_scope(self, cf: Any, check_linear: bool = True,
                   node: Optional[ast.AST] = None) -> Scope:
        """Exit current scope
        
        This:
        1. Emits deferred calls (if block not terminated)
        2. Checks linear tokens consumed (if check_linear=True)
        3. Removes variables from registry
        
        Args:
            cf: ControlFlowBuilder for checking termination
            check_linear: Whether to check linear tokens (False for early exits)
            node: Optional AST node for error reporting
        
        Returns:
            The exited Scope
        """
        if not self._scopes:
            raise RuntimeError("Cannot exit scope: no scope to exit")
        
        scope = self._scopes.pop()
        
        # 1. Emit deferred calls (if not terminated)
        if not cf.is_terminated():
            self._emit_defers_for_scope(scope)
        
        # 2. Check linear tokens (optional)
        if check_linear and not cf.is_terminated():
            self._check_linear_tokens(scope, node)
        
        # 3. Sync with var_registry
        self._var_registry.exit_scope()
        
        logger.debug(f"Exited scope {scope}")
        return scope
    
    def exit_scopes_to(self, target_depth: int, cf: Any, emit_defers: bool = True) -> List[Scope]:
        """Exit scopes down to target depth (for break/continue/goto)
        
        This emits defers for all scopes being exited but does NOT check linear
        tokens (they will be checked by the normal scope exit path).
        
        Args:
            target_depth: The depth to exit to (exclusive - this depth is NOT exited)
            cf: ControlFlowBuilder
            emit_defers: Whether to emit defers (True for break/continue, may vary for goto)
        
        Returns:
            List of exited scopes
        """
        exited = []
        
        logger.debug(f"exit_scopes_to: target_depth={target_depth}, scopes={[s.depth for s in self._scopes]}")
        
        # Emit defers from current scope down to target (not including target)
        if emit_defers and not cf.is_terminated():
            for scope in reversed(self._scopes):
                logger.debug(f"  checking scope depth={scope.depth}, type={scope.scope_type.name}, defers={len(scope.defers)}")
                if scope.depth <= target_depth:
                    logger.debug(f"  stopping at scope depth={scope.depth}")
                    break
                self._emit_defers_for_scope(scope)
        
        # Note: We don't actually pop scopes here because the control flow
        # (break/continue) will jump out, and the finally blocks will handle
        # the actual scope cleanup
        
        return exited
    
    def emit_all_defers(self, cf: Any):
        """Emit all deferred calls (for return)
        
        Emits defers from all scopes, innermost first.
        """
        if cf.is_terminated():
            return
        
        for scope in reversed(self._scopes):
            self._emit_defers_for_scope(scope)
    
    def _emit_defers_for_scope(self, scope: Scope):
        """Emit deferred calls for a single scope
        
        The defers list is NOT cleared - multiple code paths may need to emit
        the same defers (e.g., goto path and normal exit path are mutually
        exclusive at runtime but both need IR generated at compile time).
        """
        if not self._visitor:
            return
        
        cf = self._visitor._get_cf_builder()
        
        # Do NOT clear defers - they may be needed by other code paths
        defers_to_emit = scope.defers[:]
        
        executed_count = 0
        for defer_info in defers_to_emit:
            if cf.is_terminated():
                logger.debug(f"  cf is terminated, stopping defer emit")
                break
            
            logger.debug(f"  emitting defer: {defer_info.callable_obj}")
            
            # Transfer linear ownership at execution time
            for arg in defer_info.args:
                self._visitor._transfer_linear_ownership(
                    arg, reason="deferred function argument", node=defer_info.node
                )
            
            # Execute the deferred call
            defer_info.callable_obj.handle_call(
                self._visitor, defer_info.func_ref, defer_info.args, defer_info.node
            )
            executed_count += 1
        
        logger.debug(f"Emitted {executed_count}/{len(defers_to_emit)} defers for scope depth {scope.depth}")
    
    def _check_linear_tokens(self, scope: Scope, node: Optional[ast.AST] = None):
        """Check that all linear tokens in scope are consumed
        
        This checks that all linear tokens declared at this scope depth
        have been consumed before the scope exits.
        
        Args:
            scope: The scope being exited
            node: Optional AST node for error reporting
        """
        if self._visitor is None:
            return
        
        unconsumed = []
        scope_depth = scope.depth
        
        # Check variables in current scope
        for var_info in self._var_registry.get_all_in_current_scope():
            if var_info.linear_scope_depth == scope_depth:
                for path, state in var_info.linear_states.items():
                    if state == 'active':
                        path_str = self._visitor._format_linear_path(
                            var_info.name, path, var_info.type_hint
                        )
                        actual_line = self._visitor._get_actual_line_number(var_info.line_number)
                        unconsumed.append(f"'{path_str}' (declared at line {actual_line})")
        
        if unconsumed:
            logger.error(
                f"Linear tokens not consumed before scope exit: {', '.join(unconsumed)}",
                node=node
            )
    
    def register_defer(self, callable_obj: Any, func_ref: ValueRef, 
                       args: List[ValueRef], node: ast.AST):
        """Register a deferred call in the current scope
        
        Args:
            callable_obj: Object with handle_call method
            func_ref: Function reference
            args: Arguments (captured now, used at execution)
            node: AST node for error reporting
        """
        if not self._scopes:
            raise RuntimeError("Cannot register defer: no scope")
        
        defer_info = DeferInfo(
            callable_obj=callable_obj,
            func_ref=func_ref,
            args=args,
            node=node
        )
        self._scopes[-1].defers.append(defer_info)
        
        logger.debug(f"Registered defer at scope depth {self.current_depth}")
    
    def get_loop_targets(self) -> Tuple[Optional[Any], Optional[Any]]:
        """Get the nearest loop's continue and break targets
        
        Returns:
            (continue_target, break_target) or (None, None) if not in a loop
        """
        for scope in reversed(self._scopes):
            if scope.scope_type == ScopeType.LOOP:
                return (scope.continue_target, scope.break_target)
        return (None, None)
    
    def get_loop_scope_depth(self) -> Optional[int]:
        """Get the depth of the nearest enclosing loop scope"""
        for scope in reversed(self._scopes):
            if scope.scope_type == ScopeType.LOOP:
                return scope.depth
        return None
    
    def is_in_loop(self) -> bool:
        """Check if currently inside a loop"""
        return self.get_loop_scope_depth() is not None
    
    def declare_variable(self, var_info: Any, allow_shadow: bool = False):
        """Declare a variable in the current scope
        
        This syncs with var_registry and tracks in scope.
        """
        self._var_registry.declare(var_info, allow_shadow=allow_shadow)
        if self._scopes:
            self._scopes[-1].variables[var_info.name] = var_info
    
    def lookup_variable(self, name: str) -> Optional[Any]:
        """Look up a variable in the scope chain"""
        return self._var_registry.lookup(name)
    
    def scope(self, scope_type: ScopeType, cf: Any = None,
              continue_target: Any = None, break_target: Any = None,
              node: Optional[ast.AST] = None):
        """Context manager for automatic scope management
        
        Usage:
            with scope_manager.scope(ScopeType.IF, cf, node=if_node) as scope:
                # ... code ...
            # Defers executed, scope cleaned up
        
        Args:
            scope_type: Type of scope (IF, LOOP, etc.)
            cf: ControlFlowBuilder for termination checks
            continue_target: Target block for continue statements
            break_target: Target block for break statements
            node: AST node for error reporting
        """
        return _ScopeContext(self, scope_type, cf, continue_target, break_target, node)
    
    def clear(self):
        """Clear all scopes (for testing or function end)"""
        self._scopes.clear()


class _ScopeContext:
    """Context manager for ScopeManager.scope()"""
    
    def __init__(self, manager: ScopeManager, scope_type: ScopeType, 
                 cf: Any, continue_target: Any, break_target: Any,
                 node: Optional[ast.AST] = None):
        self._manager = manager
        self._scope_type = scope_type
        self._cf = cf
        self._continue_target = continue_target
        self._break_target = break_target
        self._scope: Optional[Scope] = None
        self._node = node
    
    def __enter__(self) -> Scope:
        self._scope = self._manager.enter_scope(
            self._scope_type,
            continue_target=self._continue_target,
            break_target=self._break_target
        )
        return self._scope
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cf is not None:
            self._manager.exit_scope(self._cf, node=self._node)
        else:
            # No CF provided, just pop scope without defer execution
            if self._manager._scopes:
                self._manager._scopes.pop()
                self._manager._var_registry.exit_scope()
        return False
