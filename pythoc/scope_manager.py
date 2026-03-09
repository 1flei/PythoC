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
    """Unified scope manager for variables, defers, and loop context

    This replaces the separate mechanisms:
    - var_registry.enter_scope() / exit_scope()
    - scope_depth tracking
    - _defer_stack management
    - loop_stack / loop_scope_stack

    Usage:
        with scope_manager.scope(ScopeType.IF) as scope:
            # Variables, defers automatically managed
            ...
        # Defers executed, scope cleaned up
    """
    
    def __init__(self):
        """Initialize scope manager with its own VariableRegistry"""
        from .registry import VariableRegistry
        self._var_registry = VariableRegistry()
        self._scopes: List[Scope] = []
        self._visitor: Optional[Any] = None  # Set by visitor

        # Label tracking state (single source of truth)
        self._scoped_label_stack: List[Any] = []    # Active labels (nested hierarchy)
        self._scope_labels: Dict[int, List[Any]] = {}  # parent_depth -> [LabelContext]
        self._all_labels: Dict[str, Any] = {}        # name -> LabelContext
        self._pending_scoped_gotos: List[Any] = []   # Forward goto refs
    
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
    
    def exit_scope(self, cf: Any, node: Optional[ast.AST] = None) -> Scope:
        """Exit current scope
        
        This:
        1. Emits deferred calls for pending forward gotos from this scope
           (MUST happen before var_registry.exit_scope() so vars are still accessible)
        2. Emits deferred calls for normal exit (if block not terminated)
        3. Removes variables from registry
        
        Args:
            cf: ControlFlowBuilder for checking termination
            node: Optional AST node for error reporting
        
        Returns:
            The exited Scope
        """
        if not self._scopes:
            raise RuntimeError("Cannot exit scope: no scope to exit")
        
        scope = self._scopes.pop()
        
        try:
            # 1. Emit defers for pending forward gotos from this scope
            self._emit_defers_for_pending_gotos(scope, cf)
            
            # 2. Emit deferred calls for normal exit (if not terminated)
            if not cf.is_terminated():
                self._emit_defers_for_scope(scope)
        finally:
            # 4. Sync with var_registry (must happen even if checks raise)
            self._var_registry.exit_scope()
        
        logger.debug(f"Exited scope {scope}")
        return scope
    
    def _emit_defers_for_pending_gotos(self, scope: Scope, cf: Any):
        """Emit defers for pending forward gotos that originate from this scope.
        
        When exiting a scope, any pending forward goto that was issued from within
        this scope MUST be exiting this scope (since the label hasn't been defined yet).
        We emit the defers for this scope at the goto point NOW, while variables are
        still in the registry.
        
        This is the key insight: if goto("forward") is inside scope S, and we're
        exiting S without having seen label("forward"), then the goto MUST be
        jumping out of S. So we emit S's defers at the goto point.
        
        Args:
            scope: The scope being exited
            cf: ControlFlowBuilder
        """
        if not self._visitor:
            return

        if not self._pending_scoped_gotos:
            return

        from pythoc.builtin_entities.defer import _execute_single_defer

        scope_depth = scope.depth

        for pending in self._pending_scoped_gotos:
            # Only process gotos that originate from this scope or deeper
            # If goto_scope_depth >= scope_depth, the goto is from within this scope
            if pending.goto_scope_depth < scope_depth:
                continue
            
            # Skip if we've already emitted defers for this scope depth
            # (emitted_to_depth tracks the minimum depth we've emitted to)
            if pending.emitted_to_depth <= scope_depth:
                continue
            
            # Find defers in the snapshot that belong to this scope
            defers_for_this_scope = [
                (callable_obj, func_ref, args, defer_node)
                for depth, callable_obj, func_ref, args, defer_node in pending.defer_snapshot
                if depth == scope_depth
            ]
            
            if not defers_for_this_scope:
                # Still update emitted_to_depth even if no defers
                pending.emitted_to_depth = scope_depth
                continue
            
            logger.debug(f"Emitting {len(defers_for_this_scope)} defers for pending goto "
                        f"'{pending.label_name}' at scope depth {scope_depth}")
            
            # Save current position
            saved_block = self._visitor.builder.block
            saved_cfg_block_id = cf._current_block_id
            
            # Position at pending block for both IR and CFG
            # Note: pending.block may have been updated by previous defer emissions
            pending_block_id = cf._get_cfg_block_id(pending.block)
            cf.position_at_end(pending.block)
            
            # Temporarily clear terminated flag so defer can emit IR
            was_terminated = cf._terminated.get(pending_block_id, False)
            cf._terminated[pending_block_id] = False
            
            # Emit defers for this scope
            for callable_obj, func_ref, args, defer_node in defers_for_this_scope:
                _execute_single_defer(self._visitor, callable_obj, func_ref, args, defer_node)
            
            # Update emitted_to_depth to track that we've emitted defers for this scope
            pending.emitted_to_depth = scope_depth
            
            # CRITICAL: Update pending.block to current block after defer execution
            # Defer execution may create new blocks (e.g., closure inlining with labels)
            # The final branch to target label should be from the current block, not
            # the original pending block which may now be terminated.
            pending.block = self._visitor.builder.block
            
            # Restore terminated flag for original block
            cf._terminated[pending_block_id] = was_terminated
            
            # Restore position for both IR and CFG
            cf._current_block_id = saved_cfg_block_id
            self._visitor.builder.position_at_end(saved_block)
    
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
        
        from pythoc.builtin_entities.defer import _execute_single_defer
        
        cf = self._visitor._get_cf_builder()
        
        # Do NOT clear defers - they may be needed by other code paths
        defers_to_emit = scope.defers[:]
        
        executed_count = 0
        for defer_info in defers_to_emit:
            if cf.is_terminated():
                logger.debug(f"  cf is terminated, stopping defer emit")
                break
            
            logger.debug(f"  emitting defer: {defer_info.callable_obj}")
            
            # Use the unified defer execution function
            _execute_single_defer(
                self._visitor,
                defer_info.callable_obj,
                defer_info.func_ref,
                defer_info.args,
                defer_info.node
            )
            executed_count += 1
        
        logger.debug(f"Emitted {executed_count}/{len(defers_to_emit)} defers for scope depth {scope.depth}")
    
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

    def is_declared_in_current_scope(self, name: str) -> bool:
        """Check if variable is declared in the current scope"""
        return self._var_registry.is_declared_in_current_scope(name)

    def get_all_in_current_scope(self) -> List[Any]:
        """Get all variables in the current scope"""
        return self._var_registry.get_all_in_current_scope()

    @property
    def scopes(self):
        """Access underlying scope stack (for debug logging)"""
        return self._var_registry.scopes

    def get_all_visible(self) -> Dict[str, Any]:
        """Get all currently visible variables (from all scopes)"""
        return self._var_registry.get_all_visible()

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
    
    # ========================================================================
    # Label Tracking Methods
    # ========================================================================

    def reset_label_tracking(self):
        """Reset label tracking at the start of each function."""
        self._scoped_label_stack = []
        self._scope_labels = {}
        self._all_labels = {}
        self._pending_scoped_gotos = []

    def register_label(self, ctx: Any):
        """Register a label in all tracking structures.

        Args:
            ctx: LabelContext to register
        """
        self._scoped_label_stack.append(ctx)
        self._all_labels[ctx.name] = ctx
        parent_depth = ctx.parent_scope_depth
        if parent_depth not in self._scope_labels:
            self._scope_labels[parent_depth] = []
        self._scope_labels[parent_depth].append(ctx)

    def unregister_label(self, ctx: Any):
        """Pop label from the active label stack.

        Keeps it in _all_labels and _scope_labels for sibling access.

        Args:
            ctx: LabelContext to unregister
        """
        if self._scoped_label_stack and self._scoped_label_stack[-1].name == ctx.name:
            self._scoped_label_stack.pop()

    def find_label(self, name: str) -> Optional[Any]:
        """Find label for goto. Checks ancestors and siblings/uncles.

        goto can target:
        - Self (inside the label)
        - Ancestors (containing labels)
        - Siblings (labels at same parent_depth)
        - Uncles (ancestor's siblings)

        Returns:
            LabelContext if found and visible, None otherwise
        """
        # 1. Check ancestor chain (including self)
        for ctx in self._scoped_label_stack:
            if ctx.name == name:
                return ctx

        # 2. Check siblings and uncles
        ancestor_depths = {0}  # Function level always included
        for ctx in self._scoped_label_stack:
            ancestor_depths.add(ctx.parent_scope_depth)

        for depth in ancestor_depths:
            if depth in self._scope_labels:
                for ctx in self._scope_labels[depth]:
                    if ctx.name == name:
                        return ctx

        return None

    def find_label_for_end(self, name: str) -> Optional[Any]:
        """Find label for goto_end. Only checks ancestors (must be inside target).

        goto_end can ONLY target:
        - Self (inside the label)
        - Ancestors (containing labels)

        Returns:
            LabelContext if found in ancestor chain, None otherwise
        """
        for ctx in self._scoped_label_stack:
            if ctx.name == name:
                return ctx
        return None

    def is_ancestor_label(self, ctx: Any) -> bool:
        """Check if a label is in the current ancestor chain"""
        return ctx in self._scoped_label_stack

    def capture_defer_snapshot(self) -> List:
        """Capture current defer state for forward goto resolution.

        Returns a list of (depth, callable_obj, func_ref, args, node) tuples
        representing all defers currently registered across all scopes.
        """
        snapshot = []
        current_depth = self.current_depth
        for scope in self._scopes:
            if scope.depth <= current_depth:
                for defer_info in scope.defers:
                    snapshot.append((
                        scope.depth,
                        defer_info.callable_obj,
                        defer_info.func_ref,
                        defer_info.args,
                        defer_info.node
                    ))
        return snapshot

    def resolve_pending_gotos_for_label(self, ctx: Any):
        """Resolve pending forward goto references when a label is defined.

        When a label is entered, any forward gotos targeting it can now be
        resolved. This emits remaining defers and generates IR branches.

        Args:
            ctx: LabelContext being entered
        """
        if not self._visitor:
            return

        from pythoc.builtin_entities.defer import _execute_single_defer

        cf = self._visitor._get_cf_builder()
        label_name = ctx.name
        target_parent = ctx.parent_scope_depth

        resolved = []

        for pending in self._pending_scoped_gotos:
            if pending.label_name != label_name:
                continue

            if pending.is_goto_end:
                logger.error(f"Internal error: forward goto_end to '{label_name}'",
                            node=pending.node, exc_type=RuntimeError)
                continue

            # Save current position
            saved_block = self._visitor.builder.block
            saved_cfg_block_id = cf._current_block_id

            pending_block_id = cf._get_cfg_block_id(pending.block)

            # Position at pending block for both IR and CFG
            cf.position_at_end(pending.block)

            # Temporarily clear terminated flag so defer can emit IR
            was_terminated = cf._terminated.get(pending_block_id, False)
            cf._terminated[pending_block_id] = False

            # Emit defers that haven't been emitted yet:
            # target_parent < depth < emitted_to_depth
            defers_to_execute = [
                (callable_obj, func_ref, args, defer_node)
                for depth, callable_obj, func_ref, args, defer_node in pending.defer_snapshot
                if target_parent < depth < pending.emitted_to_depth
            ]
            logger.debug(f"Resolving forward goto '{pending.label_name}': "
                        f"executing {len(defers_to_execute)} defers "
                        f"(goto_scope={pending.goto_scope_depth}, "
                        f"target_parent={target_parent}, "
                        f"emitted_to_depth={pending.emitted_to_depth})")

            for callable_obj, func_ref, args, defer_node in defers_to_execute:
                _execute_single_defer(self._visitor, callable_obj, func_ref, args, defer_node)

            # Generate the branch instruction
            self._visitor.builder.branch(ctx.begin_block)

            # Restore terminated flag
            cf._terminated[pending_block_id] = was_terminated

            # Restore position for both IR and CFG
            cf._current_block_id = saved_cfg_block_id
            self._visitor.builder.position_at_end(saved_block)

            # Add CFG edge for the goto
            begin_block_id = cf._get_cfg_block_id(ctx.begin_block)
            cf.cfg.add_edge(pending_block_id, begin_block_id, kind='goto')

            resolved.append(pending)

        for item in resolved:
            self._pending_scoped_gotos.remove(item)

    def add_pending_goto(self, pending: Any):
        """Add a pending forward goto reference."""
        self._pending_scoped_gotos.append(pending)

    def resolve_pending_gotos(self, label_name: str) -> List[Any]:
        """Get and remove pending gotos for a given label name.

        Returns:
            List of resolved PendingGoto objects
        """
        resolved = [p for p in self._pending_scoped_gotos if p.label_name == label_name]
        for item in resolved:
            self._pending_scoped_gotos.remove(item)
        return resolved

    def check_goto_consistency(self, func_node: Any):
        """Check that all scoped gotos have been resolved.

        Called at the end of function compilation.

        Args:
            func_node: The function AST node (for error location)
        """
        if self._pending_scoped_gotos:
            for pending in self._pending_scoped_gotos:
                goto_type = "goto_end" if pending.is_goto_end else "goto"
                logger.error(f"Undefined label '{pending.label_name}' in {goto_type} statement",
                            node=pending.node, exc_type=SyntaxError)

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
